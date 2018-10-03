import tensorflow as tf
import graph_manager
import tf_utils
import tfrecords_utils
import viz


def load_image(im_id, image_size, image_folder, image_format):
    """Resolve the correct image path from the given arguments.
    
    Args:
        im_id: image id saved in the tfrecords
        image_size: integer specifying the square size to resize the image to
        image_folder: image folder path
        image_format: Used to resolve the correct image path and format
    
    Returns:
        The loaded image as a 3D Tensor
    """
    if image_format == 'vedai':     # VEDAI
        filename = image_folder  + '/' + tf.as_string(im_id, fill='0', width=8) + '_co.png'
        type = 'png'
    elif image_format == 'sdd':     # STANFORD DRONE DATASET
        filename = image_folder  + '/' + tf.as_string(im_id, fill='0', width=8) + '.jpeg'
        type = 'jpg'
    elif image_format == 'dota':    # DOTA
        filename = image_folder  +  '/' + tf.as_string(im_id, fill='0', width=7) + '.jpg'
        type = 'jpg'
    else:
        raise NotImplementedError("Unrecognized image format `%s`" % image_format)

    # Parse image
    image = tf.read_file(filename)
    if type == 'jpg':
        image = tf.image.decode_jpeg(image, channels=3)
    elif type == 'png':
        image = tf.image.decode_png(image, channels=3)
    else:
        raise NotImplementedError('unknown image type %s' % type)
    image = tf.image.convert_image_dtype(image, tf.float32)    
    
    # Resize image
    image = tf.image.resize_images(image, (image_size, image_size))
    return image


def parse_basic_feature(parsed_features, image_folder, image_format, image_size=448):
    """"Parse TFRecords features.
    
    Args:
        parsed_features: Parsed TFRecords features.
        num_classes: Number of classes in the dataset. Used to infer the dataset.
        image_folder: Image directory.
        image_format: Used to resolve the correct image path and format.
        image_size: Resize to the given image size. Defaults to 448.
        
    Returns:
        image_id, an integer (exact format depends on the dataset)
        image, Tensor with values in [0, 1], shape (image_size, image_size, 3)
        num_boxes, Number of valid boxes for this image
        bounding_boxes, Bounding boxes for this image, shape (max_num_bbs, 4)
    """
    im_id = tf.cast(parsed_features['im_id'], tf.int32)  
    image = load_image(im_id, image_size, image_folder, image_format)        
    num_boxes = tf.cast(parsed_features['num_boxes'], tf.int32)
    bounding_boxes = parsed_features["bounding_boxes"]
    return {'im_id': im_id, 'image': image, 'num_boxes': num_boxes, 'bounding_boxes': bounding_boxes}           


def apply_data_augmentation(in_, num_samples, data_augmentation_threshold):
    """ Perform data augmentation (left/right flip).
    
    Args:
        in_: A batch from the dataset (output of iterator.get_next()).
        num_samples:  batch size
        keys: Inputs dictionnary keys  
        data_augmentation_threshold: threshold in [0, 1]
        
    Returns:
        Dataset with left/right data augmentation applied
    """
    condition_shape = tf.shape(in_['image'])[:1]
    condition = (tf.random_uniform(condition_shape) >= data_augmentation_threshold)
        
    # Flip image
    in_['image'] = tf.where(condition, in_['image'], tf.reverse(in_['image'], [2]))
    
    # Set is_flipped flag
    in_['is_flipped'] = tf.where(condition, in_['is_flipped'], 1. - in_['is_flipped'])
    
    # Flip bounding boxes coordinates, (batch, num_bbs, 4)
    in_['bounding_boxes'] = tf.where(condition, in_['bounding_boxes'], 
                                     tf.abs([1., 0., 1., 0.] - tf.gather(in_['bounding_boxes'], [2, 1, 0, 3], axis=-1)))
        
    # Flip active/empty cell mask, (batch, num_cells_x, num_cells_y, 1, num_bbs)
    in_['obj_i_mask_bbs'] = tf.where(condition, in_['obj_i_mask_bbs'], tf.reverse(in_['obj_i_mask_bbs'], [2]))

    # Flip groups bounding boxes coordinates, (batch, num_cells, num_cells, 1, 4)
    if 'group_bounding_boxes_per_cell' in in_:
        in_['group_bounding_boxes_per_cell'] = tf.where(
            condition, in_['group_bounding_boxes_per_cell'], tf.abs([1., 0., 1., 0.] - tf.gather(
            tf.reverse(in_['group_bounding_boxes_per_cell'], [2]), [2, 1, 0, 3], axis=-1)))
        
    # Flip groups ground-truth flags, (batch, num_cells, num_cells, 1, 1)
    if 'group_flags' in in_:
        in_['group_flags'] = tf.where(condition, in_['group_flags'], tf.reverse(in_['group_flags'], [2]))
        
    # Flip groups classes, (batch, num_cells, num_cells, 1, num_classes)
    if 'group_class_labels' in in_:
        in_['group_class_labels'] = tf.where(condition, in_['group_class_labels'],
                                             tf.reverse(in_['group_class_labels'], [2]))
        
    # Return
    return in_


def get_tf_dataset(tfrecords_file,
                   record_keys,
                   image_format,
                   max_num_bbs,
                   with_classes=False,
                   num_classes=None,
                   with_groups=True,
                   grid_offsets=None,
                   batch_size=1,
                   num_epochs=1,
                   image_size=448,
                   image_folder='',
                   data_augmentation_threshold=0.5,
                   num_shards=1,
                   shard_index=0,
                   num_threads=4,
                   shuffle_buffer=1,
                   prefetch_capacity=1,
                   make_initializable_iterator=False,
                   verbose=1):
    """Returns a queue containing the inputs batches.

    Args:
      tfrecords_file: Path to the TFRecords file containing the data.
      record_keys: Feature keys present in the TFrecords. Loaded from the metadata file
      max_num_bbs: Maximum number of bounding boxes in the dataset. Used for reshaping the `bounding_boxes` records.   
      num_classes: Number of classes in the dataset. Only used if with-classes is True
      with_classes: wheter to use class information
      with_groups: whether to pre-compute grouped instances ground-truth
      grid_offsets: Precomputed grid offsets 
      batch_size: Batch size.
      num_epochs: Number of epochs to repeat.
      image_size: The square size which to resize images to.
      image_folder: path to the directory containing the images in the dataset.
      data_augmentation_threshold: Data augmentation probabilitiy (in [0, 1])
      num_shards: Number of shards. Each shard gets a data batch of size "batch_size"
      shard_index: Index of the sard, default to 0
      num_threads: Number of readers for the batch queue.
      subset: If positive, extract the given number of samples as a subset of the dataset
      shuffle_buffer: Size of the shuffling buffer.
      prefetch_capacity: Buffer size for prefetching.
      make_initializable_iterator: if True, make an initializable and add its initializer to the collection `iterator_init`
      verbose: Verbosity level

    Returns: 
      A tf.Data.dataset iterator (and its initializer if initializable_iterator)
    """
    # Asserts
    assert not (with_classes and num_classes is None)
    assert len(record_keys)
    assert batch_size > 0
    assert image_size > 0
    assert 0. <= data_augmentation_threshold <= 1.
    if grid_offsets is not None:
        num_cells = grid_offsets.shape[:2]
    assert num_shards > 0
    assert shard_index >= 0
    assert shard_index < num_shards
    assert num_threads > 0
    assert shuffle_buffer > 0
    
    if verbose == 2:
        print(' \033[31m> load_inputs\033[0m')
    elif verbose == 1:
        print(' > load_inputs')
    # Normalize grid cells offsets
    if grid_offsets is not None:
        grid_offsets_mins = grid_offsets / num_cells
        grid_offsets_maxs = (grid_offsets + 1.) / num_cells 
    
    # Create TFRecords feature
    features = tfrecords_utils.read_tfrecords(record_keys, max_num_bbs=max_num_bbs)
    
    # Preprocess
    def parsing_function(example_proto):
        # Basic features
        parsed_features = tf.parse_single_example(example_proto, features)
        output = parse_basic_feature(parsed_features, image_folder, image_format, image_size=image_size)
        bounding_boxes = output['bounding_boxes']
        
        # Empty/active cells mask
        # obj_i_mask_bbs: (num_cells, num_cells, 1, num_bbs)
        mins, maxs = tf.split(bounding_boxes, 2, axis=-1) # (num_bbs, 2)
        inters = tf.maximum(0., tf.minimum(maxs, grid_offsets_maxs) - tf.maximum(mins, grid_offsets_mins))
        inters = tf.reduce_prod(inters, axis=-1)
        obj_i_mask = tf.expand_dims(tf.to_float(inters > 0.) , axis=-2)
        output["obj_i_mask_bbs"] = obj_i_mask
                    
        # Grouped instances 
        # group_bounding_boxes_per_cell: (num_cells, num_cells, 1, 4), cell bounding box after grouping
        # group_flags: (num_cells, num_cells, 1, 1), whether a cell contains a group or not
        # num_group_boxes: (), number of bounding boxes after grouping
        if with_groups:
            obj_i_mask = tf.transpose(obj_i_mask, (0, 1, 3, 2)) # (num_cells, num_cells, num_bbs, 1)
            mins = mins + 1. - obj_i_mask 
            mins = tf.reduce_min(mins, axis=2, keepdims=True) # (num_cells, num_cells, 1, 2)
            maxs = maxs * obj_i_mask
            maxs = tf.reduce_max(maxs, axis=2, keepdims=True)
            group_bounding_boxes_per_cell = tf.concat([mins, maxs], axis=-1)
            group_bounding_boxes_per_cell = tf.clip_by_value(group_bounding_boxes_per_cell, 0., 1.)
            output["group_bounding_boxes_per_cell"] = group_bounding_boxes_per_cell
            
            num_bbs_per_cell = tf.reduce_sum(obj_i_mask, axis=2, keepdims=True)
            num_group_boxes = tf.reduce_sum(tf.to_int32(num_bbs_per_cell > 0))
            output["num_group_boxes"] = num_group_boxes
            
            group_flags = tf.maximum(tf.minimum(num_bbs_per_cell, 2.) - 1., 0.)
            output["group_flags"] = group_flags
          
        # is_flipped flag: (), indicates whether the image has been flipped during data augmentation
        output["is_flipped"] = tf.constant(0.)
            
        # Optional : add classes
        if with_classes:            
            class_labels = tf.one_hot(parsed_features['classes'], num_classes, 
                                      axis=-1, on_value=1, off_value=0, dtype=tf.int32)
            output['class_labels'] = class_labels
            
            # Group classes (majority vote) # (num_cells, num_cells, 1, num_classes)
            if with_groups:
                percell_class_labels = tf.expand_dims(tf.expand_dims(class_labels, axis=0), axis=0)
                percell_class_labels = obj_i_mask * tf.to_float(percell_class_labels)
                percell_class_labels = tf.reduce_sum(percell_class_labels, axis=2, keepdims=True)
                group_class_labels = tf.argmax(percell_class_labels, axis=-1)
                group_class_labels = tf.one_hot(group_class_labels, num_classes,
                                                axis=-1, on_value=1, off_value=0, dtype=tf.int32)
                group_class_labels = tf.to_int32(percell_class_labels * tf.to_float(group_class_labels))
                output["group_class_labels"] = group_class_labels
        return output
                    
        
    ## Create the dataset
    with tf.name_scope('load_dataset'):
        # Parse data
        dataset = tf.data.TFRecordDataset(tfrecords_file)
        # Shard
        if num_shards > 1:
            dataset = dataset.shard(num_shards, shard_index)       
        # Map
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.map(parsing_function, num_parallel_calls=num_threads)
        # Repeat
        if num_epochs > 1:
            dataset = dataset.repeat(num_epochs)
        # Batch
        dataset = dataset.batch(batch_size)
        if prefetch_capacity > 0: 
            dataset = dataset.prefetch(prefetch_capacity)
        # Iterator
        if make_initializable_iterator:
            iterator = dataset.make_initializable_iterator()
            iterator_init = iterator.initializer
            tf.add_to_collection('iterator_init', iterator_init)
        else:
            iterator = dataset.make_one_shot_iterator()    
            iterator_init = None
        in_ = iterator.get_next()
        
    ## Data augmentation
    with tf.name_scope('data_augmentation'):
        if data_augmentation_threshold > 0.:
            apply_data_augmentation(in_, batch_size, data_augmentation_threshold)
              
    ## Verbose log
    if verbose == 2:
        print('\n'.join("    \033[32m%s\033[0m: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in in_.items()))
    elif verbose == 1:
        print('\n'.join("    *%s*: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in in_.items()))
    return in_, iterator_init


def filter_individuals(predicted_boxes, predicted_scores, predicted_group_flags, strong_confidence_threshold=1.0):
    """Filter out individuals predictions with confidence higher than the given threhsold"""
    # is_group: (batch, num_boxes, 1)
    is_group = tf.to_float(tf.nn.sigmoid(predicted_group_flags) > 0.5)
    is_group = tf_utils.flatten_percell_output(is_group)
    # should_be_refined: (batch, num_boxes, 1) : groups and not strongly confident individuals
    is_not_strongly_confident = tf.to_float(predicted_scores <= strong_confidence_threshold)
    # Filter them out from potential crops
    should_be_refined = tf.minimum(1., is_group + is_not_strongly_confident)
    predicted_scores *= should_be_refined
    predicted_boxes *= should_be_refined
    # Return filtered boxes and filter
    return predicted_boxes, predicted_scores, tf.squeeze(1. - should_be_refined, axis=-1)


def filter_threshold(predicted_boxes, predicted_scores, confidence_threshold=-1.):
    """Filter out boxes with confidence below the given threshold"""
    filtered = tf.to_float(predicted_scores > confidence_threshold)
    predicted_scores *= filtered
    predicted_boxes *= filtered
    return predicted_boxes, predicted_scores


def extract_groups(inputs, 
                   predicted_boxes,
                   predicted_scores,
                   predicted_group_flags=None,
                   predicted_offsets=None,
                   mode='train',
                   verbose=False,
                   epsilon=1e-8,
                   **kwargs): 
    """ Extract crops from the outputs of intermediate stage.
    
    Args:
        inputs: Inputs dictionnary of stage s
        predicted_boxes: A (batch_size, num_cells, num_cells, num_boxes, 4) array
        predicted_scores: A (batch_size, num_cells, num_cells, num_boxes, 1) array
        predicted_group_flags: A (batch_size, num_cells, num_cells, num_boxes, 1) array
        predicted_offsets: A (batch_size, num_cells, num_cells, num_boxes, 2) array
        mode: If test, the boxes are only passed to the next stage if they are worth being refined 
            (ie groups or unprecise individual)
        
    Kwargs:
        {train, test}_patch_confidence_threshold: Minimum confidene threshold to qualify for refinement
        patch_nms_threshold: NMS threshold
        {train, test}_num_crops: Number of crops to extract
        test_patch_strong_confidence_threshold: high confidence threshold
        
    #Returns:
        Extracted crops and their confidence scores
    """
    if mode == 'train':
        (confidence_threshold, nms_threshold, num_outputs) = graph_manager.get_defaults(
            kwargs, ['train_patch_confidence_threshold', 'train_patch_nms_threshold', 'train_num_crops'], verbose=verbose)
    elif mode in ['val', 'test']:
        (confidence_threshold, nms_threshold, num_outputs) = graph_manager.get_defaults(
            kwargs, ['test_patch_confidence_threshold', 'test_patch_nms_threshold', 'test_num_crops'], verbose=verbose)
    else:
        raise ValueError('Unknown mode', mode)
    if verbose:        
        print('  > extracting %d crops' % num_outputs)
        
    ## Outputs
    kept_out_filter = None
        
    ## Flatten
    # predicted_score: (batch, num_boxes, 1)
    # predicted_boxes: (batch, num_boxes, 4)
    with tf.name_scope('flat_output'):
        predicted_boxes = tf_utils.flatten_percell_output(predicted_boxes)
        predicted_scores = tf_utils.flatten_percell_output(predicted_scores)
        
    ## Filter
    with tf.name_scope('filter_groups'):
        # At test time, we keep out individual confidences with high confidence
        if mode in ['test', 'val'] and predicted_group_flags is not None:
            strong_confidence_threshold = graph_manager.get_defaults(
                kwargs, ['test_patch_strong_confidence_threshold'], verbose=verbose)[0]
            predicted_boxes, predicted_scores, kept_out_filter = filter_individuals(
                predicted_boxes, predicted_scores, predicted_group_flags, strong_confidence_threshold)
        
        # Additionally, we filter out boxes with confidence below the threshold
        if confidence_threshold > 0.:
            with tf.name_scope('filter_confidence'):
                predicted_boxes, predicted_scores = filter_threshold(
                    predicted_boxes, predicted_scores, confidence_threshold)
        
    ## Rescale remaining  boxes with the learned offsets
    with tf.name_scope('offsets_rescale_boxes'):
        if predicted_offsets is not None:
            predicted_boxes = tf_utils.rescale_with_offsets(
                predicted_boxes, tf_utils.flatten_percell_output(predicted_offsets), epsilon)
    
    ## Extract n best patches
    # crop_boxes: (batch, num_crops, 4)
    # crop_boxes_confidences: (batch, num_crops)
    predicted_scores = tf.squeeze(predicted_scores, axis=-1)
    if num_outputs > 0:    
        # Non-Maximum Suppression
        if nms_threshold < 1.0:
            batch_size = graph_manager.get_defaults(kwargs, ['batch_size'], verbose=verbose)[0]
            current_batch = tf.shape(inputs['image'])[0]
            with tf.name_scope('nms'):
                nms_boxes = []
                nms_boxes_confidences = []
                for i in range(batch_size):
                    boxes, scores = tf.cond(
                        i < current_batch, # last batch can be smaller  
                        true_fn=lambda:tf_utils.nms_with_pad(predicted_boxes[i, :, :], 
                                                             predicted_scores[i, :],
                                                             num_outputs, 
                                                             iou_threshold=nms_threshold),
                        false_fn=lambda: (tf.zeros((num_outputs, 4)), tf.zeros((num_outputs,))) 
                    )
                    nms_boxes.append(boxes)
                    nms_boxes_confidences.append(scores)
                # Reshape nms boxes output
                predicted_boxes = tf.stack(nms_boxes, axis=0) 
                predicted_boxes = tf.slice(predicted_boxes, (0, 0, 0), (current_batch, -1, -1))
                predicted_boxes = tf.reshape(predicted_boxes, (-1, num_outputs, 4))
                # Reshape nms scores output
                predicted_scores = tf.stack(nms_boxes_confidences, axis=0) 
                predicted_scores = tf.slice(predicted_scores, (0, 0), (current_batch, -1))
                predicted_scores = tf.reshape(predicted_scores, (-1, num_outputs))
        # No NMS
        else:
            predicted_scores, top_indices = tf.nn.top_k(predicted_scores, k=num_outputs)
            batch_indices = tf.range(tf.shape(predicted_boxes)[0])
            batch_indices = tf.tile(tf.expand_dims(batch_indices, axis=-1), (1, num_outputs))
            gather_indices = tf.stack([batch_indices, top_indices], axis=-1)
            predicted_boxes = tf.gather_nd(predicted_boxes, gather_indices)
    # No filtering 
    return predicted_boxes, predicted_scores, kept_out_filter


def tile_and_reshape(t, num_crops):
    """ Given an initial Tensor `t` of shape (batch_size, s1...sn), tile and reshape it to size 
        (batch_size * `num_crops`, s1..sn) to be forwarded to the next stage input.
        Note that s1...sn should be a *fully defined* shape.
    """
    new_shape = t.get_shape().as_list()
    new_shape[0] = -1
    t = tf.expand_dims(t, axis=1)
    tile_pattern = [1] * len(t.get_shape())
    tile_pattern[1] = num_crops
    t = tf.tile(t, tile_pattern)
    assert not None in new_shape
    t = tf.reshape(t, new_shape)
    return t


def get_next_stage_inputs(inputs, 
                          crop_boxes,
                          batch_size=None,
                          image_size=256,
                          previous_batch_size=None,
                          full_image_size=1024,
                          image_folder=None,
                          image_format=None,
                          grid_offsets=None,
                          intersection_ratio_threshold=0.25,
                          epsilon=1e-8,
                          use_queue=False,
                          shuffle_buffer=1,
                          num_threads=1,
                          capacity=5000,
                          verbose=False):
    """
    Create input queue for the second - and final - stage.
    Args:
        inputs, a dictionnary of inputs
        crop_boxes, a (batch_size, num_crops, 4) tensor of crops
        image_folder: Image directory, used for reloading the full resolution images if needed
        batch_size: Batch size for the output of this pipeline
        image_size: Size of the images patches in the new dataset
        full_image_size: Size of the images to load before applying the croppings
        grid_offsets: A (num_cells, num_cells) array
        use_queue: Whether to use a queue or directly output the new inputs dictionary
        shuffle_buffer: shuffle buffer of the output queue
        num_threads: number of readers in the output queue
        capacity: Output queue capacity
        verbose: verbosity        
    """
    assert 0. <= intersection_ratio_threshold < 1.
    num_crops = crop_boxes.get_shape()[1].value   
    new_inputs = {}
    
    # new_im_id: (batch_size * num_crops,)
    if 'im_id' in inputs:
        with tf.name_scope('im_ids'):
            new_inputs['im_id'] = tile_and_reshape(inputs['im_id'], num_crops)        
        
    # classes: (batch_size * num_crops, num_classes)
    if 'class_labels' in inputs:
        with tf.name_scope('class_labels'):
            new_inputs['class_labels'] = tile_and_reshape(inputs['class_labels'], num_crops)
    
    # new_image: (num_patches, image_size, image_size, 3)
    with tf.name_scope('extract_image_patches'):
        # Re-load full res image (flip if necessary)
        if image_folder is not None and full_image_size > 0:
            full_images = []
            current_batch = tf.shape(inputs['im_id'])[0]
            for i in range(previous_batch_size): 
                image = tf.cond(i < current_batch, # last batch can be smaller                                
                                true_fn=lambda: load_image(inputs['im_id'][i], full_image_size, image_folder, image_format),
                                false_fn=lambda: tf.zeros((full_image_size, full_image_size, 3)))
                full_images.append(image)
            full_images = tf.stack(full_images, axis=0)
            full_images = tf.slice(full_images, (0, 0, 0, 0), (current_batch, -1, -1, -1))
            full_images = tf.reshape(full_images, (-1, full_image_size, full_image_size, 3))
            full_images = tf.where(inputs["is_flipped"] > 0., tf.reverse(full_images, [2]), full_images)
        # Otherwise extract patches from the image directly
        else:
            full_images = inputs['image']
        # Extract patches and resize
        # crop_boxes_indices: (batch_size * num_crops,)
        # crop_boxes_flat: (batch_size * num_crops, 4)
        crop_boxes_indices = tf.ones(tf.shape(crop_boxes)[:2], dtype=tf.int32)
        crop_boxes_indices = tf.cumsum(crop_boxes_indices, axis=0, exclusive=True)
        crop_boxes_indices = tf.reshape(crop_boxes_indices, (-1,))
        crop_boxes_flat = tf.gather(tf.reshape(crop_boxes, (-1, 4)), [1, 0, 3, 2], axis=-1)
        new_inputs['image'] = tf.image.crop_and_resize(full_images, crop_boxes_flat, crop_boxes_indices, 
                                                       (image_size, image_size), name='extract_groups')
        
    # new_bounding_boxes: (num_patches, max_num_bbs, 4)
    # rescale bounding boxes coordinates to the cropped image
    if 'bounding_boxes' in inputs:
        with tf.name_scope('shift_bbs'):
            # bounding_boxes: (batch, num_crops, max_num_bbs, 4)
            # crop_boxes: (batch, num_crops, 1, 4)
            bounding_boxes = inputs['bounding_boxes']
            max_num_bbs = bounding_boxes.get_shape()[1].value
            bounding_boxes = tf.expand_dims(bounding_boxes, axis=1)
            bounding_boxes = tf.tile(bounding_boxes, (1, num_crops, 1, 1))
            crop_boxes = tf.expand_dims(crop_boxes, axis=2)
            # Filter out cut bbs
            ratios = tf_utils.get_intersection_ratio(tf.split(bounding_boxes, 4, axis=-1), tf.split(crop_boxes, 4, axis=-1))
            condition = tf.tile(ratios > intersection_ratio_threshold, (1, 1, 1, 4))
            bounding_boxes *= tf.to_float(condition)
            # Rescale coordinates to the cropped image
            crop_mins, crop_maxs = tf.split(crop_boxes, 2, axis=-1)
            bounding_boxes -= tf.tile(crop_mins, (1, 1, 1, 2))
            bounding_boxes /= tf.maximum(epsilon, tf.tile(crop_maxs - crop_mins, (1, 1, 1, 2)))
            bounding_boxes = tf.clip_by_value(bounding_boxes, 0., 1.)
            bounding_boxes = tf.reshape(bounding_boxes, (-1, max_num_bbs, 4))
            new_inputs['bounding_boxes'] = bounding_boxes

    # number of valid boxes: (num_patches,)
    if 'num_boxes' in inputs:
        with tf.name_scope('num_boxes'):
            valid_boxes = ((bounding_boxes[..., 2] > bounding_boxes[..., 0]) & 
                           (bounding_boxes[..., 3] > bounding_boxes[..., 1]))
            num_boxes =  tf.to_float(valid_boxes)
            new_inputs['num_boxes'] = tf.to_int32(tf.reduce_sum(num_boxes, axis=-1) )
        
    # Compute the box presence in cell mask
    # obj_i_mask_bbs: (num_patches, num_cells, num_cells, 1, num_gt)
    if 'obj_i_mask_bbs' in inputs:
        with tf.name_scope('grid_offsets'):
            if grid_offsets is not None:            
                num_cells = grid_offsets.shape[:2]
                grid_offsets_mins = grid_offsets / num_cells
                grid_offsets_maxs = (grid_offsets + 1.) / num_cells      
                bounding_boxes = tf.reshape(bounding_boxes, (-1, 1, 1, max_num_bbs, 4))
                mins, maxs = tf.split(bounding_boxes, 2, axis=-1)
                inters = tf.maximum(0., tf.minimum(maxs, grid_offsets_maxs) - tf.maximum(mins, grid_offsets_mins))
                inters = tf.reduce_prod(inters, axis=-1)
                obj_i_mask = tf.expand_dims(tf.to_float(inters > 0.) , axis=-2)
                new_inputs['obj_i_mask_bbs'] = obj_i_mask
        
    # Enqueue the new inputs during training, or pass the output directly to the next stage at test time
    if use_queue:
        assert batch_size is not None
        filter_valid = tf.logical_and(crop_boxes[..., 2] > crop_boxes[..., 0], crop_boxes[..., 3] > crop_boxes[..., 1] )
        filter_valid = tf.reshape(filter_valid, (-1,))
        out_ = tf.train.maybe_batch(
            new_inputs, filter_valid, batch_size, num_threads=num_threads, enqueue_many=True, capacity=capacity)
    else:
        out_ = new_inputs    
    
    if verbose == 1:
        print('\n'.join("    \033[32m%s\033[0m: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in out_.items() if key != 'batch_size'))
    elif verbose > 1:
        print('\n'.join("    *%s*: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in out_.items() if key != 'batch_size'))
    return out_   