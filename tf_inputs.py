import tensorflow as tf
import graph_manager
import tf_utils
import tfrecords_utils
import viz


def load_image(im_id, num_classes, image_size, image_folder):
    """Resolve the correct image path from the given arguments.
    
    Args:
        im_id: image id saved in the tfrecords
        num_classes: number of classes, used to resolve the dataset
        image_size: integer specifying the square size to resize the image to
        image_folder: image folder path
    
    Returns:
        The loaded image as a 3D Tensor
    """
    if num_classes == 9:     # VEDAI
        filename = image_folder  + '/' + tf.as_string(im_id, fill='0', width=8) + '_co.png'
        type = 'png'
    elif num_classes == 6:   # STANFORD
        filename = image_folder  + '/' + tf.as_string(im_id, fill='0', width=8) + '.jpeg'
        type = 'jpg'
    elif num_classes == 15:  # DOTA
        filename = image_folder  +  '/' + tf.as_string(im_id, fill='0', width=7) + '.jpg'
        type = 'jpg'
    else:
        raise NotImplementedError("Unrecognized dataset (num_classes = %d)" % num_classes)

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


def parse_basic_feature(parsed_features, num_classes, image_folder, image_size=448):
    """"Parse TFRecords features.
    
    Args:
        parsed_features: Parsed TFRecords features.
        num_classes: Number of classes in the dataset. Used to infer the dataset.
        image_folder: Image directory.
        image_size: Resize to the given image size. Defaults to 448.
        
    Returns:
        image_id, an integer (exact format depends on the dataset)
        image, Tensor with values in [0, 1], shape (image_size, image_size, 3)
        num_boxes, Number of valid boxes for this image
        bounding_boxes, Bounding boxes for this image, shape (max_num_bbs, 4)
    """
    im_id = tf.cast(parsed_features['im_id'], tf.int32)  
    image = load_image(im_id, num_classes, image_size, image_folder)        
    num_boxes = tf.cast(parsed_features['num_boxes'], tf.int32)
    bounding_boxes = parsed_features["bounding_boxes"]
    return im_id, image, num_boxes, bounding_boxes           


def get_inputs_keys(record_keys, with_groups=True, with_classes=True):
    """Infer the keys to be added to the inputs dictionnary
    
    Args:
        record_keys: List of keys that are being loaded from the TFRecord
        with_groups: Whether to add group ground-truth
        with_classification: Whether to load class specific information
        
    Returns: 
        List of keys in the inputs dictionnary. Subset of :
            im_id: int () Image ID
            image: float (w, h, 2) Image
            num_boxes: int () Number of valid bounding boxes
            bounding_boxes: float (max_num_bbs, 4) Ground truth individual boxes
            class_labels: int (max_num_bbs, num_classes) Class per individual box
            obj_i_mask_bbs: (num_cells, num_cells, 1, max_num_bbs) mask indicating whether a'
                bounding box is present in a cell
            num_group_boxes: int () Number of valid bounding boxes after grouping.
            group_bounding_boxes_per_cells: float (num_cells, num_cells, 1, 4) per cell bounding boxes with groups.
            group_flags: float (num_cells, num_cells, 1, 1) per cell binary group flag
            group_class_labels: int (num_cells, num_cells, 1, num_classes) per cell class with groups
            is_flipped: float () indicates whether the image was left-right flipped duing data annotations
    """
    # Basic features
    keys = ['im_id', 'image', 'num_boxes', 'bounding_boxes']
    
    if with_classes:
        keys.append('class_labels')
        
    keys.append("obj_i_mask_bbs")
                    
    # Groups computed on the fly
    if with_groups:
        keys.append("num_group_boxes")
        keys.append("group_bounding_boxes_per_cell")
        keys.append("group_flags") 
        if with_classes:      
            keys.append("group_class_labels") 
        
    # Whether the image has been flipped in data augmentation
    keys.append("is_flipped")
    return keys


def get_dummy_dataset(keys,
                      num_repeats, 
                      image_size, 
                      num_classes, 
                      num_cells,
                      max_num_bbs):
    """Create dummy data samples to pad the validation dataset
    
    Args:
        keys: keys present in the inputs dictionnary
        num_repeats: Number of dummy samples to add
        image_size: Size of images in the dataset
        num_classes: Number of classes in the dataset 
        num_cells: Number of cells in the output grid
        max_num_bbs: Maximum number of gt bounding boxes
        
    Returns:
        A dummy dataset with `num_repeats` entry
    """
    assert num_repeats > 0
    in_ = []
    
    # im_id = -1, image, num_boxes, bounding boxes
    in_.append(-1)
    in_.append(tf.zeros((image_size, image_size, 3)))
    in_.append(0)
    in_.append(tf.zeros((max_num_bbs, 4)) + [1., 1., 0., 0.])
    
    # classes
    if "class_labels" in keys:
        in_.append(tf.zeros((max_num_bbs, num_classes), dtype=tf.int32) - 1)
    
    # obj_i_mask_bbs
    in_.append(tf.zeros((num_cells[0], num_cells[1], 1, max_num_bbs)))
        
    # groups
    if "num_group_boxes" in keys:
        in_.append(0)        
    if "group_bounding_boxes_per_cell" in keys:
        in_.append(tf.zeros((num_cells[0], num_cells[1], 1, 4)) + [1., 1., 0., 0.])
    if "group_flags" in keys:
        in_.append(tf.zeros((num_cells[0], num_cells[1], 1, 1)))
    if "group_class_labels" in keys:
        in_.append(- tf.ones((num_cells[0], num_cells[1], 1, num_classes)))
            
    # is_flipped
    in_.append(0.)   
        
    dataset = tf.data.Dataset.from_tensors(tuple(in_))
    dataset = dataset.repeat(num_repeats)
    return dataset


def apply_data_augmentation(in_, num_samples, keys, data_augmentation_threshold):
    """ Perform data augmentation.
    
    Args:
        in_: A batch from the dataset (output of iterator.get_next()).
        num_samples:  batch size
        keys: Inputs dictionnary keys  
        data_augmentation_threshold: threshold in [0, 1]
        
    Returns:
        in_ after data_augmentation applied
    """
    condition = (tf.random_uniform((num_samples,)) > data_augmentation_threshold)
        
    # Flip image
    index = keys.index('image')
    in_[index] = tf.where(condition, in_[index], tf.reverse(in_[index], [2]))
    
    # Add is_flipped flag
    index = keys.index('is_flipped')
    in_[index] = tf.where(condition, in_[index], 1. - in_[index])
    
    # Flip bounding_boxes (batch, num_bbs, 4)
    index = keys.index('bounding_boxes')
    in_[index] = tf.where(condition, in_[index], tf.abs([1., 0., 1., 0.] - tf.gather(in_[index], [2, 1, 0, 3], axis=-1)))
        
    # Flip empty cell mask (batch, num_cells_x, num_cells_y, 1, num_bbs)
    index = keys.index('obj_i_mask_bbs')
    in_[index] = tf.where(condition, in_[index], tf.reverse(in_[index], [2]))

    # Flip groups boxes (batch, num_cells, num_cells, 1, 4)
    try:
        index = keys.index('group_bounding_boxes_per_cell')
        in_[index] = tf.where(
            condition, in_[index], tf.abs([1., 0., 1., 0.] - tf.gather(
            tf.reverse(in_[index], [2]), [2, 1, 0, 3], axis=-1))
                             )
    except ValueError:
        pass
        
    # Flip groups flags (batch, num_cells, num_cells, 1, 1)
    try:
        index = keys.index('group_flags')
        in_[index] = tf.where(condition, in_[index], tf.reverse(in_[index], [2]))
    except ValueError:
        pass
        
    # Flip groups classes (batch, num_cells, num_cells, 1, num_classes)
    try:
        index = keys.index('group_class_labels')
        in_[index] = tf.where(condition, in_[index], tf.reverse(in_[index], [2]))
    except ValueError:
        pass            
    return in_


def get_tf_dataset(tfrecords_file,
                   record_keys,
                   max_num_bbs,
                   num_classes,
                   with_classes=True,
                   with_groups=True,
                   grid_offsets=None,
                   batch_size=1,
                   image_size=448,
                   image_folder='',
                   data_augmentation_threshold=0.5,
                   num_splits=1,
                   num_threads=1,
                   subset=-1,
                   shuffle_buffer=1,
                   prefetch_capacity=1,
                   pad_with_dummies=0,
                   verbose=True):
    """Returns a queue containing the inputs batches.

    Args:
      tfrecords_file: Path to the TFRecords file containing the data.
      feature_keys: Feature keys present in the TFrecords
      max_num_bbs: Maximum number of bounding boxes in the dataset. Used for reshaping the `bounding_boxes` records.   
      num_classes: Number of classes in the dataset.
      with_classes: wheter to use class information
      with_groups: whether to pre-compute clustered groups ground-truth
      grid_offsets: Precomputed grid offset 
      batch_size: Batch size.
      image_size: The square size which to resize images to.
      image_folder: path to the directory containing the images in the dataset.
      data_augmentation_threshold: Data augmentation probabilitiy (in [0, 1])
      num_splits: Create num_splits splits of the data each of size batch-size
      num_threads: Number of readers for the batch queue.
      subset: If positive, extract the given number of samples as a subset of the dataset
      shuffle_buffer: Size of the shuffling buffer.
      prefetch_capacity: Buffer size for prefetching.
      pad_with_dummies: If positive, pad the dataset with the given number of dummy samples *before* repeat

    Returns: 
      A list of `num_splits` dictionnary of inputs.
    """
    # Asserts
    assert num_classes in [6, 9, 15]
    assert len(record_keys)
    assert batch_size > 0
    assert image_size > 0
    assert 0. <= data_augmentation_threshold <= 1.
    if grid_offsets is not None:
        num_cells = grid_offsets.shape[:2]
    assert num_splits > 0
    assert num_threads > 0
    assert shuffle_buffer > 0
    assert pad_with_dummies < batch_size * num_splits
    
    # Create TFRecords feature
    keys = get_inputs_keys(record_keys, with_groups, with_classes)
    features = tfrecords_utils.read_tfrecords(record_keys, max_num_bbs=max_num_bbs)
    
    # Normalize grid cells offsets
    if grid_offsets is not None:
        grid_offsets_mins = grid_offsets / num_cells
        grid_offsets_maxs = (grid_offsets + 1.) / num_cells 
    
    # Preprocess
    def parsing_function(example_proto):
        # Basic features
        parsed_features = tf.parse_single_example(example_proto, features)
        output = list(parse_basic_feature(parsed_features, num_classes, image_folder, image_size))
        bounding_boxes = output[-1]
        
        if 'class_labels' in keys:
            class_labels = tf.one_hot(parsed_features['classes'], num_classes, 
                                      axis=-1, on_value=1, off_value=0, dtype=tf.int32)
            output.append(class_labels)   
        
        # obj_i_mask_bbs: (num_cells, num_cells, 1, num_bbs)
        assert 'obj_i_mask_bbs' in keys
        mins, maxs = tf.split(bounding_boxes, 2, axis=-1) # (num_bbs, 2)
        inters = tf.maximum(0., tf.minimum(maxs, grid_offsets_maxs) - tf.maximum(mins, grid_offsets_mins))
        inters = tf.reduce_prod(inters, axis=-1)
        obj_i_mask = tf.expand_dims(tf.to_float(inters > 0.) , axis=-2)
        output.append(obj_i_mask)
                    
        # Grouped bounding boxes 
        if "group_bounding_boxes_per_cell" in keys:
            assert "num_group_boxes" in keys
            assert "group_flags" in keys
            # group_bounding_boxes_per_cell: (num_cells, num_cells, 1, 4)
            obj_i_mask = tf.transpose(obj_i_mask, (0, 1, 3, 2)) # (num_cells, num_cells, num_bbs, 1)
            mins = mins + 1. - obj_i_mask 
            mins = tf.reduce_min(mins, axis=2, keep_dims=True) # (num_cells, num_cells, 1, 2)
            maxs = maxs * obj_i_mask
            maxs = tf.reduce_max(maxs, axis=2, keep_dims=True)
            group_bounding_boxes_per_cell = tf.concat([mins, maxs], axis=-1)
            group_bounding_boxes_per_cell = tf.clip_by_value(group_bounding_boxes_per_cell, 0., 1.)
            
            # group_flags: (num_cells, num_cells, 1, 1)
            # num_group_boxes: ()
            num_bbs_per_cell = tf.reduce_sum(obj_i_mask, axis=2, keep_dims=True)
            num_group_boxes = tf.reduce_sum(tf.to_int32(num_bbs_per_cell > 0))
            group_flags = tf.maximum(tf.minimum(num_bbs_per_cell, 2.) - 1., 0.)
            
            # Add to outputs
            output.append(num_group_boxes)
            output.append(group_bounding_boxes_per_cell)
            output.append(group_flags)
            
        # Group classes (majority vote) # (num_cells, num_cells, 1, num_classes)
        if "group_class_labels" in keys:
            assert "group_bounding_boxes_per_cell" in keys
            assert "class_labels" in keys
            percell_class_labels = tf.expand_dims(tf.expand_dims(class_labels, axis=0), axis=0)
            percell_class_labels = obj_i_mask * tf.to_float(percell_class_labels) # (num_cells, num_cells, num_bbs, num_classes)
            percell_class_labels = tf.reduce_sum(percell_class_labels, axis=2, keep_dims=True)
            group_class_labels = tf.argmax(percell_class_labels, axis=-1)
            group_class_labels = tf.one_hot(group_class_labels, num_classes,
                                            axis=-1, on_value=1, off_value=0, dtype=tf.int32)
            group_class_labels = tf.to_int32(percell_class_labels * tf.to_float(group_class_labels))
            output.append(group_class_labels)
          
        # is_reversed flag: ()
        output.append(tf.constant(0.))
        return output
                    
        
    ## Create the dataset
    with tf.name_scope('load_dataset'):
        # Parse data
        dataset = tf.data.TFRecordDataset(tfrecords_file)
        if subset > 0: dataset = dataset.take(subset)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.map(parsing_function, num_parallel_calls=num_threads)
        # Pad samples for test data
        if pad_with_dummies > 0:
            pad_dataset = get_dummy_dataset(keys, pad_with_dummies, image_size, num_classes, num_cells, max_num_bbs)
            dataset = dataset.concatenate(pad_dataset)
        dataset = dataset.repeat()
        # Batch
        dataset = dataset.batch(num_splits * batch_size)
        if prefetch_capacity > 0: dataset = dataset.prefetch(prefetch_capacity)
        # Iterator
        iterator = dataset.make_one_shot_iterator()    
        in_ = list(iterator.get_next())
        
    ## Data augmentation
    with tf.name_scope('data_augmentation'):
        apply_data_augmentation(in_, num_splits * batch_size, keys, data_augmentation_threshold)
    
    ## Create inputs dictionary
    with tf.name_scope('create_inputs'):
        inputs = [{'batch_size': batch_size} for _ in range(num_splits)]    
        for i, key in enumerate(keys):
            splits = tf.split(in_[i], num_splits, axis=0)
            for s in range(num_splits):
                inputs[s][key] = tf.identity(splits[s], name="%s_device%d" % (key, i))
                
    if verbose == 1:
        print('\n'.join("    \033[32m%s\033[0m: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in inputs[0].items() if key != 'batch_size'))
    elif verbose > 1:
        print('\n'.join("    *%s*: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in inputs[0].items() if key != 'batch_size'))
    return inputs  


def extract_groups(inputs, 
                   outputs,
                   mode='train',
                   verbose=False,
                   epsilon=1e-8,
                   **kwargs): 
    """ Extract crops from the outputs of intermediate stage.
    
    Args:
        inputs: Inputs dictionnary of stage s
        outputs: Outputs dictionnary of stage s
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
    assert mode in ['train', 'test']
    (confidence_threshold, nms_threshold, num_outputs) = graph_manager.get_defaults(
        kwargs, ['%s_patch_confidence_threshold' % mode, 'patch_nms_threshold', '%s_num_crops' % mode], verbose=verbose)
    if verbose:        
        print('  > extracting %d crops' % num_outputs)
        
    ## Flatten
    # predicted_score: (batch, num_boxes, 1)
    # predicted_boxes: (batch, num_boxes, 4)
    with tf.name_scope('flat_output'):
        predicted_scores = tf_utils.flatten_percell_output(outputs["confidence_scores"])
        predicted_boxes = tf_utils.flatten_percell_output(outputs["bounding_boxes"])
        
    # At test time we only extract crops if groups or low confidence individuals
    with tf.name_scope('filter_groups'):
        if mode == 'test' and 'group_classification_logits' in outputs:
            strong_confidence_threshold = graph_manager.get_defaults(
                kwargs, ['test_patch_strong_confidence_threshold'], verbose=verbose)[0]
            # is_group: (batch, num_boxes, 1)
            is_group = tf.to_float(tf.nn.sigmoid(outputs['group_classification_logits']) > 0.5)
            is_group = tf_utils.flatten_percell_output(is_group)
            # should_be_refined: (batch, num_boxes, 1) : groups and not strongly confident individuals
            is_not_strongly_confident = tf.to_float(predicted_scores <= strong_confidence_threshold)
            should_be_refined = tf.minimum(1., is_group + is_not_strongly_confident)
            # Add confident single boxes as additional output of (batch, num_boxes, 4) and (batch, num_boxes, 1) shape
            outputs['added_detection_scores'] = (1. - should_be_refined) * tf_utils.flatten_percell_output(
                outputs["detection_scores"])
            outputs['added_bounding_boxes'] = (1. - should_be_refined) * predicted_boxes
            # Filter them out from potential crops
            predicted_scores *= should_be_refined
            predicted_boxes *= should_be_refined
        
    ## Filter out low confidences
    # predicted_score: (batch, num_boxes)
    predicted_scores = tf.squeeze(predicted_scores, axis=-1)
    with tf.name_scope('filter_confidence'):
        filtered = tf.to_float(predicted_scores > confidence_threshold)
        predicted_scores *= filtered
        predicted_boxes *= tf.expand_dims(filtered, axis=-1)
        
    ## Rescale boxes with the learned offsets
    with tf.name_scope('offsets_rescale_boxes'):
        if 'offsets' in outputs:
            predicted_offsets = tf_utils.flatten_percell_output(outputs["offsets"])
            predicted_boxes = tf_utils.rescale_with_offsets(predicted_boxes, predicted_offsets, epsilon)
    
    ## Non-maximum suppression
    # nms_boxes: (batch, num_crops, 4)
    # nms_boxes_confidences: (batch, num_crops)
    with tf.name_scope('nms'):
        nms_boxes = []
        nms_boxes_confidences = []
        for i in range(inputs['batch_size']):
            boxes, scores = tf_utils.nms_with_pad(predicted_boxes[i, :, :], 
                                                  predicted_scores[i, :],
                                                  num_outputs, 
                                                  iou_threshold=nms_threshold)
            nms_boxes.append(boxes)
            nms_boxes_confidences.append(scores)
        nms_boxes = tf.stack(nms_boxes, axis=0) 
        nms_boxes = tf.reshape(nms_boxes, (-1, num_outputs, 4))
        nms_boxes_confidences = tf.stack(nms_boxes_confidences, axis=0) 
        nms_boxes_confidences = tf.reshape(nms_boxes_confidences, (-1, num_outputs))
        
    ## Return
    outputs['crop_boxes'] = nms_boxes
    outputs['crop_boxes_confidences'] = nms_boxes_confidences
    return nms_boxes, nms_boxes_confidences


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
                          image_folder=None,
                          batch_size=32,
                          num_classes=80,
                          image_size=256,
                          full_image_size=1024,
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
        image_folder: Image directory, used for reloading the full resolution images
        batch_size: Batch size for the output of this pipeline
        num_classes: Number of classes in the dataset
        image_size: Size of the images patches in the new dataset
        full_image_size: Size of the images to load before applying the croppings
        grid_offsets: A (num_cells, num_cells) array
        use_queue: Whether to use a queue or directly output the new inputs dictionary
        shuffle_buffer: shuffle buffer of the output queue
        num_threads: number of readers in the output queue
        capacity: Output queue capacity
        verbose: verbosity        
    """
    assert batch_size > 0   
    assert 0. <= intersection_ratio_threshold < 1.
    num_crops = crop_boxes.get_shape()[1].value
    assert num_crops > 0.
    new_inputs = {}
    
    # new_im_id: (num_patches,)
    with tf.name_scope('im_ids'):
        new_inputs['im_id'] = tile_and_reshape(inputs['im_id'], num_crops)        
        
    if 'class_labels' in inputs:
        with tf.name_scope('class_labels'):
            new_inputs['class_labels'] = tile_and_reshape(inputs['class_labels'], num_crops)
    
    # new_image: (num_patches, image_size, image_size, 3)
    with tf.name_scope('extract_image_patches'):
        # Re-load full res image (flip if necessary)
        if image_folder is not None and full_image_size > 0:
            print('   > Upscale patch from %dx%d ground-truth' % (full_image_size, full_image_size))
            full_images = []
            for i in range(inputs['batch_size']):
                image = tf.cond(inputs['im_id'][i] >= 0,
                                true_fn=lambda: load_image(inputs['im_id'][i], num_classes, full_image_size, image_folder),
                                false_fn=lambda: tf.zeros((full_image_size, full_image_size, 3)))
                full_images.append(image)
            full_images = tf.stack(full_images, axis=0)     
            full_images = tf.where(inputs["is_flipped"] > 0., tf.reverse(full_images, [2]), full_images)
        else:
            print('   > Extract patch directly from input image')
            full_images = inputs['image']
        # Extract patches and resize
        # crop_boxes_indices: (batch * num_crops,)
        # crop_boxes_flat: (batch * num_crops, 4)
        crop_boxes_indices = tf.ones(tf.shape(crop_boxes)[:2], dtype=tf.int32)
        crop_boxes_indices = tf.cumsum(crop_boxes_indices, axis=0, exclusive=True)
        crop_boxes_indices = tf.reshape(crop_boxes_indices, (-1,))
        crop_boxes_flat = tf.gather(tf.reshape(crop_boxes, (-1, 4)), [1, 0, 3, 2], axis=-1)
        new_inputs['image'] = tf.image.crop_and_resize(full_images, crop_boxes_flat, crop_boxes_indices, 
                                                       (image_size, image_size), name='extract_groups')
        
    # new_bounding_boxes: (num_patches, max_num_bbs, 4)
    # rescale bounding boxes to the cropped image
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
    with tf.name_scope('num_boxes'):
        valid_boxes = ((bounding_boxes[..., 2] > bounding_boxes[..., 0]) & 
                       (bounding_boxes[..., 3] > bounding_boxes[..., 1]))
        num_boxes =  tf.to_float(valid_boxes)
        new_inputs['num_boxes'] = tf.to_int32(tf.reduce_sum(num_boxes, axis=-1) )
        
    # Compute the box presence in cell mask
    # obj_i_mask_bbs: (num_patches, num_cells, num_cells, 1, num_gt)
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
        
    # Enqueue thje new inputs during training, or pass the output directly to the next stage at test time
    if use_queue:
        filter_valid = tf.logical_and(crop_boxes[..., 2] > crop_boxes[..., 0], crop_boxes[..., 3] > crop_boxes[..., 1] )
        filter_valid = tf.reshape(filter_valid, (-1,))
        if shuffle_buffer <= 1:
            out_ = tf.train.maybe_batch(
                new_inputs, filter_valid, batch_size, num_threads=num_threads, enqueue_many=True, capacity=capacity)
        else:
            out_ = tf.train.maybe_shuffle_batch(
                new_inputs, batch_size, capacity, shuffle_buffer, filter_valid, num_threads=num_threads, enqueue_many=True)
    else:
        out_ = new_inputs        
    out_['batch_size'] = batch_size       
    
    if verbose == 1:
        print('\n'.join("    \033[32m%s\033[0m: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in out_.items() if key != 'batch_size'))
    elif verbose > 1:
        print('\n'.join("    *%s*: shape=%s, dtype=%s" % (key, value.get_shape().as_list(), value.dtype) 
                        for key, value in out_.items() if key != 'batch_size'))
    return out_   