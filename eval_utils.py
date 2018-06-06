import tensorflow as tf
import tf_utils
import graph_manager


def get_samples_running_counters(inputs, outputs, verbose=False, **kwargs):
    """Count number of test samples and this value to the graph"""
    metrics = []
    # Number of samples (excluding dummies)
    metrics.append(('num_samples_eval' , None, tf.reduce_sum(tf.to_float(inputs['im_id'] >= 0))))

    # Number of non-empty samples
    num_valid_bbs = tf.to_float(inputs['num_boxes'])
    is_non_empty = tf.minimum(num_valid_bbs, 1)
    metrics.append(('num_valid_samples_eval' , None, tf.reduce_sum(is_non_empty)))
    return metrics


def get_standard_eval(inputs, 
                      outputs,  
                      verbose=False,
                      epsilon=1e-8,
                      base_name="net",
                      add_sample_counters=False,
                      **kwargs):
    """ Compute and store running evaluation metrics
    
    Args:
        inputs: Dictionnary of inputs.
        outputs: Dictionnary of outputs. 
        
    Kwargs:
        retrieval_top_n: Max number of bbs to retrieve. 
        retrieval_intersection_threshold. Compute map (resp. recall) at the given iou (resp. inter) threshold
        retrieval_confidence_threshold. Confidence threshold for predictions
        retrieval_nms_threshold. Confidence threshold for predictions
    
    Returns:
        A list of (key, tensor) representing each evaluation metric name and Tensor containing its value.
        Note: Summed over the batch so it can be normalized afterwards
    """     
    # kwargs
    (retrieval_top_n, retrieval_intersection_threshold, retrieval_confidence_threshold) = graph_manager.get_defaults(
        kwargs, ['retrieval_top_n', 'retrieval_intersection_threshold', 'retrieval_confidence_threshold'], verbose=verbose)
    assert retrieval_top_n > 0
            
    metrics = []
        
    ## Flatten output
    with tf.name_scope('flat_output'):
        # predicted_score: (batch, num_classes, num_boxes)
        predicted_scores = tf.stack([tf.layers.flatten(x) for x in tf.unstack(outputs["detection_scores"], axis=-1)], axis=1)
        num_classes = predicted_scores.get_shape()[1].value     
        # predicted_bbs: (batch, num_classes, num_boxes, 4)   
        predicted_bbs = tf_utils.flatten_percell_output(outputs['bounding_boxes'])
        predicted_bbs = tf.expand_dims(predicted_bbs, axis=1)
        predicted_bbs = tf.tile(predicted_bbs, (1, num_classes, 1, 1))
        
    ## Order boxes by decreasing confidence
    with tf.name_scope('select_top_boxes'):
        # top_predicted_scores: (batch_size, num_classes, topn)
        top_predicted_scores, boxes_index = tf.nn.top_k(predicted_scores, k=retrieval_top_n)
        boxes_index = tf.expand_dims(boxes_index, axis=-1)
        batch_index = tf.range(inputs['batch_size'])
        class_index = tf.range(num_classes)
        # indices: (batch_size, num_classes, topn, 3)
        indices = tf.stack(tf.meshgrid(batch_index, class_index), axis=-1)
        indices = tf.transpose(indices, (1, 0, 2))
        indices = tf.expand_dims(indices, axis=-2)
        indices = tf.tile(indices, (1, 1, retrieval_top_n, 1))
        indices = tf.concat([indices, boxes_index], axis=-1)
        # top_predicted_boxes: (batch_size, num_classes, topn, 4)
        top_predicted_boxes = tf.gather_nd(predicted_bbs, indices) 
       
    ## Filter out boxes with low confidences
    with tf.name_scope('filter_confidence'):
        filtered = tf.to_float(top_predicted_scores > retrieval_confidence_threshold)
        top_predicted_scores *= filtered
        top_predicted_boxes *= tf.expand_dims(filtered, axis=-1)
        
    ## Apply NMS
    with tf.name_scope('non_maximum_suppression'):
        nms_threshold = graph_manager.get_defaults(kwargs, ['retrieval_nms_threshold'], verbose=verbose)[0]
        final_boxes = []
        for i in range(inputs['batch_size']):
            perclass_final_boxes = []
            for k in range(num_classes):
                perclass_final_boxes.append(
                    tf_utils.nms_with_pad(top_predicted_boxes[i, k, :, :], 
                                          top_predicted_scores[i, k, :], 
                                          retrieval_top_n, 
                                          iou_threshold=nms_threshold)[0])
            final_boxes.append(tf.stack(perclass_final_boxes, axis=0))
        top_predicted_boxes = tf.stack(final_boxes, axis=0)
        
    ## Pairwise intersection over union scores
    with tf.name_scope('compute_ious'):
        if num_classes == 1:
            # true_coords: (batch_size, 1, 1, num_gt)
            num_relevant_bbs_perclass = tf.to_float(tf.expand_dims(inputs['num_boxes'], axis=-1))
            true_coords = tf.split(
                tf.expand_dims(tf.transpose(inputs['bounding_boxes'], (0, 2, 1)), axis=1), 4, axis=-2)
        else:
            assert 'class_labels' in inputs
            assert inputs['class_labels'].get_shape()[-1].value == num_classes
            class_key = 'class_labels'
            num_relevant_bbs_perclass = tf.reduce_sum(inputs[class_key], axis=1)
            true_coords = tf.expand_dims(inputs[gt_bbs_key], axis=2) # (batch, num_gt, 1, 4)
            true_coords *= tf.to_float(tf.expand_dims(inputs[class_key], axis=-1)) # (batch, num_gt, num_classes, 4)
            true_coords = tf.transpose(true_coords, (0, 2, 3, 1))
            true_coords = tf.split(true_coords, 4, axis=2)  
            for i in range(num_classes):
                metrics.append(('num_valid_samples_%d_eval' % (i), None, 
                                tf.reduce_sum(tf.to_float(num_valid_bbs_perclass[:, i] > 0))))
        # ious: (batch_size, num_classes, topn, num_gt)
        pred_coords = tf.split(top_predicted_boxes, 4, axis=-1)
        ious = tf_utils.get_iou(true_coords, pred_coords)
    
    for thresh in retrieval_intersection_threshold:                    
        with tf.name_scope('compute_mapat%.2f' % thresh):
            # correct_boxes:  (batch_size, num_classes, topn, num_gt)
            # maps ground-truth to the first prediction that retrieves it
            correct_boxes = tf.to_float(ious > thresh)   
            duplicates = tf.to_float(tf.cumsum(correct_boxes, axis=-2, exclusive=True) > 0)
            correct_boxes = correct_boxes * (1. - duplicates)
            
            # precisions: (batch_size, num_classes, topn)
            # precision per rank: |relevant n retrieved| / |retrieved|
            num_correctly_retrieved = tf.reduce_sum(correct_boxes, axis=-1) # number of retrieval per prediction
            num_correctly_retrieved = tf.cumsum(num_correctly_retrieved, axis=-1) # cumulative correct retrieval per rank   
            num_predicted = tf.to_float(tf.reshape(tf.range(retrieval_top_n), (1, 1, retrieval_top_n))) + 1.
            precisions = tf.minimum(1., num_correctly_retrieved / num_predicted)
            
            # recalls: (batch_size, num_classes, topn)
            # recalls per rank: |relevant n retrieved| / |retrieved|
            #num_relevant = tf.reshape(tf.to_float(inputs['num_boxes']), (-1, 1, 1))
            #recalls = tf.minimum(1., num_correctly_retrieved / num_relevant) 
            
            # average_precisions: (batch_size, num_classes)
            # Compute the average of precisions at each rank a box is detected (change in recall)
            change_in_recall = tf.minimum(1., tf.reduce_sum(correct_boxes, axis=-1))
            precisions_at_recallchange = precisions * change_in_recall
            average_precisions = tf.reduce_sum(precisions_at_recallchange, axis=-1) / (num_relevant_bbs_perclass + epsilon)
            
            # total_recall: (batch_size, num_classes)
            # Ratio of ground-truth bounding boxes correctly retrieved from the full list of documents (i.e. max recall)
            is_gt_retrieved = tf.minimum(1., tf.reduce_sum(correct_boxes, axis=-2))
            total_recall = tf.reduce_sum(is_gt_retrieved, axis=-1)  / (num_relevant_bbs_perclass + epsilon)
            
            # Add to metrics
            if num_classes == 1:
                maps = tf.reduce_sum(average_precisions)
                metrics.append(('%s_avgprec_at%.2f_eval' % (base_name, thresh), 'num_valid_samples_eval', maps))
                rec = tf.reduce_sum(total_recall)
                metrics.append(('%s_maxrecall_at%.2f_eval' % (base_name, thresh), 'num_valid_samples_eval', rec))
            else:
                data_classes = kwargs['data_classes']
                for i in range(num_classes):
                    maps = tf.reduce_sum(average_precision[:, i])
                    metrics.append(('%s_avgprec_at%.2f_%s_eval' % (base_name, thresh, '_'.join(data_classes[i].split())), 
                                    'num_valid_samples_%d_eval' % (i), maps))    
                    rec = tf.reduce_sum(total_recall[:, i])
                    metrics.append(('%s_maxrecall_at%.2f_%s_eval' % (base_name, thresh, '_'.join(data_classes[i].split())), 
                                    'num_valid_samples_%d_eval' % (i), rec))                   
            
    return metrics



def get_odgi_eval(inputs, 
                  outputs,  
                  verbose=False,
                  epsilon=1e-8,
                  base_name="net",
                  **kwargs):
    """ Compute and store running evaluation metrics
    
    Args:
        inputs: Dictionnary of inputs.
        outputs: Dictionnary of outputs. 
        
    Kwargs:
        retrieval_top_n: Max number of bbs to retrieve. 
        retrieval_intersection_threshold. Compute map (resp. recall) at the given iou (resp. inter) threshold
        retrieval_confidence_threshold. Confidence threshold for predictions
        retrieval_nms_threshold. Confidence threshold for predictions
    
    Returns:
        A list of (key, tensor) representing each evaluation metric name and Tensor containing its value.
        Note: Summed over the batch so it can be normalized afterwards
    """     
    # Initialize number of samples
    metrics = []
    
    # Average number of crops
    with tf.name_scope('num_valid_crops'):
        crop_boxes = outputs["crop_boxes"]
        num_crop_boxes = tf.reduce_sum(tf.to_float(tf.logical_and(
            crop_boxes[..., 2] > crop_boxes[..., 0], crop_boxes[..., 3] > crop_boxes[..., 1])))
        metrics.append(('num_crops', 'num_samples_eval', num_crop_boxes))
        
    # Group classification accuracy
    if 'group_flags' in inputs and 'group_classification_logits' in outputs:
        with tf.name_scope('group_accuracy'):
            # non_empty_cells (batch, num_cells, num_cells, 1, 1)
            non_empty_cells = tf.minimum(1., tf.reduce_sum(inputs['obj_i_mask_bbs'], axis=-1, keep_dims=True))
            preds = tf.to_float(tf.nn.sigmoid(outputs['group_classification_logits']) > 0.5)
            # accuracies (batch, num_cells, num_cells, 1, 1)
            accuracies = 1. - tf.abs(inputs['group_flags'] - preds)
            accuracies *= non_empty_cells
            percell_accuracies = (tf.reduce_sum(accuracies, axis=(1, 2, 3, 4)) / 
                                  (tf.reduce_sum(non_empty_cells, axis=(1, 2, 3, 4)) + epsilon))
            metrics.append(('group_flag_accuracy', 'num_valid_samples_eval', tf.reduce_sum(percell_accuracies)))
            
    # Group average precision
    if 'group_bounding_boxes_per_cell' in inputs:
        (retrieval_intersection_threshold, retrieval_confidence_threshold) = graph_manager.get_defaults(
            kwargs, ['retrieval_intersection_threshold', 'retrieval_confidence_threshold'], verbose=verbose)
        number_outputs = outputs['bounding_boxes'].get_shape()[1].value * outputs['bounding_boxes'].get_shape()[2].value
        # predicted_score: (batch, num_boxes)
        predicted_scores = tf.layers.flatten(outputs["confidence_scores"])        
        # predicted_bbs: (batch, num_boxes, 4)   
        predicted_bbs = tf_utils.flatten_percell_output(outputs['bounding_boxes'])            
        # order boxes by decreasing confidence
        with tf.name_scope('select_top_boxes'):
            # top_predicted_scores: (batch_size, topn)
            top_predicted_scores, boxes_index = tf.nn.top_k(predicted_scores, k=number_outputs)
            boxes_index = tf.expand_dims(boxes_index, axis=-1)
            # indices: (batch_size, topn, 2)
            indices = tf.range(inputs['batch_size'])
            indices = tf.expand_dims(tf.expand_dims(indices, axis=-1), axis=-1)
            indices = tf.tile(indices, (1, number_outputs, 1))
            # top_predicted_boxes: (batch_size, topn, 4)
            top_predicted_boxes = tf.gather_nd(predicted_bbs, tf.concat([indices, boxes_index], axis=-1)) 
            
        ## Filter out boxes with low confidences
        with tf.name_scope('filter_confidence'):
            filtered = tf.to_float(top_predicted_scores > retrieval_confidence_threshold)
            top_predicted_scores *= filtered
            top_predicted_boxes *= tf.expand_dims(filtered, axis=-1)
            
        ## Apply NMS
        with tf.name_scope('non_maximum_suppression'):
            nms_threshold = graph_manager.get_defaults(kwargs, ['retrieval_nms_threshold'], verbose=verbose)[0]
            final_boxes = []
            for i in range(inputs['batch_size']):
                final_boxes.append(tf_utils.nms_with_pad(top_predicted_boxes[i, :, :], 
                                                         top_predicted_scores[i, :], 
                                                         number_outputs,
                                                         iou_threshold=nms_threshold)[0])
            top_predicted_boxes = tf.stack(final_boxes, axis=0)
        
        ## Pairwise intersection over union scores
        with tf.name_scope('compute_ious'):
            # true_coords: (batch_size, 1, num_gt)
            num_relevant_bbs = tf.reduce_sum(non_empty_cells, axis=(1, 2, 3, 4))
            true_coords = tf_utils.flatten_percell_output(inputs['group_bounding_boxes_per_cell'])
            true_coords = tf.split(tf.transpose(true_coords, (0, 2, 1)), 4, axis=-2)
            # ious: (batch_size, topn, num_gt)
            pred_coords = tf.split(top_predicted_boxes, 4, axis=-1)
            ious = tf_utils.get_iou(true_coords, pred_coords)

        ## map@thresholds
        for thresh in retrieval_intersection_threshold:                    
            with tf.name_scope('compute_mapat%.2f' % thresh):
                # correct_boxes:  (batch_size, topn, num_gt)
                # maps ground-truth to the first prediction that retrieves it
                correct_boxes = tf.to_float(ious > thresh)   
                duplicates = tf.to_float(tf.cumsum(correct_boxes, axis=-2, exclusive=True) > 0)
                correct_boxes = correct_boxes * (1. - duplicates)            
                # precisions: (batch_size, topn)
                # precision per rank: |relevant n retrieved| / |retrieved|
                num_correctly_retrieved = tf.reduce_sum(correct_boxes, axis=-1) # number of retrieval per prediction
                num_correctly_retrieved = tf.cumsum(num_correctly_retrieved, axis=-1) # cumulative correct retrieval per rank   
                num_predicted = tf.to_float(tf.reshape(tf.range(number_outputs), (1, number_outputs))) + 1.
                precisions = tf.minimum(1., num_correctly_retrieved / num_predicted)            
                # average_precisions: (batch_size,)
                # Compute the average of precisions at each rank a box is detected (change in recall)
                change_in_recall = tf.minimum(1., tf.reduce_sum(correct_boxes, axis=-1))
                precisions_at_recallchange = precisions * change_in_recall
                average_precisions = tf.reduce_sum(precisions_at_recallchange, axis=-1) / (num_relevant_bbs + epsilon)  
                # Add to metrics
                maps = tf.reduce_sum(average_precisions)
                metrics.append(('%s_avgprec_at%.2f_eval' % (base_name, thresh), 'num_valid_samples_eval', maps))          
                # total_recall: (batch_size,)
                # Ratio of ground-truth bounding boxes correctly retrieved from the full list of documents (i.e. max recall)
                is_gt_retrieved = tf.minimum(1., tf.reduce_sum(correct_boxes, axis=-2))
                total_recall = tf.reduce_sum(is_gt_retrieved, axis=-1)  / (num_relevant_bbs + epsilon)
                # Add to metrics
                rec = tf.reduce_sum(total_recall)
                metrics.append(('%s_maxrecall_at%.2f_eval' % (base_name, thresh), 'num_valid_samples_eval', rec))
            
    return metrics