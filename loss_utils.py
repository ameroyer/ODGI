import tensorflow as tf
import tf_utils
import graph_manager


def get_standard_loss(inputs, 
                      outputs,
                      is_chief=True,
                      verbose=False,
                      base_name="net",
                      epsilon=1e-8,
                      num_cells=None,
                      **kwargs):
    """ Compute the loss function for standard object detection (final stage).
    
    Args:
        inputs: A dictionnary of inputs
        outputs: A dictionnary of outputs
        is_chief: Adds additional summaries iff is_chief is True
        verbose: verbosity level
        base_name: Prefix for all the loss colelctions to be added in the graph
        epsilon: for avoiding overflow error
        num_cells: 2D array number of cells in teh grid, used to normalize the centers distance
        
    Kwargs:    
        centers_localization_loss_weight: Weights for the localization losses of centers. defaults to 1
        scales_localization_loss_weight: Weights for the localization losses of log scales. defaults to 1
        confidence_loss_weight: Weights for the confidence loss. defaults to 5
        noobj_confidence_loss_weight: Weights for the confidence loss  (empty cells). defautls to 1
        classification_loss_weight: weights for the counting loss. defaults to 1
        target_conf_fn: Function to use to compute target confidences. One of iou, soft_iou, inter_ratio
        assignment_reward_fn: function to use to compute assignment reward. One of iou, soft_iou, coords. 
    """     
    (target_conf_fn, assignment_reward_fn, centers_localization_loss_weight, scales_localization_loss_weight, 
     confidence_loss_weight, noobj_confidence_loss_weight) = graph_manager.get_defaults(kwargs, [
        'target_conf_fn', 'assignment_reward_fn', 'centers_localization_loss_weight', 'scales_localization_loss_weight', 
        'confidence_loss_weight', 'noobj_confidence_loss_weight'], verbose=verbose)
    assert num_cells is not None
    assert target_conf_fn in ['iou', 'upper_iou', 'intersection_ratio', 'coords_sims']
    assert assignment_reward_fn in ['iou', 'upper_iou', 'intersection_ratio', 'coords_sims']
    
    # obj_i_mask: (batch, num_cells, num_cells, 1, num_gt), indicates presence of a box in a cell
    obj_i_mask = inputs['obj_i_mask_bbs']
        
    ## Split coordinates
    # pred_bbs: 4 * (batch, num_cells, num_cells, num_preds, 1)
    # true_bbs: 4 * (batch, 1, 1, 1, num_gt)
    with tf.name_scope('coordinates'):
        pred_bbs = tf.split(outputs['bounding_boxes'], 4, axis=-1)
        true_bbs = tf.split(tf.expand_dims(tf.expand_dims(
            tf.transpose(inputs['bounding_boxes'], (0, 2, 1)), axis=1), axis=1), 4, axis=-2)        
        
    ## Compute target value for the assignment reward and the confidences
    # target_confs: (batch, num_cells, num_cells, num_preds, num_gt)
    with tf.name_scope('compute_target_confidence'): 
        target_confs = getattr(tf_utils, 'get_%s' % target_conf_fn)(true_bbs, pred_bbs, epsilon=epsilon)
        target_confs = tf.stop_gradient(target_confs)
    
    # assignment_rewards: (batch, num_cells, num_cells, num_preds, num_gt)
    with tf.name_scope('compute_assignment_reward'):        
        if assignment_reward_fn == target_conf_fn:
            assignment_rewards = target_confs
        else:
            assignment_rewards = getattr(tf_utils, 'get_%s' % assignment_reward_fn)(true_bbs, pred_bbs, epsilon=epsilon)
        assignment_rewards = tf.stop_gradient(assignment_rewards)    
    
    ## Create obj mask mapping ground-truth to predictors
    # obj_ij_mask: (batch, num_cells, num_cells, num_preds, num_gt, 1)
    with tf.name_scope('assign_predictors'):        
        best_reward = tf.reduce_max(assignment_rewards, axis=-2, keepdims=True)
        obj_ij_mask = tf.to_float(tf.greater_equal(assignment_rewards, best_reward))
        obj_ij_mask *= obj_i_mask
        obj_ij_mask = tf.expand_dims(obj_ij_mask, axis=-1) 
        obj_ij_mask = tf.stop_gradient(obj_ij_mask)    
    
    ## Localization loss
    with tf.name_scope('localization_loss'):
        # true_mins, true_maxs: (batch, num_cells, num_cells, num_preds, num_gt, 2)
        true_mins = tf.stack(true_bbs[:2], axis=-1)
        true_maxs = tf.stack(true_bbs[2:], axis=-1)
        # centers
        with tf.name_scope('xy_loss'):
            centers_diffs = tf.expand_dims(outputs['shifted_centers'], axis=-2) - num_cells * (true_mins + true_maxs) / 2
            centers_localization_loss = tf.losses.compute_weighted_loss(
                centers_diffs**2,
                weights=centers_localization_loss_weight * obj_ij_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)        
        # scales
        with tf.name_scope('wh_loss'):
            scales_diff = tf.expand_dims(outputs['log_scales'], axis=-2) - tf.log(tf.maximum(epsilon, true_maxs - true_mins))
            scales_localization_loss = tf.losses.compute_weighted_loss(
                scales_diff**2,
                weights=scales_localization_loss_weight * obj_ij_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        
    ## Confidence loss
    with tf.name_scope('conf_loss'):  
        # Best predictor in non-empty cells
        with tf.name_scope('non_empty'):
            # confs_diffs: (batch, num_cells, num_cells, num_preds, num_gt)
            obj_mask = tf.squeeze(obj_ij_mask, axis=-1)
            confs_diffs = target_confs - outputs["confidence_scores"]
            confidence_loss_obj = tf.losses.compute_weighted_loss(
                confs_diffs**2,
                weights=confidence_loss_weight * obj_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)        
        # Predictors in empty cells
        with tf.name_scope('empty'):
            # noobj_mask: (batch, num_cells, num_cells, 1, 1)
            noobj_mask = 1. - tf.minimum(1., tf.reduce_sum(obj_i_mask, axis=-1, keepdims=True))
            confidence_loss_noobj = tf.losses.compute_weighted_loss(
                outputs["confidence_scores"]**2,
                weights=noobj_confidence_loss_weight * noobj_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        
    ## Classification loss
    if 'classification_probs' in outputs:        
        assert 'class_labels' in inputs        
        with tf.name_scope('classification_loss'):
            classification_loss_weight = graph_manager.get_defaults(kwargs, ['classification_loss_weight'], verbose=verbose)[0]
            # labels: (batch, 1, 1, 1, num_gt, num_classes)
            labels = inputs['class_labels'] # (batch, num_gt, num_classes)
            labels = tf.expand_dims(labels, axis=1)
            labels = tf.expand_dims(labels, axis=1)
            labels = tf.stop_gradients(tf.to_float(tf.expand_dims(labels, axis=1)))
            # logits: (batch, num_cells, num_cells, num_preds, 1, num_classes)
            logits = outputs['classification_probs']
            logits = tf.expand_dims(logits, axis=4)
            # classification loss
            class_diffs = labels - logits
            classification_loss = tf.losses.compute_weighted_loss(
                class_diffs**2,
                weights=classification_loss_weight * obj_ij_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    else:
        classification_loss = 0.        
    
    ## Add informative summaries
    if is_chief:
        is_assigned_predictor = tf.to_float(tf.reduce_sum(obj_ij_mask, axis=-2) > 0.)
        outputs["target_bounding_boxes"] = outputs["bounding_boxes"] * is_assigned_predictor
        
    return [('%s_centers_localization_loss' % base_name, centers_localization_loss),
            ('%s_scales_localization_loss' % base_name, scales_localization_loss),
            ('%s_confidence_obj_loss' % base_name, confidence_loss_obj),
            ('%s_confidence_noobj_loss' % base_name, confidence_loss_noobj),
            ('%s_classification_loss' % base_name, classification_loss)]


def get_odgi_loss(inputs, 
                  outputs,
                  is_chief=True,
                  verbose=False,
                  base_name="net",
                  epsilon=1e-8,
                  num_cells=None,
                  **kwargs):
    """ Compute the ODGI loss function.
    
    Args:
        inputs: A dictionnary of inputs
        outputs: A dictionnary of outputs
        is_chief: Adds additional summaries iff is_chief is True
        verbose: verbosity level
        base_name: Prefix for all the loss colelctions to be added in the graph
        epsilon: for avoiding overflow error
        num_cells: 2D array number of cells in teh grid, used to normalize the centers distance
        
    Kwargs:    
        class_key: Key to the ground-truth classes in teh dictionnary.
        centers_localization_loss_weight: Weights for the localization losses of centers. defaults to 1
        scales_localization_loss_weight: Weights for the localization losses of log scales. defaults to 1
        confidence_loss_weight: Weights for the confidence loss. defaults to 5
        noobj_confidence_loss_weight: Weights for the confidence loss  (empty cells). defaults to 1
        group_classification_loss_weight: weights for the group flags loss. defaults to 1
        offsets_loss_weight: weights for the offsets loss. defaults to 1
        classification_loss_weight: weights for the classification loss. defaults to 1
        target_conf_fn: Function to use to compute target confidences. One of iou, soft_iou, inter_ratio.
    """     
    (target_conf_fn, centers_localization_loss_weight, scales_localization_loss_weight, 
     confidence_loss_weight, noobj_confidence_loss_weight) = graph_manager.get_defaults(kwargs, [
        'target_conf_fn', 'centers_localization_loss_weight', 'scales_localization_loss_weight', 
        'confidence_loss_weight', 'noobj_confidence_loss_weight'], verbose=verbose)
    assert num_cells is not None
    assert target_conf_fn in ['iou', 'upper_iou', 'intersection_ratio', 'coords_sims']
    
    # obj_i_mask: (batch, num_cells, num_cells, 1, num_gt), indicates presence of a box in a cell
    obj_i_mask = inputs['obj_i_mask_bbs']
    non_empty_cell_mask = tf.minimum(1., tf.reduce_sum(obj_i_mask, axis=-1, keepdims=True))    
    
    ## Split coordinates
    # pred_bbs: 4 * (batch, num_cells, num_cells, 1, 1)
    # true_bbs: 4 * (batch, num_cells, num_cells, 1, 1)
    with tf.name_scope('coordinates'):
        pred_bbs = tf.split(outputs['bounding_boxes'], 4, axis=-1)
        true_bbs = tf.split(inputs['group_bounding_boxes_per_cell'], 4, axis=-1)        
        
    ## Compute target value for the confidences
    # target_confs: (batch, num_cells, num_cells, 1, 1)
    with tf.name_scope('compute_target_confidence'): 
        target_confs = getattr(tf_utils, 'get_%s' % target_conf_fn)(true_bbs, pred_bbs, epsilon=epsilon)
        target_confs = tf.stop_gradient(target_confs)                    
    
    ## Localization loss
    with tf.name_scope('localization_loss'):
        # true_mins, true_maxs: (batch, num_cells, num_cells, 1, 2)
        true_mins = tf.concat(true_bbs[:2], axis=-1)
        true_maxs = tf.concat(true_bbs[2:], axis=-1)
        # centers
        with tf.name_scope('xy_loss'):
            centers_diffs = outputs['shifted_centers'] - num_cells * (true_mins + true_maxs) / 2
            centers_localization_loss = tf.losses.compute_weighted_loss(
                centers_diffs**2,
                weights=centers_localization_loss_weight * non_empty_cell_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        # scales) 
        with tf.name_scope('wh_loss'):
            scales_diff = outputs['log_scales'] - tf.log(tf.maximum(1e-8, true_maxs - true_mins))
            scales_localization_loss = tf.losses.compute_weighted_loss(
                scales_diff**2,
                weights=scales_localization_loss_weight * non_empty_cell_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        
    ## Confidence loss
    with tf.name_scope('conf_loss'):  
        # Best predictor in non-empty cells
        with tf.name_scope('non_empty'):
            # confs_diffs: (batch, num_cells, num_cells, 1, 1)
            confs_diffs = target_confs - outputs["confidence_scores"]
            confidence_loss_obj = tf.losses.compute_weighted_loss(
                confs_diffs**2,
                weights=confidence_loss_weight * non_empty_cell_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)        
        # Predictors in empty cells
        with tf.name_scope('empty'):
            confidence_loss_noobj = tf.losses.compute_weighted_loss(
                outputs["confidence_scores"]**2,
                weights=noobj_confidence_loss_weight * (1. - non_empty_cell_mask),
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        
    ## Group classification loss
    if 'group_classification_logits' in outputs:  
        group_classification_loss_weight = graph_manager.get_defaults(
            kwargs, ['group_classification_loss_weight'], verbose=verbose)[0]
        assert 'group_flags' in inputs
        labels = inputs['group_flags']
        logits = outputs['group_classification_logits']
        # Flatten 
        labels = tf.reshape(labels, (-1, 1))
        logits = tf.reshape(logits, (-1, 1))
        weights = tf.reshape(non_empty_cell_mask, (-1, 1))
        # Loss
        group_classification_loss = tf.losses.sigmoid_cross_entropy(
            tf.stop_gradient(labels), 
            logits, 
            weights=group_classification_loss_weight * weights,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    else:
        group_classification_loss = 0.
        
    ## Offsets loss
    if 'offsets' in outputs:
        offsets_loss_weight, offsets_margin = graph_manager.get_defaults(
            kwargs, ['offsets_loss_weight', 'offsets_margin'], verbose=verbose)
        # pred_centers, pred_scales: (batch, num_cells, num_cells, 1, 2)
        pred_centers = tf.concat([pred_bbs[0] + pred_bbs[2], pred_bbs[1] + pred_bbs[3]], axis=-1) / 2.
        pred_scales = tf.concat([pred_bbs[2] - pred_bbs[0], pred_bbs[3] - pred_bbs[1]], axis=-1)  
        
        # coords: (batch, num_cells, num_cells, 1, 2, 2)
        x_coords = tf.stack([true_bbs[0], true_bbs[2]], axis=-1)
        y_coords = tf.stack([true_bbs[1], true_bbs[3]], axis=-1)
        coords = tf.concat([x_coords, y_coords], axis=-2)
        # target_scales: (batch, num_cells, num_cells, 1, 2)
        target_scales = tf.reduce_max(tf.abs(tf.expand_dims(pred_centers, axis=-1) - coords), axis=-1)
        target_scales = 2. * (target_scales + offsets_margin)
        
        # target_offsets: (batch, num_cells, num_cells, num_preds, 2)     
        target_offsets = tf.minimum(1., pred_scales / tf.maximum(epsilon, target_scales))        
        target_offsets = tf.stop_gradient(target_offsets)
        offsets_diffs = target_offsets - outputs["offsets"]
        offsets_loss = tf.losses.compute_weighted_loss(
            offsets_diffs**2,
            weights=offsets_loss_weight * non_empty_cell_mask,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    else:
        offsets_loss = 0.        
    
    ## Classification loss
    if 'classification_probs' in outputs:        
        assert 'group_class_labels' in inputs        
        with tf.name_scope('classification_loss'):
            classification_loss_weight = graph_manager.get_defaults(kwargs, ['classification_loss_weight'], verbose=verbose)[0]
            # labels: (batch, 1, 1, 1, num_gt, num_classes)
            labels = inputs['group_class_labels'] # (batch, num_cells, num_cells, 1, num_classes)
            # logits: (batch, num_cells, num_cells, num_preds, 1, num_classes)
            logits = outputs['classification_probs']
            # classification loss
            class_diffs = labels - logits
            classification_loss = tf.losses.compute_weighted_loss(
                class_diffs**2,
                weights=classification_loss_weight * empty_cell_mask,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    else:
        classification_loss = 0.
    
    ## Add informative summaries
    if is_chief:
        outputs["target_bounding_boxes"] = outputs["bounding_boxes"] * non_empty_cell_mask   
        mins = pred_centers - target_scales / 2 
        maxs = pred_centers + target_scales / 2 
        outputs["target_bounding_boxes_rescaled"] = tf.concat([mins, maxs], axis=-1) * non_empty_cell_mask              
                
        
    return [('%s_centers_localization_loss' % base_name, centers_localization_loss),
            ('%s_scales_localization_loss' % base_name, scales_localization_loss),
            ('%s_confidence_obj_loss' % base_name, confidence_loss_obj),
            ('%s_confidence_noobj_loss' % base_name, confidence_loss_noobj),
            ('%s_group_classification_loss' % base_name, group_classification_loss),
            ('%s_classification_loss' % base_name, classification_loss),
            ('%s_offsets_loss' % base_name, offsets_loss)]