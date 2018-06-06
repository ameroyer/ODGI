import io
import time
from defaults import defaults_dict

import numpy as np
import tensorflow as tf
import tf_utils


def draw_bounding_boxes(image, bbs):
    """Draw bounding boxes on the current image in Tensorflow.
    
    Args:
        image: A Tensor of shape (batch, size, size, 3).
        bbs: A Tensor of shape (batch, n_boxes, 4) where the last axis is ordered as (xmin, ymin, xmax, ymax).
    """
    with tf.name_scope('draw_bb'):
        r = len(bbs.get_shape())
        assert r >= 3
        if r == 3:
            return tf.image.draw_bounding_boxes(image, tf.gather(bbs, [1, 0, 3, 2], axis=-1))
        else:
            bbs_ = tf_utils.flatten_percell_output(bbs)
            return tf.image.draw_bounding_boxes(image, tf.gather(bbs_, [1, 0, 3, 2], axis=-1))


def draw_bounding_boxes_numpy(image, bb, width=2, color=(0, 1., 0), fill=False):
    """Draw a bounding box on the given numpy array.
    
    Args:
        image: Numpy array of shape (w, h, 3).
        bb: Bounding box of format (xmin, ymin, xmax, ymax).
        width: Line width of the drawn bounding box.
        color: Color of the drawn bounding box.    
        fill: If True, draw a filled box, otherwise only the bounding box
    """
    w, h = image.shape[:2]
    bb = (np.array(bb) * np.array([h, w, h, w])).astype(np.int)
    if fill:
        image[bb[1]:bb[3], bb[0]:bb[2]] = color
    else:
        image[bb[1]:bb[3], bb[0]:bb[0] + width] = color
        image[bb[1]:bb[3], bb[2] - width:bb[2]] = color
        image[bb[1]:bb[1] + width, bb[0]:bb[2]] = color
        image[bb[3] - width:bb[3], bb[0]:bb[2]] = color 
    
    
def get_heatmap(cov, rescale, min_cov=None, max_cov=None, min_value=0.5, epsilon=1e-8, threshold=0.):
    """Create a HSV heatmap with white values on 0, and blue/red for the extrems.
    
    Args:
        cov: A matrix of shape (batch_size, w, h, 1) representing the data
        rescale: A 2D int Tensor indicating scaling coefficient for the final image heatmap
        min_cov: Minimum value for normalizing negative values in `cov`
        max_cov: Maximum value for normalizing negative values in `cov`
        min_value: The minimum value (in HSV domain) for the colors in the heatmap
        epsilon: small float to avoid overflow
    """
    # Build heatmap viz in HSV space
    if min_cov is None:
        min_cov = tf.reduce_min(cov)
    if max_cov is None:
        max_cov = tf.reduce_max(cov)
    hue_cov = tf.where(cov >= threshold, tf.zeros(tf.shape(cov)), tf.ones(tf.shape(cov)) * 0.6)
    value_cov = tf.where(cov >= threshold, 
                         1. - (1. - min_value) * tf.minimum(1., cov / (max_cov + epsilon)),
                         1. - (1. - min_value) * tf.minimum(1., cov / (min_cov - epsilon)))
    sat_cov = tf.where(cov >= 0, 
                       tf.minimum(1., cov / max_cov * 4), 
                       tf.minimum(1., cov / min_cov * 4))
    cov = tf.concat([hue_cov, sat_cov, value_cov], axis=-1)
    cov = tf.image.hsv_to_rgb(cov)
    
    # Resize
    cov = tf.image.resize_images(cov, (tf.shape(cov)[1] * rescale[0], tf.shape(cov)[2] * rescale[1]),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return cov


def add_text_summaries(configuration, family='configuration'):
    """ Add text summary for the experiment configuration and the list of classes in the dataset.
    """
    config = defaults_dict.copy()
    config.update(configuration)
    summary_str = tf.convert_to_tensor('###Configuration \n\t'  + '\n\t'.join(
        ['"%s" = %s' % (key.upper(), config[key]) for key in sorted(config.keys()) 
         if key not in ['grid_offsets']]))
    tf.summary.text('%s/configuration' % family, summary_str, collections=['config'])    
    
    if 'data_classes' in configuration:
        summary_str = tf.convert_to_tensor('###Classes \n\t' + '\n\t'.join(
            ['%02d: %s' % (i, c) for (i, c) in enumerate(configuration['data_classes'])]))
        tf.summary.text('%s/data_classes' % family, summary_str, collections=['config'])
        
    
def add_image_summaries(inputs, 
                        outputs, 
                        num_summaries, 
                        confidence_thresholds=[0.],
                        family=None,
                        display_inputs=True,
                        collection='outputs'):
    """Add image and classification summaries
    
    Args:
        inputs: Dictionary of inputs
        outputs: Dictionary of outputs
        num_summaries: Max number of outputs to display
        confidence_thresholds: Adds summary with predicted boxes whose predicted 
            confidence is above this threshold
        family: Summary family
        display_inputs: whether to display the inputs summary
        collection: Collection key to add the summaries to
    """
    # Inputs
    if display_inputs:
        with tf.name_scope('1_inputs'):
            image = inputs['image']
            if 'obj_i_mask_bbs' in inputs:
                nonempty_cells = tf.minimum(1., tf.reduce_sum(inputs['obj_i_mask_bbs'], axis=-1))
                image_size = inputs['image'].get_shape()[1].value
                nonempty_cells = tf.image.resize_images(
                    nonempty_cells, (image_size, image_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                image -= 0.4 * image * (1. - nonempty_cells)
            if 'group_bounding_boxes_per_cell' in inputs:
                image = draw_bounding_boxes(image, inputs['group_bounding_boxes_per_cell'])
            else:
                image = draw_bounding_boxes(image, inputs['bounding_boxes'])
            image = tf.cast(image * 255., tf.uint8)
            tf.summary.image('image', image, max_outputs=num_summaries, collections=[collection], family=family)
        
    # Targets boxes (before and after offsets)
    with tf.name_scope('2_assignment'):    
        image = inputs['image']
        if 'target_bounding_boxes' in outputs:
            tf.summary.image('assigned_predictors', draw_bounding_boxes(image, outputs['target_bounding_boxes']), 
                             max_outputs=num_summaries, collections=[collection], family=family)
        if 'target_bounding_boxes_rescaled' in outputs:
            tf.summary.image(
                'assigned_predictors_with_offsets', draw_bounding_boxes(image, outputs['target_bounding_boxes_rescaled']), 
                max_outputs=num_summaries, collections=[collection], family=family)
            
    # Boxes
    with tf.name_scope('3_outputs_boxes'):    
        tiled_confs = tf.reduce_max(outputs['detection_scores'], axis=-1, keep_dims=True)            
        for ct in confidence_thresholds:
            bbs = tf.to_float(tiled_confs > ct) * outputs['bounding_boxes']
            image = draw_bounding_boxes(inputs['image'], bbs)          
            tf.summary.image('confidence_threshold=%.2f' % ct, image,
                             max_outputs=num_summaries, collections=[collection], family=family) 
        
    # Predicted crops
    with tf.name_scope('4_crops'):    
        if 'crop_boxes' in outputs:
            tf.summary.image('image', draw_bounding_boxes(inputs['image'], outputs['crop_boxes']), 
                             max_outputs=num_summaries, collections=[collection], family=family)
            
    # Group flags confusion matrix
    with tf.name_scope('5_group_flags'):
        if 'group_classification_logits' in outputs:    
            confusions = inputs['group_flags'] - tf.nn.sigmoid(outputs['group_classification_logits'])
            confusions *= tf.minimum(1., tf.reduce_sum(inputs['obj_i_mask_bbs'], axis=-1, keep_dims=True))
            confusions = tf.squeeze(confusions, axis=-1)
            tf.summary.image('confusion_matrix', get_heatmap(confusions, [3, 3], min_cov=-1., max_cov=1.), 
                             max_outputs=num_summaries, collections=[collection], family=family)


def display_graph_size(name):
    """Display the size of the current default graph in MB"""
    print('\x1b[37m%s graph: %.2f MB\x1b[0m' % (name, tf.get_default_graph().as_graph_def().ByteSize() / 10e6))    
    
                
def display_loss(loss_widget,
                 global_step_, 
                 full_loss_, 
                 start_time,
                 iter_size,
                 num_samples):
    """Display the loss in the given ipython widget.
    
    Args:
        loss_widget: An HTML widget
        global_step_: An integer specifying the current global step
        full_loss_: A float specifying the current loss value
        start_time: The time at wich execution started
        iter_size: Number of samples run per step
        num_samples: Total number of samples in the dataset
    """
    elapsed_time = time.time() - start_time
    loss_widget.value = ('<span style="color:rgb(0, 255, 255)">' + 
                         '<u>Step %d</u></span> ' % global_step_ +
                         '<i>(epoch %d)</i>: ' % ((global_step_ * iter_size) // num_samples + 1)  + 
                         '%.5f' % (full_loss_) +
                         '&nbsp;' * 10 +
                         time.strftime('%Hh %Mmn %Ss', 
                                       time.gmtime(elapsed_time))
                        )  
    assert not np.isnan(full_loss_), 'loss has NaN values'
    
    
def display_eval(loss_widget, 
                 global_step_, 
                 step, 
                 num_steps,
                 start_time):
    """Display the evaluation process in the given ipython widget.
    
    Args:
        loss_widget: An HTML widget
        global_step_: An integer specifying the current global step
        step: The current evaluation steps
        num_steps: The total number of evaluation steps to process
        start_time: The time at wich execution started
    """
    elapsed_time = time.time() - start_time
    loss_widget.value = ('<span style="color:rgb(255, 0, 255)">' + 
                         '<u>Step %d / %d (eval at %d) </u></span> ' % (
                             step, num_steps, global_step_) +
                         '&nbsp;' * 10 +
                         time.strftime('%Hh %Mmn %Ss', 
                                       time.gmtime(elapsed_time))
                        )    