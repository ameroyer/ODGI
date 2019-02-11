import os
import numpy as np
import tensorflow as tf


"""Utils for bounding boxes manipulation."""


def flatten_percell_output(t):
    """
    Args:
        A Tensor of shape (batch, d1,..., dn, num_outputs)
        
    Returns:
        Reshapes Tensor of shape (batch, d1 * ... * dn, num_outputs)
    """
    return tf.stack([tf.layers.flatten(x) for x in tf.unstack(t, axis=-1)], axis=-1)


def nms_with_pad(boxes, scores, num_outputs, iou_threshold=0.5, predicted_groups=None, predicted_offsets=None):
    """Returns boxes filtered by Non maximum suppresion 
        and with the batch dimension padded to `num_outputs`.
    
    Args:
        boxes: A (num_boxes, 4) Tensor of bounding boxes
        scores: A (num_boxes,) Tensor of confidences
        num_outputs: the suppression procedure will output at most `num_outputs` boxes
        iou_threshold: IoU threshold for the overlapping criterion
        
    Return:
        A (num_outputs, 4) Tensor of bounding boxes, padded if necessary and (num_outputs) Tensor of 
    """
    selected = tf.image.non_max_suppression(boxes, scores, num_outputs, iou_threshold=iou_threshold)
    new_boxes = tf.gather(boxes, selected, axis=0)
    new_scores = tf.gather(scores, selected, axis=0)
    # selected: (num_outputs, 4)
    new_boxes = tf.pad(new_boxes - np.array([1., 1., 0., 0.], dtype=np.float32),
                      ((0, num_outputs - tf.shape(new_boxes)[0]), (0, 0))
                     ) + np.array([1., 1., 0., 0.], dtype=np.float32)
    new_scores = tf.pad(new_scores, ((0, num_outputs - tf.shape(new_scores)[0]),)) 
    return new_boxes, new_scores


def rescale_with_offsets(predicted_boxes, predicted_offsets, epsilon=1e-8):  
    """Rescale boxes to a square with the given offsets
    
    Args:
        predicted_boxes: A (batch, num_cells, num_cells, num_boxes, 4) Tensor
        predicted_offsets: A (batch, num_cells, num_cells, num_boxes, 2) Tensor 
            representing offset multiplicative coefficients
        for the width and height of the predicted boxes
        
    Returns:
        A (batch, num_cells, num_cells, num_boxes, 4) Tensor of the rescaled boxes
    """
    pred_bbs = tf.unstack(predicted_boxes, num=4, axis=-1)
    pred_centers = tf.stack([pred_bbs[0] + pred_bbs[2], pred_bbs[1] + pred_bbs[3]], axis=-1) / 2.
    pred_scales = tf.stack([pred_bbs[2] - pred_bbs[0], pred_bbs[3] - pred_bbs[1]], axis=-1)
    # Rescale predictions
    target_scales = pred_scales / (predicted_offsets + epsilon)
    target_scales = tf.minimum(1., tf.reduce_max(target_scales, axis=-1, keep_dims=True))  # square size
    # Final boxes
    predicted_boxes = tf.concat([pred_centers - target_scales / 2, 
                                 pred_centers + target_scales / 2], axis=-1)
    predicted_boxes = tf.clip_by_value(predicted_boxes, 0., 1.)
    return predicted_boxes


def get_intersection(A, B):
    """ Compute the intersection between the input Tensors representing bounding boxes with format 
    (xmin, ymin, xmax, ymax).
        
        Args:
            A: List of 4 tensors of shape (a1...an).
            B: List of 4 Tensors of shape (..., a1...an).
            
        Returns: 
            Tensor of shape (..., a1...an),
            The component-wise intersections between all the bounding boxes.
    """
    x1, y1, x2, y2 = A
    p1, q1, p2, q2 = B
    intersection_x = tf.maximum(0., tf.minimum(x2, p2) - tf.maximum(x1, p1))
    intersection_y = tf.maximum(0., tf.minimum(y2, q2) - tf.maximum(y1, q1))
    return intersection_x * intersection_y


def get_intersection_ratio(A, B, epsilon=1e-8):
    """Return the component-wise intersection ratio of A relatively to B.
    
        Args:
            A: List of 4 tensors of shape (a1...an).
            B: List of 4 Tensors of shape (..., a1...an).
            epsilon: Intersection is considered empty below that threshold.
            
        Returns: 
            Tensor of shape (a1...an), component-wise intersection ratios.
    """
    intersection = get_intersection(A, B)
    area = get_area(A)
    return intersection / tf.maximum(epsilon, area)

def get_area(A):
    """Compute area of the given bounding boxes.
    
        Args:
            A: List of 4 tensors of shape (a1...an).
            
        Returns: 
            Tensor of shape (a1...an) containing the area of each bounding box.
    """
    x1, y1, x2, y2 = A
    return tf.maximum(0., x2 - x1) *  tf.maximum(0., y2 - y1)


def get_iou(A, B, epsilon=1e-8):
    """Return copmponent-wise intersection over union.
    
        Args:
            A: List of 4 Tensors of shape (a1...an).
            B: List of 4 Tensors of shape (..., a1...an).
            epsilon: Intersection is considered empty below that threshold.
            
        Returns: 
            Tensor of shape (..., a1...an), iou(A, B)
    """
    intersection = get_intersection(A, B)
    union = get_area(A) + get_area(B) - intersection
    return intersection / tf.maximum(epsilon, union)