import numpy as np
import tensorflow as tf
import tf_utils
import graph_manager


def append_individuals_detection_output(file_path,
                                        image_ids, 
                                        num_gt_boxes, 
                                        gt_boxes, 
                                        pred_boxes,
                                        pred_confidences,
                                        verbose=1,
                                        **kwargs):
    """Write outputs of the evaluation pass to a file.
    
    Args:
    """
    iou_threshold, score_threshold = graph_manager.get_defaults(
        kwargs, ['retrieval_nms_threshold', 'retrieval_confidence_threshold'], verbose=verbose)
    del kwargs
    
    with open(file_path, 'a') as f:
        for im_id, num_gt, gt, pred, pred_c in zip(image_ids, num_gt_boxes, gt_boxes, pred_boxes, pred_confidences):
            
            # first line (id, number of ground-truth, ground-truth boxes)
            f.write('%s-gt\t%d\t%s\n' % (im_id, num_gt, '\t'.join(
                ','.join('%.5f' % x for x in b) for b in gt[:num_gt])))
            
            # following lines (id, class, number of boxes, predicted boxes with scores and nms filter boolean)
            num_classes = pred_c.shape[-1]
            pred_c_flat = np.reshape(pred_c, (-1, num_classes))
            pred_boxes_flat = np.reshape(pred, (-1, 4))
            for c in range(num_classes):
                output = non_max_suppression(
                    pred_boxes_flat, pred_c_flat[:, c], iou_threshold=iou_threshold, score_threshold=score_threshold)
                f.write('%s-pred-%d\t%d\t%s\n' % (im_id, c, pred_boxes_flat.shape[0], '\t'.join(
                    '%.5f,%.5f,%.5f,%.5f,%.3f,%d' % tuple(x) for x in output)))          
                

def non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.):
    """Non maximum suppression
    
    Args:
        boxes: A (num_boxes, 4) numpy array
        scores: A (num_boxes, ) numpy array.
        iou_threshold: A float in [0., 1.]. Defaults to 0.5. ( >= threshold)
        score_threshold: A float in [0., 1.]. Defaults to 0. ( >= threshold)
        
    Returns:
        A (num_boxes, 6) array containing boxes (coordinates, confidence and nms filtering boolean) sorted 
        by decreasing confidence score
    """
    indices = np.argsort(- scores, axis=None)    
    output = np.zeros((boxes.shape[0], 6))
    nms_boxes = None 
    for it, index in enumerate(indices):
        # coordinate and scores
        box = boxes[index]
        score = scores[index]
        # score thresholding
        flag = (score >= score_threshold) 
        # NMS thresholding
        if flag:
            if it == 0:
                nms_boxes = np.expand_dims(box, axis=0)
            else: # check if intersect with one of the existing box
                iou = max_iou(box, nms_boxes)
                if iou >= iou_threshold:
                    flag = False
                else:
                    nms_boxes = np.concatenate([nms_boxes, np.expand_dims(box, axis=0)], axis=0)            
        # set
        output[it, :4] = box
        output[it, 4] = score
        output[it, 5] = flag
    # end
    return output
    
    
def max_iou(box, boxes):
    """ Maximum iou between box and all the boxes in `boxes`
    
    Args:    
        box: A (4,) numpy array
        boxes: A (num_boxes, 4) numpy array
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersections = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    return np.max(intersections)