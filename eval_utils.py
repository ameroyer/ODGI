import numpy as np
import tensorflow as tf
import tf_utils
import graph_manager
from collections import defaultdict

#################################################### Write Output of the feed forward pass    
def append_individuals_detection_output(file_path,
                                        image_ids, 
                                        num_gt_boxes, 
                                        gt_boxes, 
                                        pred_boxes,
                                        pred_confidences,
                                        s1_boxes=None,
                                        s1_confidences=None,
                                        s1_kept_out_filter=None,
                                        **kwargs):
    """Write outputs of the evaluation pass to a file.
    
    Args:
    """
    iou_threshold, score_threshold = graph_manager.get_defaults(
        kwargs, ['retrieval_nms_threshold', 'retrieval_confidence_threshold'])
    del kwargs
    
    with open(file_path, 'a') as f:
        batch_size = image_ids.shape[0]
        num_classes = pred_confidences.shape[-1]
        for i in range(batch_size):            
            # first line (id, number of ground-truth, ground-truth boxes)
            im_id = image_ids[i]
            num_gt = num_gt_boxes[i]
            f.write('%s-gt\t%d\t%s\n' % (im_id, num_gt, '\t'.join(
                ','.join('%.5f' % x for x in b) for b in gt_boxes[i, :num_gt])))
            
            
            # following lines (id, class, number of boxes, predicted boxes with scores and nms filter boolean)            
            pred_c_flat = np.reshape(pred_confidences[i], (-1, num_classes))
            pred_boxes_flat = np.reshape(pred_boxes[i], (-1, 4))
            
            # Collect kept-out boxes from the previous stage
            if not (s1_boxes is None or s1_confidences is None or s1_kept_out_filter is None):
                assert num_classes == 1 #TODO(aroyer) group clasess
                index = np.where(s1_kept_out_filter[i])
                kept_out_boxes = np.reshape(s1_boxes[i], (-1, 4))
                kept_out_boxes = kept_out_boxes[index]
                pred_boxes_flat = np.concatenate([pred_boxes_flat, kept_out_boxes], axis=0)
                kept_out_scores = np.reshape(s1_confidences[i], (-1, 1))
                kept_out_scores = kept_out_scores[index]
                pred_c_flat = np.concatenate([pred_c_flat, kept_out_scores], axis=0)
                
            for c in range(num_classes):
                output = non_max_suppression(
                    pred_boxes_flat, pred_c_flat[:, c], iou_threshold=iou_threshold, score_threshold=score_threshold)
                f.write('%s-pred-%d\t%d\t%s\n' % (im_id, c, pred_boxes_flat.shape[0], '\t'.join(
                    '%.6f,%.6f,%.6f,%.6f,%.3f,%d' % tuple(x) for x in output)))          
                
def is_valid(box):
    return (box[..., 2] > box[..., 0]) and (box[..., 3] > box[..., 1])
    
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
    indices = np.argsort(- scores, axis=None, kind='mergesort')
    output = np.zeros((boxes.shape[0], 6))
    nms_boxes = None 
    it = 0
    for index in indices:
        # coordinates
        box = boxes[index]
        if not is_valid(box):
            continue
        # scores
        score = scores[index]
        flag = (score >= score_threshold) 
        # NMS thresholding
        if flag:
            if it == 0:
                nms_boxes = np.expand_dims(box, axis=0)
            else: # check if intersect with one of the existing box
                _, iou = max_iou(box, nms_boxes)
                if iou >= iou_threshold:
                    flag = False
                else:
                    nms_boxes = np.concatenate([nms_boxes, np.expand_dims(box, axis=0)], axis=0)            
        # set
        output[it, :4] = box
        output[it, 4] = score
        output[it, 5] = flag
        it += 1
    # end
    return output[:it]
    
def max_iou(box, boxes, epsilon=1e-12):
    """ Maximum iou between box and all the boxes in `boxes`
    
    Args:    
        box: A (4,) numpy array
        boxes: A (num_boxes, 4) or (batch, num_boxes, 4) numpy array
        
    Returns:
        A scalar or (batch,) numpy array containing the max iou and its position
    """
    assert len(boxes.shape) in [2, 3]
    x1 = np.maximum(box[0], boxes[..., 0])
    y1 = np.maximum(box[1], boxes[..., 1])
    x2 = np.minimum(box[2], boxes[..., 2])
    y2 = np.minimum(box[3], boxes[..., 3])
    intersections = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    unions = (np.maximum(box[2] - box[0], 0) * np.maximum(box[3] - box[1], 0) +
              np.maximum(boxes[..., 2] - boxes[..., 0], 0) * np.maximum(boxes[..., 3] - boxes[..., 1], 0) - 
              intersections)
    # iou: (batch, num_boxes)
    iou = intersections / (unions + epsilon)
    if len(boxes.shape) == 2:
        i = np.argmax(iou)
        return i, iou[i]
    else:
        i = np.argmax(iou, axis=-1)
        return i, np.array([iou[x, index] for x, index in enumerate(i)])


###################################################################### Map evaluation
def detect_eval(output_file_path, **kwargs):
    """PASCAL VOC-2010+ style evaluation for one class
    Based off https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py#L192
    
    Args:
        output_file_path: Path to the file written with `append_individuals_detection_output`
        kwargs will be queried for `iou_threshold`, the IOU thresholds to compute the map for
    """
    iou_threshold = graph_manager.get_defaults(kwargs, ['retrieval_iou_threshold'], verbose=False)[0]
    del kwargs
    current_image_id = 0
    ap = defaultdict(lambda: np.zeros(len(iou_threshold),))
    num_images = 0.
    with open(output_file_path, 'r') as f:
        for line in f.read().splitlines()[1:]:
            header, content = line.split('\t', 1)
            # Ground-truth boxes: 
            if header.endswith('-gt'):
                aux = content.split('\t')
                current_image_id = header.rsplit('-', 1)[0]
                # If gt_boxes are present
                if len(aux[1]):
                    gt_boxes = np.array([list(map(float, box.split(','))) for box in aux[1:]], dtype=np.float32)
                    num_gt = gt_boxes.shape[0]
                    gt_boxes = np.expand_dims(gt_boxes, axis=0) # (1, num_gt, 4)
                    num_images += 1
                else:
                    gt_boxes = None
                
            # Predictions: keeping only the nms filtered ones (already sorted)
            elif '-pred-' in header:
                # check that we've parsed the associated ground-truth already
                im_id, class_index = header.split('-pred-')
                assert current_image_id == im_id
                
                # skip empty images, nothing to be retrieved, precisions = 0 everywehre
                if gt_boxes is None:
                    continue
                
                # parse boxes
                aux = content.split('\t')                
                pred_boxes = np.array([list(map(float, box.split(',')[:4])) for box in aux[1:] if int(box[-1])], dtype=np.float32)
                
                # Match current best predictions to gt until all gt have been matched
                num_preds = pred_boxes.shape[0]
                free_gt = np.ones((len(iou_threshold), num_gt, 1))
                correct_preds = np.zeros((len(iou_threshold), num_preds,))
                for i, pred in enumerate(pred_boxes):
                    best_gt, best_iou = max_iou(pred, gt_boxes * free_gt)
                    # Found a match
                    for t, thresh in enumerate(iou_threshold):
                        if best_iou[t] > thresh:
                            free_gt[t, best_gt[t], :] = 0
                            correct_preds[t, i] = 1
                # Precisions and recalls at all points
                num_retrieved_at_k =  np.cumsum(correct_preds, axis=-1)
                precisions = num_retrieved_at_k / np.expand_dims(1. + np.arange(num_preds), axis=0)
                recalls = num_retrieved_at_k / num_gt
                # Compute AP at recall change points
                average_precision = np.sum(precisions * correct_preds, axis=-1) / num_gt
                # Append 
                ap[class_index] += average_precision
    # Return    
    ap = {k: v / num_images for k, v in ap.items()}
    return ap, iou_threshold