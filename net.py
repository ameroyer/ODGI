import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from graph_manager import get_defaults
           
    
#########################################
#          Format Outputs               #
#########################################
    
def get_detection_outputs(activations,
                          is_training=False,
                          reuse=False,
                          verbose=False,
                          grid_offsets=None,
                          scope_name='format_output',
                          **kwargs):
    """ Standard detection outputs (final stage).
    
    Args:
        activations: Before-to-last convolutional outputs
        outputs: Dictionnary to store outputs
        is_training: Whether to return outputs needed for training: confidence_scores, shifted_centers, log_scales
        reuse: whether to reuse the current scope
        verbose: controls verbose output
        grid_offsets: Precomputed grid_offsets
        scope_name: Scope name
        
    Kwargs:
        with_classification: whether to predict class output. Defaults to False
        num_classes: number of classes to predict.
        num_boxes: number of boxes to predict per cells
        
    Returns:
        bounding_boxes: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 4)
        detection_scores: a Tensor of shape (batch, num_cells, num_cells, num_boxes, num_classes)
        [opt] shifted_centers: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 2)
        [opt] log_scales: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 2)
        [opt] confidence_scores: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 1)
    """
    # Set number of outputs
    assert grid_offsets is not None
    num_boxes, with_classification = get_defaults(
        kwargs, ['num_boxes', 'with_classification'], verbose=verbose)    
    if with_classification:
        num_classes = get_defaults(kwargs, ['num_classes'], verbose=verbose)[0]   
        assert num_classes > 1
    else:
        num_classes = 0
    del kwargs
         
    ## Fully connected layer
    with tf.variable_scope(scope_name, reuse=reuse):   
        # For each bounding box, outputs center coordinates, width and height, 
        # confidence, and optional classes probabilities.
        num_outputs = [2, 2, 1, num_classes]
        with tf.name_scope('conv_out'):
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)
            out = tf.layers.conv2d(activations, 
                                   num_boxes * sum(num_outputs),
                                   [1, 1], 
                                   strides=[1, 1],
                                   padding='valid',
                                   activation=None,
                                   kernel_initializer=kernel_initializer,
                                   name='out_conv')
            out = tf.stack(tf.split(out, num_boxes, axis=-1), axis=-2)   
            
        if verbose:
            print('    Output layer shape *%s*' % out.get_shape())
        out = tf.split(out, num_outputs, axis=-1, name='split_output')

    ## Format outputs
    with tf.name_scope('format_center_coordinates'):
        num_cells = tf.to_float(grid_offsets.shape[:2])
        out[0] = tf.nn.sigmoid(out[0]) + grid_offsets
        
    with tf.name_scope('format_bbs'):
        bounding_boxes = tf.concat([(out[0] - tf.exp(out[1]) / 2.) / num_cells, 
                                    (out[0] + tf.exp(out[1]) / 2.) / num_cells], axis=-1)
        bounding_boxes = tf.clip_by_value(bounding_boxes, 0., 1., name='bounding_boxes')
        
    with tf.name_scope('format_log_scales'):
        out[1] -= tf.log(num_cells)
        
    with tf.name_scope('format_confidence_scores'):
        out[2] = tf.nn.sigmoid(out[2], name='confidences_out')
        detection_scores = out[2]
        
    with tf.name_scope('format_class_probabilities'):
        if with_classification:
            out[3] = tf.nn.softmax(out[3], name='classification_probs_out')
            detection_scores *= out[3]
            
    ## Return
    return (out[0] if is_training else None, 
            out[1] if is_training else None,
            out[2],
            out[3] if with_classification else None, 
            bounding_boxes, 
            detection_scores)            
            
            
def get_detection_outputs_with_groups(activations,
                                      grid_offsets=None,
                                      is_training=True,
                                      reuse=False,
                                      verbose=False,
                                      scope_name='format_output',
                                      **kwargs):
    """ Add the final convolution and preprocess the outputs for Yolov2.
    
    Args:
        activations: Before-to-last convolutional outputs
        grid_offsets: Precomputed grid_offsets
        reuse: whether to reuse the current scope
        verbose: controls verbose output
        scope_name: Scope name
        
    Kwargs:
        with_classification: whether to predict class output.
        with_group_flags: whether to predict group flags.
        num_classes: number of classes to predict.
        num_boxes: number of boxes to predict per cells
        
    Returns:
        shifted_centers: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 2)
        log_scales: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 2)
        confidence_scores: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 1)
        offsets: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 2)
        group_flags: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 2)
        classification_probs: a Tensor of shape (batch, num_cells, num_cells, num_boxes, num_classes)
        bounding_boxes: a Tensor of shape (batch, num_cells, num_cells, num_boxes, 4)
        detection_scores: a Tensor of shape (batch, num_cells, num_cells, num_boxes, num_classes)
    """
    # Kwargs    
    assert grid_offsets is not None    
    with_classification, with_group_flags, with_offsets = get_defaults(
        kwargs, ['with_classification', 'with_group_flags', 'with_offsets'], verbose=verbose)  
    if with_classification:
        num_classes = get_defaults(kwargs, ['num_classes'], verbose=verbose)[0]   
        assert num_classes > 1
    else:
        num_classes = 0
    del kwargs
    
    ## Fully connected layer
    with tf.variable_scope(scope_name, reuse=reuse):
        # For each cell, outputs center coordinates, width and height, 
        # confidence, group flag, optional offsets and optional classes probabilities.
        num_outputs = [2, 2, 1, 1, 2 if with_offsets else 0, num_classes]
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)
        out = tf.layers.conv2d(activations, sum(num_outputs),
                               [1, 1], 
                               strides=[1, 1],
                               padding='valid',
                               activation=None,
                               kernel_initializer=kernel_initializer,
                               name='out_conv')
        out = tf.stack(tf.split(out, 1, axis=-1), axis=-2)            
            
        # Split: centers, log_scales, confidences, offsets, group_flags, class_confidences 
        if verbose:
            print('    Output layer shape *%s*' % out.get_shape())
        out = tf.split(out, num_outputs, axis=-1, name='split_output')

    ## Format outputs
    with tf.name_scope('format_center_coordinates'):
        num_cells = tf.to_float(grid_offsets.shape[:2])
        out[0] = tf.nn.sigmoid(out[0]) +  grid_offsets

    with tf.name_scope('format_bbs'):
        bounding_boxes = tf.concat([(out[0] - tf.exp(out[1]) / 2.) / num_cells, 
                                    (out[0] + tf.exp(out[1]) / 2.) / num_cells], 
                                   axis=-1, name='bounding_boxes_out')
        bounding_boxes = tf.clip_by_value(bounding_boxes, 0., 1., name='bounding_boxes')
        
    with tf.name_scope('format_log_scales'):
        out[1] -= tf.log(num_cells)
        
        
    with tf.name_scope('format_confidence_scores'):
        out[2] = tf.nn.sigmoid(out[2], name='confidences_out')
        detection_scores = out[2]
        
    with tf.name_scope('format_log_scales'):
        out[3] = tf.identity(out[3], name='flags_logits_out')
        
    with tf.name_scope('format_offsets'):
        if with_offsets:
            out[3] = tf.nn.sigmoid(out[4], name='offsets_out')
            
    with tf.name_scope('format_class_probabilities'):
        if with_classification:
            out[5] = tf.nn.softmax(out[5], name='classification_probs_out')
            detection_scores *= out[5]
            
    ## Return
    return (out[0] if is_training else None, 
            out[1] if is_training else None,
            out[2],
            out[3],
            out[4] if with_offsets else None,
            out[5] if with_classification else None, 
            bounding_boxes, 
            detection_scores)
    
    
#########################################
#              BACKBONES                #
#########################################

def tiny_yolo_v2(images,
                 is_training=True,
                 reuse=False,
                 verbose=False,
                 scope_name='tinyyolov2_net',
                 **kwargs):
    """ Base tiny-YOLOv2 architecture.
    Based on https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg
    
    Args:
        images: input images in [0., 1.]
        is_training: training bool for batch norm
        scope_name: scope name
        reuse: whether to reuse the scope
        verbose: verbosity level
        
    Kwargs:
        weight_decay: Regularization constant. Defaults to 0.
        normalizer_decay: Batch norm decay. Defaults to 0.9
    """
    # kwargs:
    weight_decay, normalizer_decay = get_defaults(
        kwargs, ['weight_decay', 'normalizer_decay'], verbose=verbose)
    del kwargs
              
    # Config
    num_filters = [16, 32, 64, 128, 256, 512, 1024]
    weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    activation_fn = lambda x: tf.nn.leaky_relu(x, 0.1)
    normalizer_fn = slim.batch_norm
    normalizer_params = {'is_training': is_training, 'decay': normalizer_decay}
    
    # Network
    with tf.variable_scope(scope_name, reuse=reuse):
        # Convolutions
        with slim.arg_scope([slim.conv2d],
                            stride=1, 
                            padding='VALID',
                            weights_initializer=weights_initializer,
                            weights_regularizer=weights_regularizer,
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params):      
            with slim.arg_scope([slim.max_pool2d], padding='SAME'):     

                # Input in [0., 1.]
                with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
                    with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
                        net = images

                        # Convolutions 
                        kernel_size = 3
                        pad = kernel_size // 2
                        paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
                        pool_strides = [2] * (len(num_filters) - 2) + [1,   0]
                        for i, num_filter in enumerate(num_filters):
                            net = tf.pad(net, paddings)
                            net = slim.conv2d(net, 
                                              num_filter, 
                                              [kernel_size, 
                                               kernel_size], 
                                              scope='conv%d' % (i + 1))
                            if pool_strides[i] > 0:
                                net = slim.max_pool2d(net,
                                                      [2, 2], 
                                                      stride=pool_strides[i], 
                                                      scope='pool%d' % (i + 1))  
                        # Last conv
                        net = tf.pad(net, paddings)
                        net = slim.conv2d(net, 512, [3, 3], scope='conv_out_2')

                        # Outputs
                        return net              
                    
                    
def yolo_v2(images,
            is_training=True,
            reuse=False,
            verbose=False,
            scope_name='yolov2_net',
            **kwargs):
    """ Base YOLOv2 architecture
    Based on https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg
    
    Args:
        images: input images in [0., 1.]
        is_training: training bool for batch norm
        scope_name: scope name
        reuse: whether to reuse the scope
        verbose: verbosity level
        
    Kwargs:
        weight_decay: Regularization cosntant. Defaults to 0.
        normalizer_decay: Batch norm decay. Defaults to 0.9
    """
    # kwargs:
    weight_decay, normalizer_decay = get_defaults(
        kwargs, ['weight_decay', 'normalizer_decay'], verbose=verbose)
    del kwargs
              
    # Config
    weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    activation_fn = lambda x: tf.nn.leaky_relu(x, 0.1)
    normalizer_fn = slim.batch_norm
    normalizer_params = {'is_training': is_training, 'decay': normalizer_decay}
    
    # Network
    with tf.variable_scope(scope_name, reuse=reuse):
        # Convolutions
        with slim.arg_scope([slim.conv2d],
                            stride=1, 
                            padding='SAME',
                            weights_initializer=weights_initializer,
                            weights_regularizer=weights_regularizer,
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params):      
            with slim.arg_scope([slim.max_pool2d], padding='SAME'):     

                # Input in [0., 1.]
                with tf.control_dependencies([tf.assert_greater_equal(images, 0.)]):
                    with tf.control_dependencies([tf.assert_less_equal(images, 1.)]):
                        net = images
                        # conv 1
                        net = slim.conv2d(net, 32, [3, 3], scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')  
                        # conv 2
                        net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')  
                        # conv 3
                        net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
                        net = slim.conv2d(net, 64, [1, 1], scope='conv3_2')
                        net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
                        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')  
                        # conv 4
                        net = slim.conv2d(net, 256, [3, 3], scope='conv4_1')
                        net = slim.conv2d(net, 128, [1, 1], scope='conv4_2')
                        net = slim.conv2d(net, 256, [3, 3], scope='conv4_3')
                        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')  
                        # conv 5
                        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
                        net = slim.conv2d(net, 256, [1, 1], scope='conv5_2')
                        net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
                        net = slim.conv2d(net, 256, [1, 1], scope='conv5_4')
                        # routing
                        route = slim.conv2d(net, 512, [3, 3], scope='conv5_5')
                        net = slim.max_pool2d(route, [2, 2], stride=2, scope='pool5') 
                        # conv 6
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv6_1')
                        net = slim.conv2d(net, 512, [1, 1], scope='conv6_2')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv6_3')
                        net = slim.conv2d(net, 512, [1, 1], scope='conv6_4')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv6_5')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv6_6')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv6_7')
                        # routing
                        route = slim.conv2d(route, 64, [3, 3], scope='conv_route')
                        route = tf.space_to_depth(route, 2)
                        net = tf.concat([net, route], axis=-1)
                        # Last conv
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv_out')

                        # Outputs
                        return net           