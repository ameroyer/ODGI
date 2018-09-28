import tensorflow as tf

import graph_manager
import net
import eval_utils
import loss_utils
import tf_inputs
import tf_utils


"""Helper functions to build the train and eval graph for standard detection."""


def forward_pass(inputs, 
                 configuration,
                 scope_name='model',
                 is_training=True,
                 reuse=False, 
                 verbose=0):
    """Forward-pass in the net.
    
    Args:
        inputs: Dictionnary of inputs
        outputs: Dictionnary of outputs, to be updated
        configuration`: configuration dictionnary
        scope_name: Default scope name
        is_training: Whether the model is in training mode (for batch norm)
        reuse: whether to reuse the variable scopes
        verbose: verbosity level
    """
    network = graph_manager.get_defaults(configuration, ['network'], verbose=True)[0]
    with tf.variable_scope(scope_name, reuse=reuse):        
        if network == 'tiny-yolov2':
            activations = net.tiny_yolo_v2(
                inputs["image"], is_training=is_training, reuse=reuse, verbose=verbose, **configuration)
        elif network == 'yolov2':
            activations = net.yolo_v2(
                inputs["image"], is_training=is_training, reuse=reuse, verbose=verbose, **configuration)
        else:
            raise NotImplementedError('Uknown network architecture', network)
        return net.get_detection_outputs(activations, reuse=reuse, verbose=verbose, **configuration)
            
            
def train_pass(inputs, configuration, is_chief=False, verbose=1):
    """ Compute outputs of the net and add losses to the graph.
    
    Args:
        inputs: Dictionnary of inputs
        configuration: Configuration dictionnary
        is_chief: Whether the current training device is chief (verbosity and summaries)
        verbose: verbosity level
        
    Returns:
        Dictionnary of outputs
    """
    outputs = {}
    dev_verbose = verbose * is_chief
    base_name = graph_manager.get_defaults(configuration, ['base_name'], verbose=dev_verbose)[0]
    if dev_verbose == 2:
        print(' \033[31m> %s\033[0m' % base_name)
    elif dev_verbose == 1:
        print(' > %s' % base_name)
        
    # Feed forward
    with tf.name_scope('%s/net' % base_name):
        forward_pass(inputs, outputs, configuration, scope_name=base_name, 
                     is_training=True, reuse=not is_chief, verbose=dev_verbose) 
        
    # Add losses
    with tf.name_scope('%s/loss' % base_name):
        graph_manager.add_losses_to_graph(
            loss_utils.get_standard_loss, inputs, outputs, configuration, is_chief=is_chief, verbose=dev_verbose)
     
    # Display found losses
    if dev_verbose == 1:
        print('\n'.join("    *%s*: shape=%s, dtype=%s" % (
            key, value.get_shape().as_list(), value.dtype) for key, value in outputs.items()))
    elif dev_verbose == 2:
        print('\n'.join("    \x1b[32m*%s*\x1b[0m: shape=%s, dtype=%s" % (
            key, value.get_shape().as_list(), value.dtype) for key, value in outputs.items()))
    return outputs
        
    
def eval_pass(inputs, configuration, reuse=True, verbose=1):
    """Forward pass in test mode with Non Maximum suppression    
    
    Args:
        inputs: Dictionnary of inputs
        configuration: Configuration dictionnary
        verbose: verbosity level
        
    Returns:
        Dictionnary of outputs
    """
    base_name = graph_manager.get_defaults(configuration, ['base_name'], verbose=verbose)[0]
    if verbose == 2:
        print(' \033[31m> %s\033[0m' % base_name)
    elif verbose == 1:
        print(' > %s' % base_name)
        
    # Feed forward
    with tf.name_scope('%s/net' % base_name):
        return forward_pass(inputs, configuration, scope_name=base_name, is_training=False, 
                           reuse=reuse, verbose=verbose)
