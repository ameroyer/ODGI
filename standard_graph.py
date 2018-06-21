import tensorflow as tf

import graph_manager
import net
import eval_utils
import loss_utils
import tf_inputs
import tf_utils


"""Helper functions to build the train and eval graph for standard detection."""


def forward_pass(inputs, 
                 outputs, 
                 configuration,
                 is_training=True,
                 reuse=False, 
                 verbose=False,
                 scope_name='model'):
    """Forward-pass in the net"""
    with tf.variable_scope(scope_name, reuse=reuse):
        activations = net.tiny_yolo_v2(
            inputs["image"], is_training=is_training, reuse=reuse, verbose=verbose, **configuration)
        net.get_detection_outputs(
            activations, outputs, reuse=reuse, verbose=verbose, **configuration)
            
            
def train_pass(inputs, configuration, is_chief=False):
    """ Compute outputs of the net and add losses to the graph"""
    outputs = {}
    base_name = graph_manager.get_defaults(configuration, ['base_name'], verbose=is_chief)[0]
    if is_chief: print(' > %s:' % base_name)
        
    # Feed forward
    with tf.name_scope('%s/net' % base_name):
        forward_pass(inputs, outputs, configuration, scope_name=base_name, 
                     is_training=True, reuse=not is_chief, verbose=is_chief) 
        
    # Add losses
    with tf.name_scope('%s/loss' % base_name):
        graph_manager.add_losses_to_graph(
            loss_utils.get_standard_loss, inputs, outputs, configuration, is_chief=is_chief, verbose=is_chief)
        
    if is_chief:
        print('\n'.join("    *%s*: shape=%s, dtype=%s" % (
            key, value.get_shape().as_list(), value.dtype) for key, value in outputs.items()))
    return outputs
        
    
def eval_pass(inputs, configuration, metrics_to_norms, clear_metrics_op, update_metrics_op, 
              device=0, is_chief=False):
    """ Compute output of the net and add metrics update and reset operations to the graph"""
    outputs = {}
    base_name = graph_manager.get_defaults(configuration, ['base_name'], verbose=is_chief)[0]
    if is_chief: print(' > %s:' % base_name)
        
    # Feed forward
    with tf.name_scope('%s/net' % base_name):
        forward_pass(inputs, outputs, configuration, scope_name=base_name, is_training=False, 
                     reuse=True, verbose=is_chief) 
        
    with tf.name_scope('%s/eval' % base_name):
        # Add number of samples counter
        graph_manager.add_metrics_to_graph(
            eval_utils.get_samples_running_counters, inputs, outputs, metrics_to_norms, clear_metrics_op, 
            update_metrics_op, configuration, device=device, verbose=is_chief) 
        # Add metrics
        graph_manager.add_metrics_to_graph(
            eval_utils.get_standard_eval, inputs, outputs, metrics_to_norms, clear_metrics_op, 
            update_metrics_op, configuration, device=device, verbose=is_chief)     
    return outputs    
