import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pickle
import time
from functools import partial

import tensorflow as tf
print("Tensorflow version", tf.__version__)

from include import configuration
from include import graph_manager
from include import nets
from include import loss_utils
from include import eval_utils
from include import viz

tee = viz.Tee() 


########################################################################## Configuration
parser = argparse.ArgumentParser(description='Standard Object Detection.')
configuration.build_base_parser(parser)
args = parser.parse_args()

print('Standard detection - %s, Input size %d\n' % (args.data, args.image_size)) 
base_config = configuration.build_base_config_from_args(args, verbose=args.verbose)

config = base_config.copy()
config['image_size'] = args.image_size
config['exp_name'] += '/%s_standard_%d' % (config['network'],config['image_size'])
configuration.finalize_grid_offsets(config)

graph_manager.generate_log_dir(config)
print('    Log directory', os.path.abspath(config["log_dir"]))
with open(os.path.join(config["log_dir"], 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)
    
def log_run():        
    global tee, config
    viz.save_tee(config["log_dir"], tee)

########################################################################## Build the graph
### templates
network = configuration.get_defaults(config, ['network'], verbose=True)[0]
forward_fn = tf.make_template(network, getattr(nets, network))
decode_fn = tf.make_template('decode', nets.get_detection_outputs)
forward_pass = partial(nets.forward, forward_fn=forward_fn, decode_fn=decode_fn)

with_summaries = config['save_summaries_steps'] is not None    
if with_summaries:
    with tf.name_scope('config_summary'):
        viz.add_text_summaries(config) 

### train
print('\nTrain Graph:')
with tf.name_scope('train'):
    with tf.name_scope('inputs'):
        inputs, _ = graph_manager.get_inputs(mode='train', verbose=args.verbose, **config) 

    for i in range(config['num_gpus']):        
        with tf.device('/gpu:%d' % i):  
            with tf.name_scope('dev_%d' % i):
                verbose = args.verbose * (i == 0)
                if verbose > 0:
                    print((' > %s' if verbose == 1 else ' \033[31m> %s\033[0m') % network)

                with tf.name_scope('feed_forward'):
                    outputs = forward_pass(
                        inputs[i]['image'], config, is_training=True, verbose=verbose) 
                                        
                if verbose > 0:
                    print((' > %s' if verbose == 1 else ' \033[31m> %s\033[0m') % 'Collecting losses')
                with tf.name_scope('losses'):
                    graph_manager.add_losses_to_graph(
                        loss_fn, inputs[i], outputs, config, is_chief=i == 0, verbose=verbose)
                with tf.name_scope('summaries'):
                    if i == 0 and with_summaries:
                        print(' > summaries:')
                        graph_manager.add_summaries(
                            inputs[i], outputs, mode='train', verbose=verbose, **config)
                        
    with tf.name_scope('losses'):
        losses = graph_manager.get_total_loss(verbose=args.verbose, with_summaries=with_summaries)
        assert len(losses) == 1
        full_loss = losses[0][0]   

    print('\nTrain op:')
    with tf.name_scope('train_op'):   
        global_step, train_op = graph_manager.get_train_op(losses, verbose=args.verbose, **config)
        assert len(train_op) == 1
        train_op = train_op[0]

### inference
with tf.name_scope('eval'):     
    eval_split_placehoder = tf.placeholder_with_default(True, (), 'choose_eval_split')
    eval_inputs, eval_initializer = tf.cond(
        eval_split_placehoder,
        true_fn=lambda: graph_manager.get_inputs(mode='test', verbose=False, **config),
        false_fn=lambda: graph_manager.get_inputs(mode='val', verbose=False, **config),
        name='eval_inputs')

    for i in range(config['num_gpus']):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('dev%d' % i):
                eval_outputs = forward_pass(
                    eval_inputs[i]['image'], config, is_training=False, verbose=verbose)
                tf.add_to_collection('inference_image_ids', eval_inputs[i]['im_id'])
                tf.add_to_collection('inference_num_boxes', eval_inputs[i]['num_boxes'])
                tf.add_to_collection('inference_gt_bbs', eval_inputs[i]['bounding_boxes'])
                tf.add_to_collection('inference_pred_bbs', eval_outputs['bounding_boxes'])
                tf.add_to_collection('inference_pred_confidences', eval_outputs['detection_scores'])

    # gather predictions across gpus
    with tf.name_scope('gather'):
        eval_outputs = [tf.concat(tf.get_collection(key), axis=0) for key in [
            'inference_image_ids', 'inference_num_boxes', 'inference_gt_bbs', 
            'inference_pred_bbs', 'inference_pred_confidences']]        
        
    # eval functions
    validation_results_path = os.path.join(config["log_dir"], 'val_output.txt')
    test_results_path = os.path.join(config["log_dir"], 'test_output.txt')

    run_eval = partial(graph_manager.run_eval, eval_split_placehoder=eval_split_placehoder,
                       eval_initializer=eval_initializer, eval_outputs=eval_outputs, configuration=config)
    eval_validation = partial(run_eval, mode='val', results_path=validation_results_path)
    eval_test = partial(run_eval, mode='test', results_path=test_results_path)
            

########################################################################## Start Session
if __name__ == '__main__':    
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6))
    log_run()            
    
    try:        
        with graph_manager.get_monitored_training_session(**config) as sess:
            print('\nStart training:')
            start_time = time.time()
            global_step_ = 0
            
            try:
                while 1:                       
                    # Train
                    global_step_, full_loss_, _ = sess.run([global_step, full_loss, train_op])

                    # Display
                    if (global_step_ - 1) % args.display_loss_very_n_steps == 0:
                        viz.display_loss(global_step_, full_loss_, start_time,
                                         config["train_num_samples_per_iter"], 
                                         config["train_num_samples"])

                    # Evaluate on validation set
                    if (config["save_evaluation_steps"] is not None and (global_step_ > 1)
                        and global_step_  % config["save_evaluation_steps"] == 0):
                        eval_validation(sess, global_step_)
                        log_run()
            except tf.errors.OutOfRangeError: # End of training
                pass              

            # Evaluate on the validation and test set 
            eval_validation(sess, global_step_)
            eval_test(sess, global_step_)
            log_run()

    except KeyboardInterrupt:          # Keyboard interrupted
        print('\nInterrupted at step %d' % global_step_)
        log_run()