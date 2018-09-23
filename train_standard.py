import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import time

import tensorflow as tf
print("Tensorflow version", tf.__version__)
from tensorflow.python.training.summary_io import SummaryWriterCache

import defaults
import eval_utils
import graph_manager
import viz
from standard_graph import *

tee = viz.Tee() 
########################################################################## Base Config
parser = argparse.ArgumentParser(description='Standard Object Detection.')
defaults.build_base_parser(parser)
args = parser.parse_args()
configuration = defaults.build_base_config_from_args(args)
graph_manager.finalize_configuration(configuration, verbose=args.verbose)
print('Standard detection - %s, Input size %d\n' % (args.data, args.size)) 

########################################################################## Network Config
standard_configuration = configuration.copy()
standard_configuration['base_name'] =  args.network
standard_configuration['image_size'] = args.size
standard_configuration['num_boxes'] = 1
standard_configuration['exp_name'] += '/%s_standard_%d' % (standard_configuration['network'], 
                                                           standard_configuration['image_size'])
graph_manager.finalize_grid_offsets(standard_configuration)


########################################################################## Graph
with tf.Graph().as_default() as graph:          
    ############################### Train
    with tf.name_scope('train'):        
        print('\nGraph:')      
        add_summaries = standard_configuration['save_summaries_steps'] is not None
        for i in range(configuration['num_gpus']): 
            is_chief = (i == 0)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('inputs%d' % i):
                    train_inputs, _ = graph_manager.get_inputs(
                        mode='train', shard_index=i, verbose=args.verbose * int(is_chief), **standard_configuration) 
                    
                with tf.name_scope('dev%d' % i):
                    train_outputs = train_pass(train_inputs, standard_configuration,
                                               is_chief=is_chief, verbose=args.verbose)   
                    if is_chief and add_summaries:
                        print(' > summaries:')
                        graph_manager.add_summaries(
                            train_inputs, train_outputs, mode='train', **standard_configuration)

        # Training Objective
        with tf.name_scope('losses'):
            losses = graph_manager.get_total_loss(verbose=args.verbose, add_summaries=add_summaries)
            full_loss = tf.add_n([x[0] for x in losses])            

        # Train op    
        with tf.name_scope('train_op'):   
            global_step, train_op = graph_manager.get_train_op(losses, verbose=args.verbose, **standard_configuration)
        
        # Additional info
        print('\nLosses:')
        print('\n'.join(["    *%s*: %s tensors" % (x, len(tf.get_collection(x)))  
                        for x in tf.get_default_graph().get_all_collection_keys() if x.endswith('_loss')]))
        if add_summaries:
            with tf.name_scope('config_summary'):
                viz.add_text_summaries(standard_configuration) 

            
    ############################### Validation and Test
    with tf.name_scope('eval'):     
        # Cannot use placeholder for max_num_bbs because tf v1.4 data does not support sparse tensors
        use_test_split = tf.placeholder_with_default(True, (), 'choose_eval_split')
        eval_inputs, eval_initializer = tf.cond(
            use_test_split,
            true_fn=lambda: graph_manager.get_inputs(mode='test', verbose=False, **standard_configuration),
            false_fn=lambda: graph_manager.get_inputs(mode='val', verbose=False, **standard_configuration),
            name='eval_inputs')
        with tf.device('/gpu:%d' % 0):
            with tf.name_scope('dev%d' % 0):
                eval_outputs = eval_pass(eval_inputs, standard_configuration, verbose=False)
                
                
########################################################################## Evaluation script
    def run_eval(sess, mode, global_step_, results_path):
        assert mode in ['val', 'test']
        feed_dict = {use_test_split: mode == 'test'}
        with open(results_path, 'w') as f:
            f.write('%s results at step %d\n' % (mode, global_step_))
        sess.run(eval_initializer, feed_dict=feed_dict)
        try:
            while 1:   
                out_ = sess.run([eval_inputs['im_id'], 
                                 eval_inputs['num_boxes'],
                                 eval_inputs['bounding_boxes'],                                             
                                 eval_outputs['bounding_boxes'],
                                 eval_outputs['detection_scores']], 
                                feed_dict=feed_dict)
                eval_utils.append_individuals_detection_output(results_path, *out_, **standard_configuration)
        except tf.errors.OutOfRangeError:
            pass
        eval_aps, eval_aps_thresholds = eval_utils.detect_eval(results_path, **standard_configuration)
        print('\r%s eval at step %d:' % (mode, global_step_), ' - '.join(
            'map@%.2f = %.5f' % (thresh, sum(x[t] for x in eval_aps.values()) / len(eval_aps))
                for t, thresh in enumerate(eval_aps_thresholds)))
                

    ########################################################################## Run Session
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6))    
    print('\nLaunch session:')
    graph_manager.generate_log_dir(standard_configuration)
    summary_writer = SummaryWriterCache.get(standard_configuration["log_dir"])
    print('    Log directory', os.path.abspath(standard_configuration["log_dir"]))
    validation_results_path = os.path.join(standard_configuration["log_dir"], 'val_output.txt')
    test_results_path = os.path.join(standard_configuration["log_dir"], 'test_output.txt')
    global_step_ = 0
    viz.save_tee(standard_configuration["log_dir"], tee)
    try:        
        with graph_manager.get_monitored_training_session(**standard_configuration) as sess:             
            print('\nStart training:')
            start_time = time.time()
            try:
                while 1:                       
                    # Train
                    global_step_, full_loss_, _ = sess.run([global_step, full_loss, train_op])
                    
                    # Display
                    if (global_step_ - 1) % args.display_loss_very_n_steps == 0:
                        viz.display_loss(None, global_step_, full_loss_, start_time, 
                                         standard_configuration["train_num_samples_per_iter"], 
                                         standard_configuration["train_num_samples"])
                
                    # Evaluate (validation)
                    if (standard_configuration["save_evaluation_steps"] is not None and (global_step_ > 1)
                        and global_step_  % standard_configuration["save_evaluation_steps"] == 0):
                        run_eval(sess, 'val', global_step_, validation_results_path)
                        viz.save_tee(standard_configuration["log_dir"], tee)
            except tf.errors.OutOfRangeError: # End of training
                pass              
            # Evaluate on the test set 
            run_eval(sess, 'test', global_step_, test_results_path)
            viz.save_tee(standard_configuration["log_dir"], tee)

    except KeyboardInterrupt:          # Keyboard interrupted
        print('\nInterrupted at step %d' % global_step_)