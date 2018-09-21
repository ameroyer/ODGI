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
from odgi_graph import *


########################################################################## Base Config
parser = argparse.ArgumentParser(description='Grouped Object Detection (ODGI).')
defaults.build_base_parser(parser)
parser.add_argument('--full_image_size', default=-1, type=int, help= 'Size of the images to extract patches.')
parser.add_argument('--stage2_batch_size', type=int, help= 'Fixed batch size.')
parser.add_argument('--stage2_momentum', type=float, default=0.9, help='Beta1 parameter for ADAM in stage 2')
args = parser.parse_args()
configuration = defaults.build_base_config_from_args(args)
graph_manager.finalize_configuration(configuration, verbose=args.verbose)
print('ODGI - %s, Input size %d\n' % (args.data, args.size)) 

########################################################################## ODGI Config
multistage_configuration = configuration.copy()
multistage_configuration['full_image_size'] = args.full_image_size
stage1_configuration = multistage_configuration.copy()
stage2_configuration = multistage_configuration.copy()

# Inputs sizes
stage1_configuration['image_size'] = args.size
stage2_configuration['image_size'] = stage1_configuration['image_size'] // 2

# stage 1
stage1_configuration['num_boxes'] = 1
stage1_configuration['base_name'] = 'stage1'
stage1_configuration['with_groups'] = True
stage1_configuration['with_group_flags'] = True
stage1_configuration['with_offsets'] = True
graph_manager.finalize_grid_offsets(stage1_configuration)

# stage 2
stage2_configuration['batch_size'] = args.stage2_batch_size
stage2_configuration['previous_batch_size'] = stage1_configuration['batch_size'] 
stage2_configuration['num_boxes'] = 1
stage2_configuration['base_name'] = 'stage2'
#stage2_configuration['beta1'] = args.stage2_momentum
graph_manager.finalize_grid_offsets(stage2_configuration)

multistage_configuration['exp_name'] += '/%s_odgi_%d_%d' % (
    stage1_configuration['network'], stage1_configuration['image_size'], stage2_configuration['image_size'])
    

########################################################################## Graph
with tf.Graph().as_default() as graph:          
    ############################### Train
    with tf.name_scope('train'):
        print('\nGraph:')     
        for i in range(configuration['num_gpus']): 
            is_chief = (i == 0)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('dev%d' % i):
                    train_inputs, _ = graph_manager.get_inputs(
                        mode='train', shard_index=i, verbose=args.verbose * int(is_chief), **stage1_configuration)
                    train_s1_outputs = train_pass(train_inputs, stage1_configuration, 
                                                  intermediate_stage=True, is_chief=is_chief, verbose=args.verbose)
                    train_s2_inputs = feed_pass(train_inputs, train_s1_outputs, stage2_configuration,
                                                mode='train', is_chief=is_chief, verbose=args.verbose)
                    train_s2_outputs = train_pass(train_s2_inputs, stage2_configuration,
                                                  intermediate_stage=False, is_chief=is_chief, verbose=args.verbose) 
                    if is_chief and args.summaries:
                        print(' > summaries:')
                        graph_manager.add_summaries(train_inputs, train_s1_outputs, mode='train', 
                                                    family="train_stage1", **stage1_configuration)
                        graph_manager.add_summaries(train_s2_inputs, train_s2_outputs, mode='train', verbose=0,
                                                    family="train_stage2", **stage2_configuration)

        # Training Objective
        with tf.name_scope('losses'):
            losses = graph_manager.get_total_loss(splits=['stage1', 'stage2'])            
            full_loss = tf.add_n([x[0] for x in losses])

        # Train op    
        with tf.name_scope('train_op'):   
            global_step, train_op = graph_manager.get_train_op(losses, verbose=args.verbose, **multistage_configuration)
        
        # Additional info
        print('\nLosses:')
        print('\n'.join(["    *%s*: %s tensors" % (x, len(tf.get_collection(x)))  
                       for x in tf.get_default_graph().get_all_collection_keys() if x.endswith('_loss')]))
        if args.summaries:
            with tf.name_scope('config_summary'):
                viz.add_text_summaries(stage1_configuration, family="stage1") 
                viz.add_text_summaries(stage2_configuration, family="stage2") 
            
    ############################### Eval
    with tf.name_scope('eval'):  
        use_test_split = tf.placeholder_with_default(True, (), 'choose_eval_split')
        eval_inputs, eval_initializer = tf.cond(
            use_test_split,
            true_fn=lambda: graph_manager.get_inputs(mode='test', verbose=False, **stage1_configuration),
            false_fn=lambda: graph_manager.get_inputs(mode='val', verbose=False, **stage1_configuration),
            name='eval_inputs')
        with tf.device('/gpu:%d' % 0):
            with tf.name_scope('dev%d' % 0):
                eval_s1_outputs = eval_pass_intermediate_stage(eval_inputs, stage1_configuration, verbose=False) 
                eval_s2_inputs = feed_pass(eval_inputs, eval_s1_outputs, stage2_configuration,
                                           mode='test', verbose=False)
                eval_s2_outputs = eval_pass_final_stage(
                    eval_s2_inputs, eval_inputs,  eval_s1_outputs, stage2_configuration, verbose=False) 
                if is_chief and args.summaries:
                    print(' > summaries:')
                    
                    fck = tf.get_collection('evaluation')
                    fck = tf.summary.merge(fck)
                

    ########################################################################## Run Session
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 
    try:
        print('\nLaunch session:')
        graph_manager.generate_log_dir(multistage_configuration)
        summary_writer = SummaryWriterCache.get(multistage_configuration["log_dir"])
        print('    Log directory', os.path.abspath(multistage_configuration["log_dir"]))
        validation_results_path = os.path.join(multistage_configuration["log_dir"], 'val_output.txt')
        
        with graph_manager.get_monitored_training_session(**multistage_configuration) as sess:    
            global_step_ = 0
            start_time = time.time()
            
            print('\nStart training:')
            while 1:
                # Train
                global_step_, full_loss_, _ = sess.run([global_step, full_loss, train_op])
                    
                if (global_step_ - 1) % args.display_loss_very_n_steps == 0:
                    viz.display_loss(None, global_step_, full_loss_, start_time,
                                     multistage_configuration["train_num_samples_per_iter"],
                                     multistage_configuration["train_num_samples"])
                                    
                # Evaluate
                if (multistage_configuration["save_evaluation_steps"] is not None and (global_step_ > 1)
                    and global_step_  % multistage_configuration["save_evaluation_steps"] == 0):
                    feed_dict = {use_test_split: False}
                    with open(validation_results_path, 'w') as f:
                        f.write('Validation results at step %d\n' % global_step_)
                    sess.run(eval_initializer, feed_dict=feed_dict)
                    try:
                        it = 0
                        while 1:
                            out_ = sess.run([eval_inputs['im_id'], 
                                             eval_inputs['num_boxes'],
                                             eval_inputs['bounding_boxes'],                                             
                                             eval_s2_outputs['bounding_boxes'],
                                             eval_s2_outputs['detection_scores'],
                                             fck], 
                                            feed_dict=feed_dict)
                            if it == 0:
                                summary_writer.add_summary(out_[-1])
                                summary_writer.flush()
                            out_ = out_[:-1]
                            eval_utils.append_individuals_detection_output(
                                validation_results_path, *out_, **multistage_configuration)
                            it += 1
                            print('\r Eval step %d' % it, end='')
                    except tf.errors.OutOfRangeError:
                        pass
                    # Compute and display map
                    val_aps, val_aps_thresholds = eval_utils.detect_eval(validation_results_path, **multistage_configuration)
                    print('\rValidation eval at step %d:' % global_step_, ' - '.join(
                        'map@%.2f = %.5f' % (thresh, sum(x[t] for x in val_aps.values()) / len(val_aps))
                        for t, thresh in enumerate(val_aps_thresholds)))
                    
    except tf.errors.OutOfRangeError: # End of training
        pass              

    except KeyboardInterrupt:          # Keyboard interrupted
        print('\nInterrupted at step %d' % global_step_)