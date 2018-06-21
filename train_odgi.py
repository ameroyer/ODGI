import os
from math import ceil

import argparse
import time

import numpy as np
import tensorflow as tf
print("Tensorflow version", tf.__version__)
from tensorflow.python.training.summary_io import SummaryWriterCache

import graph_manager
import viz
from odgi_graph import *

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)


########################################################################## Command line parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data', type=str, help='Dataset to use. One of "vedai", "stanford" or "dota"')
parser.add_argument('--size', default=512, type=int, help='size of images at the first stage')
parser.add_argument('--num_epochs', type=int, help='size of images at the first stage')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--display_loss_very_n_steps', type=int, default=200, help='Print the loss at every given step')
args = parser.parse_args()
print('ODGI - %s, Input size %d' % (args.data, args.size)) 


########################################################################## Main Config
## Set dataset
configuration = {}
data = args.data
if data == 'vedai':
    configuration['setting'] = 'vedai'
    configuration['exp_name'] = 'vedai'
    configuration['save_summaries_steps'] = 100
    configuration['save_evaluation_steps'] = 250
    configuration['num_epochs'] = 1000 if args.num_epochs is None else args.num_epochs
elif data == 'stanford':
    configuration['setting'] = 'sdd'
    configuration['exp_name'] = 'sdd'
    configuration['save_summaries_steps'] = 200
    configuration['save_evaluation_steps'] = 500
    configuration['num_epochs'] = 120 if args.num_epochs is None else args.num_epochs
elif data == 'dota':
    configuration['setting'] = 'dota'
    configuration['exp_name'] = 'dota'
    configuration['save_summaries_steps'] = 500
    configuration['save_evaluation_steps'] = 1000
    configuration['num_epochs'] = 100 if args.num_epochs is None else args.num_epochs
    
## Metadata
tfrecords_path = 'Data/metadata_%s.txt'
metadata = graph_manager.load_metadata(tfrecords_path % configuration['setting'])
configuration.update(metadata)
configuration['num_classes'] = len(configuration['data_classes'])

## GPUs
configuration['num_gpus'] = args.num_gpus                                 
configuration['gpu_mem_frac'] = max(0., min(1., args.gpu_mem_frac))

## Inputs Pipeline
configuration['batch_size'] = args.batch_size
configuration['test_batch_size'] = args.batch_size
configuration['shuffle_buffer'] = 2000
configuration['subset'] = -1
    
## Evaluation
configuration['save_checkpoint_secs'] = 3600
configuration['retrieval_intersection_threshold'] = [0.25, 0.5, 0.75]

## Training
configuration['learning_rate'] = 1e-3
configuration['centers_localization_loss_weight'] = 1.
configuration['scales_localization_loss_weight']  = 1.
configuration['confidence_loss_weight']  = 5.
configuration['noobj_confidence_loss_weight']  = 1.
configuration['group_classification_loss_weight']  = 1.
configuration['offsets_loss_weight']  = 1.
graph_manager.finalize_configuration(configuration)


########################################################################## ODGI Config
multistage_configuration = configuration.copy()
multistage_configuration['full_image_size'] = 1024
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
stage2_configuration['num_boxes'] = 1
stage2_configuration['base_name'] = 'stage2'
graph_manager.finalize_grid_offsets(stage2_configuration, finalize_retrieval_top_n=False)
multistage_configuration['exp_name'] += '/odgi_%d_%d' % (
    stage1_configuration['image_size'], stage2_configuration['image_size'])

# Compute the final number of outputs (need to define k in topk for final evaluation)
num_outputs_stage1 = (stage1_configuration['num_boxes'] *  stage1_configuration['num_cells'][0] * 
                      stage1_configuration['num_cells'][1])
num_outputs_stage2 = (stage2_configuration['num_boxes'] * stage2_configuration['num_cells'][0] * 
                      stage2_configuration['num_cells'][1])
num_crops, retrieval_top_n = graph_manager.get_defaults(
    stage2_configuration, ['test_num_crops', 'retrieval_top_n'], verbose=False)
num_outputs_final = num_crops * num_outputs_stage2 + num_outputs_stage1
stage2_configuration['retrieval_top_n'] = min(retrieval_top_n, num_outputs_final)
print('Retrieval top k = %d (final)' % stage2_configuration['retrieval_top_n'])
    

########################################################################## Graph
with tf.Graph().as_default() as graph:          
    ############################### Train
    with tf.name_scope('train'):
        print('\n\033[44mLoad inputs:\033[0m')
        inputs = graph_manager.get_inputs(mode='train', verbose=True, **stage1_configuration)   
        
        print('\n\033[43mTrain Graph:\033[0m')
        viz.display_graph_size('inputs(train)')        
        for i, train_inputs in enumerate(inputs):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('dev%d' % i):
                    is_chief = (i == 0)
                    train_s1_outputs = train_pass(train_inputs, stage1_configuration, 
                                                  intermediate_stage=True, is_chief=is_chief)    
                    train_s2_inputs = feed_pass(train_inputs, train_s1_outputs, stage2_configuration,
                                                mode='train', is_chief=is_chief)
                    train_s2_outputs = train_pass(train_s2_inputs, stage2_configuration,
                                                  intermediate_stage=False, is_chief=is_chief) 
                    if is_chief:
                        print(' \033[34msummaries:\033[0m')
                        graph_manager.add_summaries(train_inputs, train_s1_outputs, mode='train', 
                                                    family="train_stage1", **stage1_configuration)
                        graph_manager.add_summaries(train_s2_inputs, train_s2_outputs, mode='train', verbose=0,
                                                    family="train_stage2", **stage2_configuration)
            viz.display_graph_size('train net (gpu:%d)' % i)

        # Training Objective
        with tf.name_scope('losses'):
            losses = graph_manager.get_total_loss(splits=['stage1', 'stage2'])            
            full_loss = tf.add_n([x[0] for x in losses])
        viz.display_graph_size('full loss')

        # Train op    
        with tf.name_scope('train_op'):   
            global_step, train_op = graph_manager.get_train_op(losses, **multistage_configuration)
        viz.display_graph_size('train op')
        
        # Additional info
        with tf.name_scope('config_summary'):
            viz.add_text_summaries(stage1_configuration, family="stage1") 
            viz.add_text_summaries(stage2_configuration, family="stage2") 
            print('\n\033[43mLosses:\033[0m')
            print('\n'.join(["    \033[35m%s:\033[0m %s tensors" % (x, len(tf.get_collection(x)))  
                            for x in tf.get_default_graph().get_all_collection_keys() 
                            if x.endswith('_loss')]))
            
    ############################### Eval
    with tf.name_scope('eval'):        
        print('\n\033[43mVal Graph:\033[0m')
        update_metrics_op = []    # Store operations to update the metrics
        clear_metrics_op = []     # Store operations to reset the metrics
        metrics_to_norms = {}
        inputs = graph_manager.get_inputs(mode='test', verbose=False, **stage1_configuration)    
        
        for i, val_inputs in enumerate(inputs):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('dev%d' % i):
                    is_chief = (i == 0)
                    val_s1_outputs = eval_pass_intermediate_stage(
                        val_inputs, stage1_configuration, metrics_to_norms, clear_metrics_op,
                        update_metrics_op, device=i, is_chief=is_chief) 
                    val_s2_inputs = feed_pass(val_inputs, val_s1_outputs, stage2_configuration,
                                              mode='test', is_chief=is_chief)
                    val_s2_outputs, val_s2_unscaled_outputs = eval_pass_final_stage(
                        val_s2_inputs, val_inputs,  val_s1_outputs, stage2_configuration, metrics_to_norms, 
                        clear_metrics_op, update_metrics_op, device=i, is_chief=is_chief)
                    
                    if is_chief:
                        with tf.name_scope('stage1'):
                            graph_manager.add_summaries(val_inputs, val_s1_outputs, mode='test', 
                                                        **stage1_configuration)   
                        with tf.name_scope('stage2'):
                            graph_manager.add_summaries(val_s2_inputs, val_s2_unscaled_outputs, mode='test',
                                                        verbose=False, **stage2_configuration) 
                        with tf.name_scope('total_pipeline'):
                            graph_manager.add_summaries(val_inputs, val_s2_outputs, mode='test', verbose=False,
                                                        display_inputs=False, **stage2_configuration)

        with tf.name_scope('eval'):
            print('    \x1b[32m%d\x1b[0m eval update ops' % len(update_metrics_op))
            print('    \x1b[32m%d\x1b[0m eval clear ops' % len(clear_metrics_op))
            update_metrics_op = tf.group(*update_metrics_op)
            clear_metrics_op = tf.group(*clear_metrics_op)
            eval_summary_op, metrics = graph_manager.get_eval_op(metrics_to_norms, output_values=True)
            accuracy = metrics['stage2_avgprec_at0.50_eval']
        
        # Additional info
        print('\n\033[43mEval metrics:\033[0m')
        print('\n'.join(["    \033[35m%s:\033[0m %s tensors" % (x, len(tf.get_collection(x)))  
                        for x in tf.get_default_graph().get_all_collection_keys() 
                        if x.endswith('_eval')]))
        print('total graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 

    ########################################################################## Run Session
    try:
        print('\n\033[44mLaunch session:\033[0m')
        graph_manager.generate_log_dir(multistage_configuration)
        summary_writer = SummaryWriterCache.get(multistage_configuration["log_dir"])
        print('    Log directory', os.path.abspath(multistage_configuration["log_dir"]))
        
        with graph_manager.get_monitored_training_session(**multistage_configuration) as sess:    
            global_step_ = 0
            start_time = time.time()
            
            print('\n\033[44mStart training:\033[0m')
            last_eval_step = 1
            while not sess.should_stop(): 
                    
                # Train
                global_step_, full_loss_, _ = sess.run([global_step, full_loss, train_op])
                
                # Evaluate
                if (multistage_configuration["save_evaluation_steps"] is not None and (global_step_ > 1)
                    and global_step_  % multistage_configuration["save_evaluation_steps"] == 0):
                    num_epochs = multistage_configuration["test_num_iters_per_epoch"]
                    sess.run(clear_metrics_op)
                    for epoch in range(num_epochs):
                        sess.run(update_metrics_op) 
                        if epoch == num_epochs - 1: 
                            _, accuracy_, eval_summary = sess.run([update_metrics_op, accuracy, eval_summary_op])
                        else:
                            sess.run(update_metrics_op) 
                    # Write summary
                    summary_writer.add_summary(eval_summary, global_step_)
                    summary_writer.flush()
                    print('   > Eval at step %d: map@0.5 = %.3f' % (global_step_, accuracy_)) 
                    
                # Display
                if (global_step_ - 1) % args.display_loss_very_n_steps == 0:
                    viz.display_loss(None, global_step_, full_loss_, start_time,
                                     multistage_configuration["train_num_samples_per_iter"],
                                     multistage_configuration["train_num_samples"])
                
    except KeyboardInterrupt:
        print('\nInterrupted at step %d' % global_step_)