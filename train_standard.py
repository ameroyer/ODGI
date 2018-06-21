import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import time

import tensorflow as tf
print("Tensorflow version", tf.__version__)
from tensorflow.python.training.summary_io import SummaryWriterCache

import graph_manager
import viz
from standard_graph import *


########################################################################## Command line parser
parser = argparse.ArgumentParser(description='Standard Object Detection.')
parser.add_argument('data', type=str, help='Dataset to use. One of "vedai", "stanford" or "dota"')
parser.add_argument('--size', default=1024, type=int, help='size of input images')
parser.add_argument('--num_epochs', type=int, help='size of images at the first stage')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--display_loss_very_n_steps', type=int, default=200, help='Print the loss at every given step')
args = parser.parse_args()
print('Standard detection - %s, Input size %d\n' % (args.data, args.size)) 


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
    
## Training
configuration['learning_rate'] = 1e-3
configuration['centers_localization_loss_weight'] = 1.
configuration['scales_localization_loss_weight']  = 1.
configuration['confidence_loss_weight']  = 5.
configuration['noobj_confidence_loss_weight']  = 1.
configuration['offsets_loss_weight']  = 1.

## Evaluation
configuration['save_checkpoint_secs'] = 3600
configuration['retrieval_intersection_threshold'] = [0.25, 0.5, 0.75]

graph_manager.finalize_configuration(configuration, verbose=2)


########################################################################## ODGI Config
vanilla_configuration = configuration.copy()
vanilla_configuration['base_name'] =  'tinyyolov2'
# Set resolution parameter [I, J] and K
vanilla_configuration['image_size'] = args.size
vanilla_configuration['num_boxes'] = 1
# Finalize
vanilla_configuration['exp_name'] += '/yolov2_%d' % vanilla_configuration['image_size']
graph_manager.finalize_grid_offsets(vanilla_configuration)
print('Retrieval top k = %d (final)' % vanilla_configuration['retrieval_top_n'])


########################################################################## Graph
with tf.Graph().as_default() as graph:          
    ############################### Train
    with tf.name_scope('train'):
        print('\nLoad inputs:')
        inputs = graph_manager.get_inputs(mode='train', verbose=2, **vanilla_configuration)   
        
        print('\nTrain Graph:')      
        for i, train_inputs in enumerate(inputs):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('dev%d' % i):
                    is_chief = (i == 0)
                    train_outputs = train_pass(train_inputs, vanilla_configuration, is_chief=is_chief)   
                    if is_chief:
                        print(' > summaries:')
                        graph_manager.add_summaries(
                            train_inputs, train_outputs, mode='train', **vanilla_configuration)

        # Training Objective
        with tf.name_scope('losses'):
            losses = graph_manager.get_total_loss()
            full_loss = tf.add_n([x[0] for x in losses])

        # Train op    
        with tf.name_scope('train_op'):   
            global_step, train_op = graph_manager.get_train_op(losses, **vanilla_configuration)
        
        # Additional info
        with tf.name_scope('config_summary'):
            viz.add_text_summaries(vanilla_configuration) 
            print('\nLosses:')
            print('\n'.join(["    *%s*: %s tensors" % (x, len(tf.get_collection(x)))  
                            for x in tf.get_default_graph().get_all_collection_keys() if x.endswith('_loss')]))
            
    ############################### Eval
    with tf.name_scope('eval'):        
        print('\nTest Graph:')
        update_metrics_op = []    # Store operations to update the metrics
        clear_metrics_op = []     # Store operations to reset the metrics
        metrics_to_norms = {}

        inputs = graph_manager.get_inputs(mode='test', verbose=False, **vanilla_configuration)         

        for i, val_inputs in enumerate(inputs):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('dev%d' % i):
                    is_chief = (i == 0)
                    val_outputs = eval_pass(val_inputs, vanilla_configuration, metrics_to_norms, 
                                            clear_metrics_op, update_metrics_op, device=i, is_chief=is_chief) 
                    if is_chief:
                        graph_manager.add_summaries(
                            val_inputs, val_outputs, mode='test', **vanilla_configuration)

        with tf.name_scope('eval'):
            print('    %d eval update ops' % len(update_metrics_op))
            print('    %d eval clear ops' % len(clear_metrics_op))
            update_metrics_op = tf.group(*update_metrics_op)
            clear_metrics_op = tf.group(*clear_metrics_op)
            eval_summary_op, metrics = graph_manager.get_eval_op(metrics_to_norms, output_values=True)
            accuracy = metrics['tinyyolov2_avgprec_at0.50_eval']

        # Additional info
        print('\nEval metrics:')
        print('\n'.join(["    *%s*: %s tensors" % (x, len(tf.get_collection(x)))  
                        for x in tf.get_default_graph().get_all_collection_keys() 
                        if x.endswith('_eval')]))

    ########################################################################## Run Session
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6))    
    try:
        print('\nLaunch session:')
        graph_manager.generate_log_dir(vanilla_configuration)
        summary_writer = SummaryWriterCache.get(vanilla_configuration["log_dir"])
        print('    Log directory', os.path.abspath(vanilla_configuration["log_dir"]))
        
        with graph_manager.get_monitored_training_session(**vanilla_configuration) as sess:
            global_step_ = 0   
            start_time = time.time()
            
            print('\nStart training:')
            while not sess.should_stop(): 
                        
                # Train
                global_step_, full_loss_, _ = sess.run([global_step, full_loss, train_op])
                
                # Evaluate
                if (vanilla_configuration["save_evaluation_steps"] is not None and (global_step_ > 1)
                    and global_step_  % vanilla_configuration["save_evaluation_steps"] == 0):
                    sess.run(clear_metrics_op)
                    num_epochs = vanilla_configuration["test_num_iters_per_epoch"]
                    for epoch in range(num_epochs):
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
                                     vanilla_configuration["train_num_samples_per_iter"], 
                                     vanilla_configuration["train_num_samples"])
                
    except KeyboardInterrupt:
        print('\nInterrupted at step %d' % global_step_)