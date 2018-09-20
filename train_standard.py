import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import time

import tensorflow as tf
print("Tensorflow version", tf.__version__)
from tensorflow.python.training.summary_io import SummaryWriterCache

import eval_utils
import graph_manager
import viz
from standard_graph import *


########################################################################## Command line parser
parser = argparse.ArgumentParser(description='Standard Object Detection.')
parser.add_argument('data', type=str, help='Dataset to use. One of "vedai", "stanford" or "dota"')
parser.add_argument('--network', type=str, default="tiny-yolov2", help='network. One of "tiny-yolov2" or "yolov2"')
parser.add_argument('--size', default=1024, type=int, help='size of input images')
parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs workers to use')
parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--display_loss_very_n_steps', type=int, default=200, help='Print the loss at every given step')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--summaries', action='store_true', help='Save Tensorboard summaries while training')
parser.add_argument('--verbose', type=int, default=2, help='Extra verbosity')
args = parser.parse_args()
print('Standard detection - %s, Input size %d\n' % (args.data, args.size)) 


########################################################################## Main Config
## Configuration
configuration = {}
configuration['network'] = args.network
data = args.data
if data == 'vedai':
    configuration['setting'] = 'vedai'
    configuration['exp_name'] = 'vedai'
    configuration['save_summaries_steps'] = 100    # Save tensorboard summaries
    configuration['save_evaluation_steps'] = 500   # Run evaluation on the validation set
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
    
## Add dataset specific configuration stored in metadata
tfrecords_path = 'Data/metadata_%s.txt'
metadata = graph_manager.load_metadata(tfrecords_path % configuration['setting'])
configuration.update(metadata)
configuration['num_classes'] = len(configuration['data_classes'])

## GPUs
configuration['num_gpus'] = args.num_gpus               
configuration['gpu_mem_frac'] = max(0., min(1., args.gpu_mem_frac))
    
## Training
configuration['batch_size'] = args.batch_size
configuration['learning_rate'] = args.learning_rate
configuration['centers_localization_loss_weight'] = 1.
configuration['scales_localization_loss_weight']  = 1.
configuration['confidence_loss_weight']  = 5.
configuration['noobj_confidence_loss_weight']  = 1.
configuration['offsets_loss_weight']  = 1.

graph_manager.finalize_configuration(configuration, verbose=args.verbose)

## Network configuration
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
        for i in range(configuration['num_gpus']): 
            is_chief = (i == 0)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('inputs%d' % i):
                    train_inputs, _ = graph_manager.get_inputs(
                        mode='train', shard_index=i, verbose=args.verbose * int(is_chief), **standard_configuration) 
                    
                with tf.name_scope('dev%d' % i):
                    train_outputs = train_pass(train_inputs, standard_configuration,
                                               is_chief=is_chief, verbose=args.verbose)   
                    if is_chief and args.summaries:
                        print(' > summaries:')
                        graph_manager.add_summaries(
                            train_inputs, train_outputs, mode='train', **standard_configuration)

        # Training Objective
        with tf.name_scope('losses'):
            losses = graph_manager.get_total_loss(verbose=args.verbose, add_summaries=args.summaries)
            full_loss = tf.add_n([x[0] for x in losses])            

        # Train op    
        with tf.name_scope('train_op'):   
            global_step, train_op = graph_manager.get_train_op(losses, verbose=args.verbose, **standard_configuration)
        
        # Additional info
        print('\nLosses:')
        print('\n'.join(["    *%s*: %s tensors" % (x, len(tf.get_collection(x)))  
                        for x in tf.get_default_graph().get_all_collection_keys() if x.endswith('_loss')]))
        if args.summaries:
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
                

    ########################################################################## Run Session
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6))    
    try:
        print('\nLaunch session:')
        graph_manager.generate_log_dir(standard_configuration)
        summary_writer = SummaryWriterCache.get(standard_configuration["log_dir"])
        print('    Log directory', os.path.abspath(standard_configuration["log_dir"]))
        validation_results_path = os.path.join(standard_configuration["log_dir"], 'val_output.txt')
        
        with graph_manager.get_monitored_training_session(**standard_configuration) as sess:
            global_step_ = 0   
            start_time = time.time()
            
            print('\nStart training:')
            # Until end of iterator
            while 1:                       
                # Train
                global_step_, full_loss_, _ = sess.run([global_step, full_loss, train_op])
                    
                # Display
                if (global_step_ - 1) % args.display_loss_very_n_steps == 0:
                    viz.display_loss(None, global_step_, full_loss_, start_time, 
                                     standard_configuration["train_num_samples_per_iter"], 
                                     standard_configuration["train_num_samples"])
                
                # Evaluate
                if (standard_configuration["save_evaluation_steps"] is not None and (global_step_ > 1)
                    and global_step_  % standard_configuration["save_evaluation_steps"] == 0):
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
                                             eval_outputs['bounding_boxes'],
                                             eval_outputs['detection_scores']], 
                                            feed_dict=feed_dict)
                            eval_utils.append_individuals_detection_output(
                                validation_results_path, *out_, **standard_configuration)
                            it += 1
                            print('\r Eval step %d' % it, end='')
                    except tf.errors.OutOfRangeError:
                        pass
                    # Compute and display map
                    val_aps, val_aps_thresholds = eval_utils.detect_eval(validation_results_path, **standard_configuration)
                    print('\rValidation eval at step %d:' % global_step_, ' - '.join(
                        'map@%.2f = %.5f' % (thresh, sum(x[t] for x in val_aps.values()) / len(val_aps))
                        for t, thresh in enumerate(val_aps_thresholds)))
    except tf.errors.OutOfRangeError: # End of training
        pass              

    except KeyboardInterrupt:          # Keyboard interrupted
        print('\nInterrupted at step %d' % global_step_)