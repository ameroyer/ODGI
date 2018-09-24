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
import odgi_graph
import standard_graph

tee = viz.Tee()    
########################################################################## Base Config
parser = argparse.ArgumentParser(description='Grouped Object Detection (ODGI).')
parser.add_argument('log_dir', type=str, help='log directory to load from')
parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
parser.add_argument('--verbose', type=int, default=2, help='Extra verbosity')
args = parser.parse_args()


########################################################################## Infer configuration from log dir name
# Data 
aux = os.path.dirname(os.path.dirname(os.path.normpath(args.log_dir)))
data = os.path.split(aux)[1]
configuration = {}
if data.startswith('vedai'):
    configuration['setting'] = data
    configuration['image_format'] = 'vedai'
elif data == 'stanford':
    configuration['setting'] = 'sdd'
    configuration['image_format'] = 'sdd'
elif data == 'sdd':
    configuration['setting'] = 'dota'
    configuration['image_format'] = 'dota'
else:
    raise ValueError("unknown data", data)
    
# Metadata
tfrecords_path = 'Data/metadata_%s.txt'
metadata = defaults.load_metadata(tfrecords_path % configuration['setting'])
configuration.update(metadata)
configuration['num_classes'] = len(configuration['data_classes'])

# Architecture
aux = os.path.split(os.path.dirname(os.path.normpath(args.log_dir)))[1]
aux = aux.split('_')
configuration['network'] = aux[0]
configuration['gpu_mem_frac'] = args.gpu_mem_frac
configuration['same_network'] = False

mode = aux[1]

with tf.Graph().as_default() as graph:
    ########################### ODGI
    if mode == 'odgi':
        stage1_configuration = configuration.copy()
        stage2_configuration = configuration.copy()

        # Read images one by one for timing purposes
        stage1_configuration['batch_size'] = 1
        stage2_configuration['batch_size'] = None 

        # Image sizes
        assert len(aux) in [4, 5]
        stage1_configuration['image_size'] = int(aux[2])
        stage2_configuration['image_size'] = int(aux[3])
        if len(aux) == 5:
            assert aux[-1] == 'same'
            configuration['same_network'] = True

        # Group flags
        stage1_configuration['base_name'] = 'stage1'
        stage1_configuration['with_groups'] = True
        stage1_configuration['with_group_flags'] = True
        stage1_configuration['with_offsets'] = True
        graph_manager.finalize_grid_offsets(stage1_configuration)
        stage2_configuration['previous_batch_size'] = stage1_configuration['batch_size'] 
        stage2_configuration['base_name'] = 'stage2'
        graph_manager.finalize_grid_offsets(stage2_configuration)

        # Graph
        with tf.name_scope('eval'):  
            eval_inputs, eval_initializer =  graph_manager.get_inputs(mode='test', verbose=False, **stage1_configuration)
            with tf.device('/gpu:%d' % 0):
                with tf.name_scope('dev%d' % 0):
                    eval_s1_outputs = odgi_graph.eval_pass_intermediate_stage(
                        eval_inputs, stage1_configuration, use_same_activations_scope=configuration['same_network'],
                        reuse=False, verbose=False) 
                    eval_s2_inputs = odgi_graph.feed_pass(
                        eval_inputs, eval_s1_outputs, stage2_configuration, mode='test', verbose=False)
                    eval_s2_outputs = odgi_graph.eval_pass_final_stage(
                        eval_s2_inputs, eval_inputs,  eval_s1_outputs, stage2_configuration, 
                        use_same_activations_scope=configuration['same_network'], reuse=False, verbose=False)                    
                    outputs = [eval_inputs['im_id'], 
                              eval_inputs['num_boxes'],
                              eval_inputs['bounding_boxes'],                       
                              eval_s2_outputs['bounding_boxes'],
                              eval_s2_outputs['detection_scores'],
                              eval_s1_outputs['bounding_boxes'],
                              eval_s1_outputs['detection_scores'],
                              eval_s1_outputs['kept_out_filter']]
                    
    ########################### Standard
    elif mode == 'standard':
        standard_configuration = configuration.copy()
        standard_configuration['batch_size'] = 1
        standard_configuration['base_name'] = configuration['network']
        standard_configuration['image_size'] = int(aux[2])
        graph_manager.finalize_grid_offsets(standard_configuration)
                    
        with tf.name_scope('eval'):     
            eval_inputs, eval_initializer = graph_manager.get_inputs(mode='test', verbose=False, **standard_configuration)
            with tf.device('/gpu:%d' % 0):
                with tf.name_scope('dev%d' % 0):
                    eval_outputs = standard_graph.eval_pass(eval_inputs, standard_configuration, reuse=False, verbose=False)
                    outputs = [eval_inputs['im_id'], 
                               eval_inputs['num_boxes'],
                               eval_inputs['bounding_boxes'],                                             
                               eval_outputs['bounding_boxes'],
                               eval_outputs['detection_scores']]

    else:
        raise ValueError('Unkown mode', mode)
                               

    ########################################################################## Start Session
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 
    print('\nLaunch session:')

    results_path = os.path.join(args.log_dir, 'timed_test_output.txt')
    gpu_mem_frac = graph_manager.get_defaults(configuration, ['gpu_mem_frac'], verbose=args.verbose)[0]    
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac), allow_soft_placement=True)
    init_iterator_op = tf.get_collection('iterator_init')
    local_init_op = tf.group(tf.local_variables_initializer(), *init_iterator_op)
    scaffold = tf.train.Scaffold(local_init_op=local_init_op)
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, checkpoint_dir=args.log_dir, config=config)

    num_samples = 0
    run_time = 0.
    run_with_write_time = 0.
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:      
        with open(results_path, 'w') as f:
            f.write('test results for model %s\n' % args.log_dir)
        sess.run(eval_initializer)
        try:
            while 1:
                num_samples += 1
                start_time = time.time()
                out_ = sess.run(outputs)
                end_time = time.time()
                eval_utils.append_individuals_detection_output(results_path, *out_, **configuration)
                end_write_time = time.time()
                run_time += end_time - start_time
                run_with_write_time += end_write_time - start_time
                print('\r Step %d' % num_samples, end='') 
        except tf.errors.OutOfRangeError:
            pass
        print()
        print('Evaluated %d samples' % num_samples)
        # Timing 
        run_time /= num_samples
        run_with_write_time /= num_samples
        print('Timings:')
        print('   Avg. Feed forward:', run_time)
        print('   Avg. Feed forward with write:', run_with_write_time)

        print()
        print('Evaluation:')
        eval_aps, eval_aps_thresholds = eval_utils.detect_eval(results_path, **configuration)
        print(' - '.join('map@%.2f = %.5f' % (thresh, sum(x[t] for x in eval_aps.values()) / len(eval_aps))
                         for t, thresh in enumerate(eval_aps_thresholds)))
