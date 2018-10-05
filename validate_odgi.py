import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import time

import tensorflow as tf
print("Tensorflow version", tf.__version__)

import defaults
import eval_utils
import graph_manager
import viz
import odgi_graph


tee = viz.Tee(filename='sweep_log.txt')  
########################################################################## Base Config
parser = argparse.ArgumentParser(description='Hyperparameter sweep on the validation set.')
parser.add_argument('log_dir', type=str, help='log directory to load from')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
args = parser.parse_args()

# Sweeps
test_num_crops_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                
test_patch_nms_threshold_sweep = [0.25, 0.5, 0.75]                      
test_patch_confidence_threshold_sweep  = [0., 0.1, 0.2, 0.3, 0.4]
test_patch_strong_confidence_threshold_sweep = [0.6, 0.7, 0.8, 0.9, 1.0]

########################################################################## Infer configuration from log dir name
# Data 
aux = os.path.dirname(os.path.dirname(os.path.normpath(args.log_dir)))
data = os.path.split(aux)[1]
configuration = {}
if data.startswith('vedai'):
    configuration['setting'] = data
    configuration['image_format'] = 'vedai'
elif data == 'sdd':
    configuration['setting'] = 'sdd'
    configuration['image_format'] = 'sdd'
elif data == 'dota':
    configuration['setting'] = 'dota'
    configuration['image_format'] = 'dota'
else:
    raise ValueError("unknown data", data)
    
# Metadata
tfrecords_path = 'Data/metadata_%s.txt'
metadata = defaults.load_metadata(tfrecords_path % configuration['setting'])
configuration.update(metadata)

# Architecture
aux = os.path.split(os.path.dirname(os.path.normpath(args.log_dir)))[1]
aux = aux.split('_')
configuration['network'] = aux[0]
configuration['gpu_mem_frac'] = args.gpu_mem_frac
mode = aux[1]

assert mode == 'odgi'
assert len(aux) in [4, 5]
stage1_configuration = configuration.copy()
stage2_configuration = configuration.copy()

# Batch size
stage1_configuration['batch_size'] = args.batch_size
stage2_configuration['batch_size'] = None 

# Image sizes
stage1_configuration['image_size'] = int(aux[2])
stage2_configuration['image_size'] = int(aux[3])

# Group flags
stage1_configuration['base_name'] = 'stage1'
stage1_configuration['with_groups'] = True
stage1_configuration['with_group_flags'] = True
stage1_configuration['with_offsets'] = True
graph_manager.finalize_grid_offsets(stage1_configuration, verbose=0)
stage2_configuration['previous_batch_size'] = stage1_configuration['batch_size'] 
stage2_configuration['base_name'] = 'stage2'
graph_manager.finalize_grid_offsets(stage2_configuration, verbose=0)

# Sweep parameters over the validation set, save best hypoerparams for each num_crops
best_val_map = {k: 0. for k in test_num_crops_sweep}
best_test_num_crops = {k: None for k in test_num_crops_sweep}
best_test_patch_nms_threshold = {k: None for k in test_num_crops_sweep}
best_test_patch_confidence_threshold = {k: None for k in test_num_crops_sweep}
best_test_patch_strong_confidence_threshold = {k: None for k in test_num_crops_sweep}


def build_graph(nms_threshold, max_test_num_crops=test_num_crops_sweep[-1]):
    """Build the graph.
    Note that nms threshold has to be defined as a float and not a placeholder
    """
    # Parameters to sweep on as placeholder
    test_num_crops = tf.placeholder(tf.int32, shape=())                                    
    test_patch_confidence_threshold = tf.placeholder(tf.float32, shape=()) 
    test_patch_strong_confidence_threshold = tf.placeholder(tf.float32, shape=()) 

    # crop extractio
    stage1_configuration["test_num_crops"] = max_test_num_crops
    stage2_configuration["test_num_crops_slice"] = test_num_crops                               
    stage1_configuration["test_patch_nms_threshold"] = test_patch_nms_threshold_                
    stage1_configuration["test_patch_confidence_threshold"] = test_patch_confidence_threshold 
    stage1_configuration["test_patch_strong_confidence_threshold"] = test_patch_strong_confidence_threshold

    # Graph
    use_test_split = tf.placeholder_with_default(True, (), 'choose_eval_split')
    with tf.device('/gpu:0'):
        eval_inputs, eval_initializer = tf.cond(
            use_test_split,
            true_fn=lambda: graph_manager.get_inputs(mode='test', verbose=False, **stage1_configuration),
            false_fn=lambda: graph_manager.get_inputs(mode='val', verbose=False, **stage1_configuration),
            name='eval_inputs')
        eval_s1_outputs = odgi_graph.eval_pass_intermediate_stage(eval_inputs,
                                                                  stage1_configuration, 
                                                                  reuse=False, 
                                                                  verbose=False) 
        eval_s2_inputs = odgi_graph.feed_pass(eval_inputs, 
                                              eval_s1_outputs["crop_boxes"],
                                              stage2_configuration, 
                                              mode='test', 
                                              verbose=False)
        eval_s2_outputs = odgi_graph.eval_pass_final_stage(eval_s2_inputs, 
                                                           eval_s1_outputs["crop_boxes"], 
                                                           stage2_configuration,
                                                           reuse=False,
                                                           verbose=False)
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 

    # Evaluation function
    def run_eval(sess, results_path, feed_dict):
        with open(results_path, 'w') as f:
            f.write('results\n')
        sess.run(eval_initializer, feed_dict=feed_dict)
        try:
            num_useful_crops = []
            while 1:
                out_ = sess.run([eval_inputs['im_id'], 
                                 eval_inputs['num_boxes'],
                                 eval_inputs['bounding_boxes'],                                             
                                 eval_s2_outputs['bounding_boxes'],
                                 eval_s2_outputs['detection_scores'],
                                 eval_s1_outputs['bounding_boxes'],
                                 eval_s1_outputs['detection_scores'],
                                 eval_s1_outputs['kept_out_filter'],
                                 eval_s2_outputs['num_useful_crops']], 
                                feed_dict=feed_dict)
                num_useful_crops.extend(out_[-1])
                eval_utils.append_individuals_detection_output(results_path, *out_[:-1], **configuration)
        except tf.errors.OutOfRangeError:
            pass
        eval_aps, eval_aps_thresholds = eval_utils.detect_eval(results_path, **configuration)
        maps = [sum(x[t] for x in eval_aps.values()) / len(eval_aps) for t, thresh in enumerate(eval_aps_thresholds)]
        print('%.3f used crops,' % (sum(num_useful_crops) / len(num_useful_crops)), ' - '.join(
            'map@%.2f = %.5f' % (thresh, m) for (thresh, m) in zip(eval_aps_thresholds, maps)))
        # return map@0.5
        return maps[0]
    
    # return
    return (use_test_split,
            test_num_crops, 
            test_patch_confidence_threshold,
            test_patch_strong_confidence_threshold, 
            run_eval)

# Session creator
gpu_mem_frac = graph_manager.get_defaults(configuration, ['gpu_mem_frac'], verbose=0)[0] 
if gpu_mem_frac < 1.0:
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac), allow_soft_placement=True)
else:
    config = tf.ConfigProto(allow_soft_placement=True)
results_path = os.path.join(args.log_dir, 'validate_temp.txt')


# test_patch_nkms_threshold has to live as afloat not a placeholder
for test_patch_nms_threshold_ in test_patch_nms_threshold_sweep:    
    with tf.Graph().as_default() as graph:
        (use_test_split, test_num_crops, test_patch_confidence_threshold,
         test_patch_strong_confidence_threshold, run_eval) = build_graph(
            test_patch_nms_threshold_, max_test_num_crops=test_num_crops_sweep[-1])

        ########################################################################## Start Session
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir, config=config)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for test_num_crops_ in test_num_crops_sweep:                                                
                for test_patch_confidence_threshold_ in test_patch_confidence_threshold_sweep:
                    for test_patch_strong_confidence_threshold_ in test_patch_strong_confidence_threshold_sweep:
                        # feed_dict
                        feed_dict = {use_test_split: False,
                                     test_num_crops: test_num_crops_,
                                     test_patch_confidence_threshold: test_patch_confidence_threshold_,
                                     test_patch_strong_confidence_threshold: test_patch_strong_confidence_threshold_}
                        # print eval
                        print('(%s, %s, %s, %s) :  ' % (test_num_crops_, test_patch_nms_threshold_,
                                                        test_patch_confidence_threshold_, 
                                                        test_patch_strong_confidence_threshold_), end='')
                        val_map = run_eval(sess, results_path, feed_dict)
                        viz.save_tee(args.log_dir, tee)
                        # save best params
                        if val_map > best_val_map[test_num_crops_]:
                            best_val_map[test_num_crops_] = val_map
                            best_test_num_crops[test_num_crops_] = test_num_crops_
                            best_test_patch_nms_threshold[test_num_crops_] = test_patch_nms_threshold_
                            best_test_patch_confidence_threshold[test_num_crops_] = test_patch_confidence_threshold_
                            best_test_patch_strong_confidence_threshold[
                                test_num_crops_] = test_patch_strong_confidence_threshold_
                            
# Output best result for each
for test_num_crops_ in test_num_crops_sweep:                       
    # Print best parameters
    print('\nBest hyperparameters for %d crops: (val = %.4f)' % (test_num_crops_, best_val_map[test_num_crops_]))
    print('  num_crops=', best_test_num_crops[test_num_crops_])
    print('  nms_threshold=', best_test_patch_nms_threshold[test_num_crops_])
    print('  tau_low=', best_test_num_crops[test_num_crops_])
    print('  tau_high=', best_test_patch_strong_confidence_threshold[test_num_crops_])
    
    # Evaluate on the test set
    print('Test set accuracy')
    with tf.Graph().as_default() as graph:
        (use_test_split, test_num_crops, test_patch_confidence_threshold,
         test_patch_strong_confidence_threshold, run_eval) = build_graph(
            best_test_patch_confidence_threshold[test_num_crops_], max_test_num_crops=test_num_crops_sweep[-1])
        
        feed_dict = {use_test_split: True,
                     test_num_crops: best_test_num_crops[test_num_crops_],
                     test_patch_confidence_threshold: best_test_num_crops[test_num_crops_],
                     test_patch_strong_confidence_threshold: best_test_patch_strong_confidence_threshold[test_num_crops_]}
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir, config=config)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            run_eval(sess, results_path, feed_dict)
            
viz.save_tee(args.log_dir, tee)