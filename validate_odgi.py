import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import pickle
import time
from functools import partial

import tensorflow as tf

from include import configuration
from include import eval_utils
from include import graph_manager
from include import nets
from include import viz
from train_odgi import stage_transition, format_final_boxes

import logging
logging.getLogger('tensorflow').setLevel(logging.INFO)

"""Perform hyperparameter sweep for the patch extraction stage for a pre-trained 
two stage ODGI model"""


tee = viz.Tee(filename='hyperparameters_sweep_log.txt')  

########################################################################## Base Config
parser = argparse.ArgumentParser(description='Hyperparameter sweep on the validation set.')
parser.add_argument('log_dir', type=str, help='log directory to load from')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
args = parser.parse_args()


########################################################################## Hyperparameters range
test_num_crops_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                
test_patch_nms_threshold_sweep = [0.25, 0.5, 0.75]                      
test_patch_confidence_threshold_sweep  = [0., 0.1, 0.2, 0.3, 0.4]
test_patch_strong_confidence_threshold_sweep = [0.6, 0.7, 0.8, 0.9, 1.0]

test_num_crops_sweep = [6, 1]                                
test_patch_nms_threshold_sweep = [0.25]                      
test_patch_confidence_threshold_sweep  = [0.1]
test_patch_strong_confidence_threshold_sweep = [0.6]

# Save best hyper-parameters for each number of crops
best_val_map = {k: 0. for k in test_num_crops_sweep}
best_test_patch_nms_threshold = {k: None for k in test_num_crops_sweep}
best_test_patch_confidence_threshold = {k: None for k in test_num_crops_sweep}
best_test_patch_strong_confidence_threshold = {k: None for k in test_num_crops_sweep}


########################################################################## Configuration(s)
stages = []
for i, base_name in enumerate(['stage1', 'stage2']):
    with open(os.path.join(args.log_dir, '%s_config.pkl' % base_name), 'rb') as f:
        config = pickle.load(f)
    network_name = configuration.get_defaults(config, ['network'], verbose=True)[0]
    ## Templates
    forward_fn = tf.make_template('%s/%s' % (base_name, network_name), getattr(nets, network_name)) 
    if i == 0:
        decode_fn = tf.make_template('%s/decode' % base_name, nets.get_detection_outputs_with_groups)
    else:
        decode_fn = tf.make_template('%s/decode' % base_name, nets.get_detection_outputs)
    forward_pass = partial(nets.forward, forward_fn=forward_fn, decode_fn=decode_fn)
    stages.append((base_name, network_name, forward_pass, config))
    
# Override
stages[0][-1]['num_gpus'] = 1
stages[0][-1]['gpu_mem_frac'] = args.gpu_mem_frac
stages[0][-1]['batch_size'] = args.batch_size
stages[1][-1]['previous_batch_size'] = args.batch_size 


with tf.Graph().as_default() as graph:
    ########################################################################## Build the graph for hyperparameter test
    # Parameters to sweep on as placeholder
    test_patch_nms_threshold_placeholder =  tf.placeholder(tf.float32, shape=())
    test_num_crops_placeholder = tf.placeholder(tf.int32, shape=())                                    
    test_patch_confidence_threshold_placeholder = tf.placeholder(tf.float32, shape=()) 
    test_patch_strong_confidence_threshold_placeholder = tf.placeholder(tf.float32, shape=())

    # Set-up config
    stages[0][-1]["test_num_crops"] = test_num_crops_sweep[-1]     # always extract test_num_crops... 
    stages[1][-1]["test_num_crops_slice"] = test_num_crops_placeholder   # ...but only use `test_num_crops_slice` in practice  
    stages[0][-1]["test_patch_nms_threshold"] = test_patch_nms_threshold_placeholder                
    stages[0][-1]["test_patch_confidence_threshold"] = test_patch_confidence_threshold_placeholder
    stages[0][-1]["test_patch_strong_confidence_threshold"] = test_patch_strong_confidence_threshold_placeholder

    # Inputs
    eval_split_placehoder = tf.placeholder_with_default(True, (), 'choose_eval_split')                  
    eval_inputs, eval_initializer = tf.cond(
        eval_split_placehoder,
        true_fn=lambda: graph_manager.get_inputs(mode='test', verbose=0, **stages[0][3]),
        false_fn=lambda: graph_manager.get_inputs(mode='val', verbose=0, **stages[0][3]),
        name='eval_inputs')

    # Graph  
    with tf.device('/gpu:0'):
        with tf.name_scope('dev0'):
            stage_inputs = eval_inputs[0]
            image_ids = stage_inputs['im_id']
            num_boxes = stage_inputs['num_boxes']
            gt_bbs = stage_inputs['bounding_boxes']

            for s, (name, _, forward_pass, stage_config) in enumerate(stages):                    
                if s > 0:
                    stage_inputs = stage_transition(
                        stage_inputs, stage_outputs, 'test', stage_config, verbose=0)

                # stage 1
                if s == 1:
                    stage1_pred_bbs = stage_outputs['bounding_boxes']
                    stage1_pred_confidences = stage_outputs['detection_scores']
                    stage1_kept_out_boxes = stage_outputs['kept_out_filter']
                    crop_boxes = stage_outputs['crop_boxes']

                stage_outputs = forward_pass(
                    stage_inputs['image'], stage_config, is_training=False, verbose=0)

                # stage 2 (final)
                if s == 1:                    
                    stage_outputs = format_final_boxes(stage_outputs, crop_boxes)
                    stage2_pred_bbs = stage_outputs['bounding_boxes']
                    stage2_pred_confidences = stage_outputs['detection_scores']

    # gather predictions across gpus
    with tf.name_scope('outputs'):
        eval_outputs = [image_ids, num_boxes, gt_bbs, stage2_pred_bbs, stage2_pred_confidences,
                        stage1_pred_bbs, stage1_pred_confidences, stage1_kept_out_boxes]   

    # eval functions
    validation_results_path = os.path.join(args.log_dir, 'val_output.txt')
    test_results_path = os.path.join(args.log_dir, 'test_output.txt')

    run_eval = partial(graph_manager.run_eval,
                       eval_split_placehoder=eval_split_placehoder,
                       eval_initializer=eval_initializer,
                       eval_outputs=eval_outputs, 
                       configuration=stages[0][-1])
    eval_validation = partial(run_eval, mode='val', results_path=validation_results_path)
    eval_test = partial(run_eval, mode='test', results_path=test_results_path)

    
    ########################################################################## Start session
    gpu_mem_frac = configuration.get_defaults(config, ['gpu_mem_frac'], verbose=0)[0] 
    if gpu_mem_frac < 1.0:
        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac), allow_soft_placement=True)
    else:
        session_config = tf.ConfigProto(allow_soft_placement=True)
        
    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(
        checkpoint_dir=args.log_dir, config=session_config)) as sess:                
        ########################################################################## Parameters sweep
        for test_num_crops in test_num_crops_sweep:           
            for test_patch_nms_threshold in test_patch_nms_threshold_sweep:                                     
                for test_patch_confidence_threshold in test_patch_confidence_threshold_sweep:
                    for test_patch_strong_confidence_threshold in test_patch_strong_confidence_threshold_sweep:     
                        # feed_dict
                        additional_feed_dict = {
                            test_num_crops_placeholder: test_num_crops,
                            test_patch_nms_threshold_placeholder: test_patch_nms_threshold,
                            test_patch_confidence_threshold_placeholder: test_patch_confidence_threshold,
                            test_patch_strong_confidence_threshold_placeholder: test_patch_strong_confidence_threshold}

                        # Evaluate 
                        mean_aps, eval_aps_thresholds, num_images = eval_validation(
                            sess, 1, additional_feed_dict=additional_feed_dict, verbose=False)
                        print('   (%s, %s, %s, %s) (%d images): %s' % (
                            test_num_crops, test_patch_nms_threshold, test_patch_confidence_threshold, 
                            test_patch_strong_confidence_threshold, num_images, ' - '.join(
                                'map@%.2f = %.5f' % (thresh, mean_aps[t])
                                for t, thresh in enumerate(eval_aps_thresholds))))
                        viz.save_tee(args.log_dir, tee)

                        # Store best parameters based on map@0.5
                        val_map = mean_aps[0]
                        if val_map > best_val_map[test_num_crops]:
                            best_val_map[test_num_crops] = val_map
                            best_test_patch_nms_threshold[test_num_crops] = test_patch_nms_threshold
                            best_test_patch_confidence_threshold[test_num_crops] = test_patch_confidence_threshold
                            best_test_patch_strong_confidence_threshold[
                                test_num_crops] = test_patch_strong_confidence_threshold

                            
        ########################################################################## Output test resutls for best config
        for test_num_crops in test_num_crops_sweep: 
            
            test_patch_nms_threshold = best_test_patch_nms_threshold[test_num_crops]
            test_patch_confidence_threshold = best_test_patch_confidence_threshold[test_num_crops]
            test_patch_strong_confidence = best_test_patch_strong_confidence_threshold[test_num_crops]

            # Print best parameters
            print('\nBest hyperparameters for %d crops: (val = %.4f)' % (test_num_crops, best_val_map[test_num_crops]))
            print('  num_crops =', test_num_crops)
            print('  nms_threshold =', test_patch_nms_threshold)
            print('  tau_low =', test_patch_confidence_threshold)
            print('  tau_high =', test_patch_strong_confidence_threshold)

            # Evaluate on the test set
            additional_feed_dict = {
                test_num_crops_placeholder: test_num_crops,
                test_patch_nms_threshold_placeholder: test_patch_nms_threshold,
                test_patch_confidence_threshold_placeholder: test_patch_confidence_threshold,
                test_patch_strong_confidence_threshold_placeholder: test_patch_strong_confidence_threshold}

            mean_aps, eval_aps_thresholds, num_images = eval_test(
                sess, 1, additional_feed_dict=additional_feed_dict, verbose=False)
            print('  evaluated %d images:' % num_images, ' - '.join(
                'map@%.2f = %.5f' % (thresh, mean_aps[t]) 
                for t, thresh in enumerate(eval_aps_thresholds)))

        viz.save_tee(args.log_dir, tee)