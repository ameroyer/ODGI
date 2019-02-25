import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import pickle
import time
from functools import partial

try:
    from scipy.misc import imread, imresize
    def load_and_resize(image_path, imsize):
        return imresize(imread(image_path, mode='RGB'), (imsize, imsize))
except ImportError:
    from PIL import Image  
    import numpy as np
    def load_and_resize(image_path, imsize):
        img = Image.open(image_path)
        img = img.resize((imsize, imsize), Image.ANTIALIAS)
        return np.array(img)

import tensorflow as tf

from include import configuration
from include import eval_utils
from include import graph_manager
from include import nets
from include import viz
from train_odgi import stage_transition, format_final_boxes


########################################################################## Base Config
parser = argparse.ArgumentParser(description='Grouped Object Detection (ODGI).')
parser.add_argument('log_dir', type=str, help='log directory to load from')
parser.add_argument('--num_iters', type=int, default=500, help='Number of iterations')
parser.add_argument('--image_size', type=int, help='Change input image size')
parser.add_argument('--stage2_image_size', type=int, help='Change stage 2 input image size')
parser.add_argument('--test_num_crops', type=int, help='Number of crops to extract for ODGI')
parser.add_argument('--device', type=str, default='cpu', help='Device', choices=['cpu', 'gpu'])
parser.add_argument('--verbose', type=int, default=2, help='Extra verbosity')
args = parser.parse_args()

tee = viz.Tee(filename='timing_experiments_%s_log.txt' % args.device)  


########################################################################## Load Configuration
odgi_mode = 'odgi' in os.path.split(os.path.dirname(os.path.normpath(args.log_dir)))[1]

if not odgi_mode:
    with open(os.path.join(args.log_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    network = configuration.get_defaults(config, ['network'], verbose=0)[0]
    forward_fn = tf.make_template(network, getattr(nets, network))
    decode_fn = tf.make_template('decode', nets.get_detection_outputs)
    forward_pass = partial(nets.forward, forward_fn=forward_fn, decode_fn=decode_fn)
    if args.image_size is not None:
        config['image_size'] = args.image_size
        configuration.finalize_grid_offsets(config)
    imsize = config['image_size']    
    config['batch_size'] = 1
else:
    stages = []
    for i, base_name in enumerate(['stage1', 'stage2']):
        with open(os.path.join(args.log_dir, '%s_config.pkl' % base_name), 'rb') as f:
            config = pickle.load(f)
        if args.test_num_crops is not None:
            config['test_num_crops'] = args.test_num_crops
        network_name = configuration.get_defaults(config, ['network'], verbose=0)[0]
        ## Templates
        forward_fn = tf.make_template('%s/%s' % (base_name, network_name), getattr(nets, network_name)) 
        if i == 0:
            decode_fn = tf.make_template('%s/decode' % base_name, nets.get_detection_outputs_with_groups)
        else:
            decode_fn = tf.make_template('%s/decode' % base_name, nets.get_detection_outputs)
        forward_pass = partial(nets.forward, forward_fn=forward_fn, decode_fn=decode_fn)
        stages.append((base_name, network_name, forward_pass, config))
    if args.image_size is not None:
        stages[0][-1]['image_size'] = args.image_size
        configuration.finalize_grid_offsets(stages[0][-1])
    if args.stage2_image_size is not None:
        stages[1][-1]['image_size'] = args.stage2_image_size
        configuration.finalize_grid_offsets(stages[1][-1])
    imsize = stages[0][-1]['image_size']
    stages[0][-1]['batch_size'] = 1
    stages[1][-1]['previous_batch_size'] = 1
    
    
########################################################################## Build the graph
with tf.Graph().as_default():
    with tf.device('/%s:0' % args.device):
        image = tf.placeholder(tf.uint8, [imsize, imsize, 3])
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        processed_image = tf.expand_dims(processed_image, axis=0)
        eval_inputs = {'image': processed_image}
        
        #### STANDARD
        if not odgi_mode:
            outputs = forward_pass(eval_inputs['image'], config, is_training=False, verbose=0)
            outputs = [outputs['bounding_boxes'], outputs['detection_scores']]
        #### ODGI
        else:
            stage_inputs = eval_inputs
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
            # final outputs        
            outputs = [stage2_pred_bbs, stage2_pred_confidences, stage1_pred_bbs, 
                       stage1_pred_confidences, stage1_kept_out_boxes]
            
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        
    print('number of parameters', total_parameters) 
    print('total graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 
    ########################################################################## Start Session
    print('Launch session from %s' % args.log_dir)

    if args.device == 'gpu': 
        config = tf.ConfigProto(allow_soft_placement=True)
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir, config=config)
    else:
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir)

    loading_time = 0.
    run_time = 0.
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        try:
            image_path = './timing_image.png'
            for i in range(args.num_iters):
                # load
                start_time = time.time()
                read_img = load_and_resize(image_path, imsize)
                feed_dict = {image: read_img}    
                # run
                load_time = time.time()  
                out_ = sess.run(outputs, feed_dict=feed_dict)
                end_time = time.time()                
                loading_time += load_time - start_time
                run_time += end_time - load_time
                if args.verbose == 2:
                    print('\r Step %d/%d' % (i + 1, args.num_iters), end='') 
        except tf.errors.OutOfRangeError:
            pass
        if args.verbose == 2:
            print('\rEvaluated %d samples' % args.num_iters)
        else:
            print('Evaluated %d samples' % args.num_iters)
        # Timing 
        loading_time /= args.num_iters
        run_time /= args.num_iters
        print('Timings:')
        print('   Avg. Loading Time:', loading_time)
        print('   Avg. Feed forward:', run_time)
        print('   Avg. time:', run_time + loading_time)
        viz.save_tee(args.log_dir, tee)