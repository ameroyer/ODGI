import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")

import argparse
import time
try:
    from scipy.misc import imread, imresize
    def load_and_resize(image_path, imsize):
        return imresize(imread(image_path, mode='RGB'), (imsize, imsize)).astype(np.uint8)
except ImportError:
    from PIL import Image  
    import numpy as np
    def load_and_resize(image_path, imsize):
        img = Image.open(image_path)
        img = img.resize((imsize, imsize), Image.ANTIALIAS)
        return np.array(img).astype(np.uint8)

import tensorflow as tf
#print("Tensorflow version", tf.__version__)

import defaults
import eval_utils
import graph_manager
import viz
import odgi_graph
import standard_graph

tee = viz.Tee(filename='time_log.txt')  
########################################################################## Base Config
parser = argparse.ArgumentParser(description='Grouped Object Detection (ODGI).')
parser.add_argument('log_dir', type=str, help='log directory to load from')
parser.add_argument('--test_patch_nms_threshold', default=0.25, type=float, help='NMS threshold')
parser.add_argument('--test_patch_confidence_threshold', default=0.1, type=float, help='Low confidence threshold')
parser.add_argument('--test_patch_strong_confidence_threshold', default=0.9, type=float, help='High confidence threshold')
parser.add_argument('--test_num_crops', default=1, type=int, help='Number of crops')
parser.add_argument('--num_runs', default=500, type=int, help='Number of timing runs')
parser.add_argument('--device', type=str, default='cpu', help='GPU or CPU')
parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
parser.add_argument('--verbose', type=int, default=2, help='Extra verbosity')
args = parser.parse_args()
assert args.device in ['cpu', 'gpu']

########################################################################## Infer configuration from log dir name
# Data 
aux = os.path.dirname(os.path.dirname(os.path.normpath(args.log_dir)))
data = os.path.split(aux)[1]
configuration = {}
if data.startswith('vedai'):
    configuration['setting'] = data
    configuration['image_suffix'] = '_co.png'
elif data == 'sdd':
    configuration['setting'] = 'sdd'
    configuration['image_suffix'] = '.jpeg'
elif data == 'dota':
    configuration['setting'] = 'dota'
    configuration['image_suffix'] = '.jpg'
else:
    raise ValueError("unknown data", data)
    
configuration['test_patch_nms_threshold'] = args.test_patch_nms_threshold
configuration['test_patch_confidence_threshold'] = args.test_patch_confidence_threshold
configuration['test_patch_strong_confidence_threshold'] = args.test_patch_strong_confidence_threshold
configuration['test_num_crops'] = args.test_num_crops
    
# Metadata
tfrecords_path = 'Data/metadata_%s.txt'
metadata = defaults.load_metadata(tfrecords_path % configuration['setting'])
configuration.update(metadata)

# Architecture
aux = os.path.split(os.path.dirname(os.path.normpath(args.log_dir)))[1]
aux = aux.split('_')
configuration['network'] = aux[0]
configuration['gpu_mem_frac'] = args.gpu_mem_frac
configuration['same_network'] = False
mode = aux[1]
imsize = int(aux[2])

with tf.Graph().as_default() as graph:
    ########################### ODGI
    if mode == 'odgi':
        assert len(aux) in [4, 5]
        stage1_configuration = configuration.copy()
        stage2_configuration = configuration.copy()

        # Read images one by one for timing purposes
        stage1_configuration['batch_size'] = 1
        stage2_configuration['batch_size'] = None 

        # Image sizes
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
        image = tf.placeholder(tf.uint8, [imsize, imsize, 3]) 
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        processed_image = tf.expand_dims(processed_image, axis=0)
        eval_inputs = {'image': processed_image}
        with tf.device('/%s:0' % args.device):
            crop_boxes, kept_out_boxes, kept_out_scores = odgi_graph.eval_pass_intermediate_stage(
                eval_inputs, stage1_configuration, reuse=False, verbose=False) 
            outputs = odgi_graph.feed_pass(
                eval_inputs, crop_boxes, stage2_configuration, mode='test', verbose=False)
            outputs = odgi_graph.eval_pass_final_stage(
                outputs, crop_boxes, stage2_configuration, reuse=False, verbose=False)    
            
                    
    ########################### Standard
    elif mode == 'standard':
        assert len(aux) == 3
        standard_configuration = configuration.copy()
        standard_configuration['batch_size'] = 1
        standard_configuration['base_name'] = configuration['network']
        standard_configuration['image_size'] = int(aux[2])
        graph_manager.finalize_grid_offsets(standard_configuration)
        imsize = standard_configuration['image_size']
                     
        image = tf.placeholder(tf.uint8, [imsize, imsize, 3])   
        processed_image = tf.image.convert_image_dtype(image, tf.float32)       
        processed_image = tf.expand_dims(processed_image, axis=0)  
        eval_inputs = {'image': processed_image}
        with tf.device('/%s:0' % args.device):
            outputs = standard_graph.eval_pass(eval_inputs, standard_configuration, reuse=False, verbose=False)

    else:
        raise ValueError('Unkown mode', mode)
                               

    ########################################################################## Start Session
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 
    print('test_num_crops', args.test_num_crops)
    print('Outputs', '\n'.join(list(map(str, outputs))))
    print('Launch session from %s:' % args.log_dir)

    if args.device == 'gpu':
        gpu_mem_frac = graph_manager.get_defaults(configuration, ['gpu_mem_frac'], verbose=args.verbose)[0]    
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac), allow_soft_placement=True)
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir, config=config)
    else:
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir)

    loading_time = 0.
    run_time = 0.
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        try:
            image_path = './timing_image.png'
            for i in range(args.num_runs):
                # load
                start_time = time.time()
                read_img = load_and_resize(image_path, imsize)
                feed_dict = {image: read_img}    
                # run
                load_time = time.time()  
                if mode == 'standard':
                    out_ = sess.run(outputs, feed_dict=feed_dict)
                else:
                    out_ = sess.run([outputs, kept_out_boxes, kept_out_scores], feed_dict=feed_dict)
                end_time = time.time()                
                loading_time += load_time - start_time
                run_time += end_time - load_time
                if args.verbose == 2:
                    print('\r Step %d/%d' % (i + 1, args.num_runs), end='') 
        except tf.errors.OutOfRangeError:
            pass
        print()
        print('Evaluated %d samples' % args.num_runs)
        # Timing 
        loading_time /= args.num_runs
        run_time /= args.num_runs
        print('Timings:')
        print('   Avg. Loading Time:', loading_time)
        print('   Avg. Feed forward:', run_time)
        print('   Avg. time:', run_time + loading_time)
        viz.save_tee(args.log_dir, tee)