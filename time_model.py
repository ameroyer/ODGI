import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import time
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
print("Tensorflow version", tf.__version__)

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
parser.add_argument('--device', type=str, default='cpu', help='GPU or CPU')
parser.add_argument('--mobilenet', type=float, default=1.0, help='GPU or CPU')
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
if aux[2] == '5boxes':
    assert mode == 'standard'
    configuration['num_boxes'] = 5
    imsize = int(aux[3])
else:
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
        
        stage1_configuration['depth_multiplier'] = args.mobilenet
        stage2_configuration['depth_multiplier'] = 0.5

        # Graph
        image = tf.placeholder(tf.uint8, [imsize, imsize, 3]) 
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        processed_image = tf.expand_dims(processed_image, axis=0)
        eval_inputs = {'image': processed_image}
        with tf.device('/%s:0' % args.device):
            eval_s1_outputs = odgi_graph.eval_pass_intermediate_stage(
                eval_inputs, stage1_configuration, reuse=False, verbose=False) 
            eval_s2_inputs = odgi_graph.feed_pass(
                eval_inputs, eval_s1_outputs['crop_boxes'], stage2_configuration, mode='test', verbose=False)
            eval_s2_outputs = odgi_graph.eval_pass_final_stage(
                eval_s2_inputs, eval_s1_outputs['crop_boxes'], stage2_configuration, reuse=False, verbose=False)                    
            outputs = [eval_s2_outputs['bounding_boxes'],
                       eval_s2_outputs['detection_scores'],
                       eval_s1_outputs['bounding_boxes'],
                       eval_s1_outputs['detection_scores'],
                       eval_s1_outputs['kept_out_filter']]
                    
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
            eval_outputs = standard_graph.eval_pass(eval_inputs, standard_configuration, reuse=False, verbose=False)
            outputs = [eval_outputs['bounding_boxes'],
                       eval_outputs['detection_scores']]

    else:
        raise ValueError('Unkown mode', mode)
                               

    ########################################################################## Start Session
    print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 
    print('\nLaunch session from %s:' % args.log_dir)

    if args.device == 'gpu':
        gpu_mem_frac = graph_manager.get_defaults(configuration, ['gpu_mem_frac'], verbose=args.verbose)[0]    
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac), allow_soft_placement=True)
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir, config=config)
    else:
        session_creator = tf.train.ChiefSessionCreator(checkpoint_dir=args.log_dir)

    num_samples = 0
    loading_time = 0.
    run_time = 0.
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        images = [os.path.join(configuration['image_folder'], x) for x in os.listdir(configuration['image_folder'])
                  if x.endswith(configuration['image_suffix'])]
        #### TODO
        images = images[:200]
        try:
            for image_path in images:
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
                num_samples += 1
                if args.verbose == 2:
                    print('\r Step %d' % num_samples, end='') 
        except tf.errors.OutOfRangeError:
            pass
        print()
        print('Evaluated %d samples' % num_samples)
        # Timing 
        loading_time /= num_samples
        run_time /= num_samples
        print('Timings:')
        print('   Avg. Loading Time:', loading_time)
        print('   Avg. Feed forward:', run_time)
        print('   Avg. time:', run_time + loading_time)
        viz.save_tee(args.log_dir, tee)