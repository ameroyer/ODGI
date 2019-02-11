import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pickle
import time

import tensorflow as tf
print("Tensorflow version", tf.__version__)

from include import configuration
from include import graph_manager
from include import nets
from include import loss_utils
from include import eval_utils
from include import viz

tee = viz.Tee()    

########################################################################## Configuration
parser = argparse.ArgumentParser(description='Grouped Object Detection (ODGI).')
configuration.build_base_parser(parser)
parser.add_argument('--stage2_batch_size', type=int, help= 'Fixed batch size.')
parser.add_argument('--stage2_image_size', type=int, help= 'Image size for the second stage.')
parser.add_argument('--stage2_network', type=str, default="tiny_yolo_v2",
                    help='Architecture for the second stage.', choices=['tiny_yolo_v2', 'yolo_v2'])
parser.add_argument('--delayed_stage2_start', default=0, type=int, help= 'If given start training stage 2 after the given number of epochs.')
args = parser.parse_args()

print('ODGI - %s, Input size %d - %d\n' % (args.data, args.image_size, args.stage2_image_size)) 
configuration = defaults.build_base_config_from_args(args)
base_config = configuration.build_base_config_from_args(args, verbose=args.verbose)

stage1_config = base_config.copy()
stage2_config = base_config.copy()

# Inputs sizes
stage1_config['image_size'] = args.image_size
stage2_config['image_size'] = args.stage2_image_size
if stage2_config['image_size'] is None:
    stage2_config['image_size'] = stage1_config['image_size'] // 2

# Enable groups predictions for early stages
stage1_config['with_groups'] = True
stage1_config['with_offsets'] = True
configuration.finalize_grid_offsets(stage1_config)

# stage 2 architecture
stage2_config['network'] = args.stage2_network
stage2_config['batch_size'] = args.stage2_batch_size 
stage2_config['previous_batch_size'] = stage1_config['batch_size'] 
configuration.finalize_grid_offsets(stage2_config)

#Exp name
base_config['exp_name'] += '/%s_odgi_%d_%d' % (
    stage1_config['network'], stage1_config['image_size'], stage2_config['image_size'])

def log_run():        
    global tee, base_config
    viz.save_tee(base_config["log_dir"], tee)
    

########################################################################## Build the graph
### templates for stage 1
network = configuration.get_defaults(stage1_config, ['network'], verbose=True)[0]
forward_fn = tf.make_template(network, getattr(nets, network))
decode_fn = tf.make_template('decode', nets.get_detection_outputs_with_groups)
stage1_forward_pass = partial(nets.forward, forward_fn=forward_fn, decode_fn=decode_fn)

### templates for stage 2
stage1_network = configuration.get_defaults(stage2_config, ['network'], verbose=True)[0]
forward_fn = tf.make_template(network, getattr(nets, network))
decode_fn = tf.make_template('decode', nets.get_detection_outputs)
stage2_forward_pass = partial(nets.forward, forward_fn=forward_fn, decode_fn=decode_fn)


with tf.name_scope('train'):
    print('\nGraph:')     
    add_summaries = multistage_configuration['save_summaries_steps'] is not None
    for i in range(configuration['num_gpus']): 
        is_chief = (i == 0)
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('dev%d' % i):
                train_inputs, _ = graph_manager.get_inputs(mode='train', 
                                                           shard_index=i, 
                                                           verbose=args.verbose * int(is_chief),
                                                           **stage1_configuration)
                train_s1_outputs = train_pass(train_inputs,
                                              stage1_configuration, 
                                              is_chief=is_chief, 
                                              intermediate_stage=True,
                                              verbose=args.verbose)
                train_s2_inputs = feed_pass(train_inputs,
                                            train_s1_outputs['crop_boxes'], 
                                            stage2_configuration,
                                            mode='train',
                                            is_chief=is_chief, 
                                            verbose=args.verbose)
                train_s2_outputs = train_pass(train_s2_inputs, 
                                              stage2_configuration, 
                                              is_chief=is_chief,
                                              intermediate_stage=False,
                                              verbose=args.verbose) 
                if is_chief and add_summaries:
                    print(' > summaries:')
                    graph_manager.add_summaries(train_inputs, train_s1_outputs, mode='train', 
                                                family="train_stage1", **stage1_configuration)
                    graph_manager.add_summaries(train_s2_inputs, train_s2_outputs, mode='train', verbose=0,
                                                family="train_stage2", **stage2_configuration)

    # Training Objective
    print('\nLosses:')
    with tf.name_scope('losses'):
        losses = graph_manager.get_total_loss(
            splits=[stage1_configuration['base_name'], stage2_configuration['base_name']], 
            add_summaries=add_summaries, verbose=args.verbose)
        assert len(losses) == 2
        full_loss = [x[0] for x in losses]

    # Train op    
    with tf.name_scope('train_op'):   
        global_step, train_ops = graph_manager.get_train_op(losses, verbose=args.verbose, **multistage_configuration)
        assert len(train_ops) == 2
        train_stage1_op = train_ops[0]
        train_stage2_op = train_ops[1]

    # Config
    if add_summaries:
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
            eval_s1_outputs = eval_pass_intermediate_stage(eval_inputs, 
                                                           stage1_configuration, 
                                                           verbose=False) 
            eval_s2_inputs = feed_pass(eval_inputs, 
                                       eval_s1_outputs['crop_boxes'],
                                       stage2_configuration,
                                       mode='test', 
                                       verbose=False)
            eval_s2_outputs = eval_pass_final_stage(eval_s2_inputs, 
                                                    eval_s1_outputs['crop_boxes'], 
                                                    stage2_configuration, 
                                                    verbose=False)


########################################################################## Evaluation script
def run_eval(sess, mode, global_step_, results_path):
    assert mode in ['val', 'test']
    feed_dict = {use_test_split: mode == 'test'}
    with open(results_path, 'w') as f:
        f.write('%s results at step %d\n' % (mode, global_step_))
    sess.run(eval_initializer, feed_dict=feed_dict)
    try:
        while 1:
            out_ = sess.run([eval_inputs['im_id'], 
                             eval_inputs['num_boxes'],
                             eval_inputs['bounding_boxes'],                                             
                             eval_s2_outputs['bounding_boxes'],
                             eval_s2_outputs['detection_scores'],
                             eval_s1_outputs['bounding_boxes'],
                             eval_s1_outputs['detection_scores'],
                             eval_s1_outputs['kept_out_filter']], 
                            feed_dict=feed_dict)
            eval_utils.append_individuals_detection_output(results_path, *out_, **multistage_configuration)
    except tf.errors.OutOfRangeError:
        pass
    eval_aps, eval_aps_thresholds = eval_utils.detect_eval(results_path, **multistage_configuration)
    print('\r%s eval at step %d:' % (mode, global_step_), ' - '.join(
        'map@%.2f = %.5f' % (thresh, sum(x[t] for x in eval_aps.values()) / len(eval_aps))
            for t, thresh in enumerate(eval_aps_thresholds)))


########################################################################## Logs
print('\ntotal graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6)) 
graph_manager.generate_log_dir(multistage_configuration)
print('    Log directory', os.path.abspath(multistage_configuration["log_dir"]))
viz.save_tee(multistage_configuration["log_dir"], tee)
validation_results_path = os.path.join(multistage_configuration["log_dir"], 'val_output.txt')
test_results_path = os.path.join(multistage_configuration["log_dir"], 'test_output.txt')
with open(os.path.join(multistage_configuration["log_dir"], 'stage1_config.pkl'), 'wb') as f:
    pickle.dump(stage1_configuration, f)
with open(os.path.join(multistage_configuration["log_dir"], 'stage2_config.pkl'), 'wb') as f:
    pickle.dump(stage2_configuration, f)

########################################################################## Start Session
print('\nLaunch session:')
global_step_ = 0
try:        
    with graph_manager.get_monitored_training_session(**multistage_configuration) as sess:              
        print('\nStart training:')  
        start_time = time.time()
        try:
            while 1:
                # Train
                if global_step_ >= multistage_configuration['train_num_iters_per_epoch'] * args.delayed_stage2_start:
                    global_step_, full_loss_, _, _ = sess.run([global_step, full_loss, train_stage1_op, train_stage2_op])
                else:
                    global_step_, full_loss_, _ = sess.run([global_step, full_loss, train_stage1_op])

                if (global_step_ - 1) % args.display_loss_very_n_steps == 0:
                    viz.display_loss(None, global_step_, full_loss_, start_time,
                                     multistage_configuration["train_num_samples_per_iter"],
                                     multistage_configuration["train_num_samples"])

                # Evaluate on validation
                if (multistage_configuration["save_evaluation_steps"] is not None and (global_step_ > 1)
                    and global_step_  % multistage_configuration["save_evaluation_steps"] == 0):
                    run_eval(sess, 'val', global_step_, validation_results_path)
                    viz.save_tee(multistage_configuration["log_dir"], tee)

        except tf.errors.OutOfRangeError: # End of training
            pass              
        # Evaluate on the test set 
        run_eval(sess, 'test', global_step_, test_results_path)
        viz.save_tee(multistage_configuration["log_dir"], tee)

except KeyboardInterrupt:          # Keyboard interrupted
    print('\nInterrupted at step %d' % global_step_)