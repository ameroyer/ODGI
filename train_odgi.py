import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pickle
import time
from functools import partial

import tensorflow as tf
print("Tensorflow version", tf.__version__)

from include import configuration
from include import graph_manager
from include import nets
from include import loss_utils
from include import eval_utils
from include import tf_inputs
from include import viz


########################################################################## Convenience functions     
def stage_transition(stage_inputs, stage_outputs, mode, config, add_summaries=False, verbose=False): 
    """Create inputs for the next stage based on the output of the current stage"""
    assert mode in ['train', 'test']
    with tf.name_scope('extract_patches'):
        stage_outputs['crop_boxes'], _, stage_outputs['kept_out_filter'] = tf_inputs.extract_groups(
            stage_outputs['bounding_boxes'], 
            stage_outputs['confidence_scores'],
            predicted_group_flags=stage_outputs['group_classification_logits'],
            predicted_offsets=stage_outputs['offsets'] if 'offsets' in stage_outputs else None,
            mode=mode, verbose=verbose, **config)  
        
    if mode == 'train':
        del stage_outputs['kept_out_filter']
        if add_summaries:
            with tf.name_scope('4_crops'): 
                tf.summary.image('image', viz.draw_bounding_boxes(stage_inputs['image'], stage_outputs['crop_boxes']),
                                 max_outputs=3, collections=['outputs'], family=None)
        
    return graph_manager.get_stage2_inputs(stage_inputs, 
                                           tf.stop_gradient(stage_outputs['crop_boxes']), 
                                           mode=mode, 
                                           verbose=verbose, 
                                           **config)


def format_final_boxes(final_stage_outputs, crop_boxes):
    """Rescale outputs relatively to the original input image for evaluating the final 
       detection results
    
    Args:
        final_stage_outputs: Output dictionnary of the last stage (stage 2)
        crop_boxes: Crops extracted from stage 1
    """
    num_crops = tf.shape(crop_boxes)[1]
    num_boxes = final_stage_outputs['bounding_boxes'].get_shape()[3].value
    num_cells = final_stage_outputs['bounding_boxes'].get_shape()[1].value
    
    # reshape
    with tf.name_scope('reshape_outputs'):
        # outputs: (stage1_batch * num_crops, num_cell, num_cell, num_boxes, ...)
        # to: (stage1_batch, num_cell, num_cell, num_boxes * num_crops, ...)
        for key, value in final_stage_outputs.items():   
            # reshape to (batch_size, num_crops, ...)
            original_shape = tf.shape(value)       
            num_dims = len(value.get_shape().as_list())
            new_shape = tf.concat([tf.stack([-1, num_crops]), original_shape[1:]], axis=0)
            batches = tf.reshape(value, new_shape)
            # transpose to (batch_size, num_cells, num_cells, num_crops, num_boxes, ...)
            transpose_axis = [0, 2, 3, 1] + list(range(4, num_dims + 1))
            batches = tf.transpose(batches, transpose_axis)
            # reshape to (batch_size, num_cells, num_cells, num_crops * num_boxes, ...)
            new_shape = tf.concat([tf.stack([-1, num_cells, num_cells, num_crops * num_boxes]), new_shape[5:]], axis=0)
            final_stage_outputs[key] = tf.reshape(batches, new_shape)  
    
    # rescale
    with tf.name_scope('rescale_bounding_boxes'):
        # tile crop_boxes to (stage1_batch, 1, 1, num_crops * num_boxes, 4)
        crop_boxes = tf.expand_dims(crop_boxes, axis=-2)
        crop_boxes = tf.tile(crop_boxes, (1, 1, num_boxes, 1))
        crop_boxes = tf.reshape(crop_boxes, (-1, 1, 1, num_crops * num_boxes, 4))
        crop_boxes = tf.split(crop_boxes, 2, axis=-1)
        # bounding_boxes: (stage1_batch, num_cells, num_cells, num_crops * num_boxes, 4)
        final_stage_outputs['bounding_boxes'] *= tf.maximum(1e-8, tf.tile(crop_boxes[1] - crop_boxes[0], (1, 1, 1, 1, 2)))
        final_stage_outputs['bounding_boxes'] += tf.tile(crop_boxes[0], (1, 1, 1, 1, 2))
        final_stage_outputs['bounding_boxes'] = tf.clip_by_value(final_stage_outputs['bounding_boxes'], 0., 1.)        
    return final_stage_outputs




if __name__ == '__main__':
    ########################################################################## Configuration
    parser = argparse.ArgumentParser(description='Grouped Object Detection (ODGI).')
    configuration.build_base_parser(parser)
    parser.add_argument('--stage2_batch_size', type=int, help=('If given, use fixed batch size.'
        'Otherwise, use the stage 1 batch_size * num_crops'))
    parser.add_argument('--stage2_image_size', type=int, help='Image size for the second stage.')
    parser.add_argument('--stage2_network', type=str, default="tiny_yolo_v2",
                        help='Architecture for the second stage.', choices=[
                            'tiny_yolo_v2', 'yolo_v2', 'mobilenet_100', 'mobilenet_50', 'mobilenet_35'])
    parser.add_argument('--stage2_starting_epoch', default=0, type=int,
                        help='Start training stage 2 after the given number of epochs.')
    parser.add_argument('--share_variables', action='store_true', help='Share variables across stages.')
    args = parser.parse_args()
    if args.stage2_image_size is None:
        args.stage2_image_size = args.image_size // 2

    print('ODGI %s - %s, Input size %d - %d\n' % (args.network, args.stage2_network, 
                                                  args.image_size, args.stage2_image_size)) 
    base_config = configuration.build_base_config_from_args(args, verbose=args.verbose)
    base_config['exp_name'] += '/%s_odgi_%d_%d' % (
        args.network, args.image_size, args.stage2_image_size)

    with_summaries = base_config['save_summaries_steps'] is not None  
    graph_manager.generate_log_dir(base_config)
    print('    Log directory', os.path.abspath(base_config["log_dir"]))  


    tee = viz.Tee()  
    def log_run():        
        global tee, base_config
        viz.save_tee(base_config["log_dir"], tee)
        
    ########################################################################## Stage configuration
    stage1_config = base_config.copy()
    stage2_config = base_config.copy()

    # Inputs sizes
    stage1_config['image_size'] = args.image_size
    stage2_config['image_size'] = args.stage2_image_size

    # Enable groups predictions for early stages
    stage1_config['with_groups'] = True
    stage1_config['with_offsets'] = True
    configuration.finalize_grid_offsets(stage1_config)

    # stage 2 architecture
    stage2_config['network'] = args.stage2_network
    stage2_config['previous_batch_size'] = stage1_config['batch_size'] 
    stage2_config['batch_size'] = args.stage2_batch_size
    configuration.finalize_grid_offsets(stage2_config)


    ### templates for each stage
    stages = []
    stages_configs = [stage1_config, stage2_config]
    shared_scope = None
    
    if args.share_variables:
        network_name = configuration.get_defaults(stage1_config, ['network'], verbose=True)[0]
        for config in stages_configs:
            assert network_name == configuration.get_defaults(config, ['network'], verbose=True)[0]
        forward_fn = tf.make_template(network_name, getattr(nets, network_name)) 
        shared_scope = network_name
        
        
    for i, config in enumerate(stages_configs):
        base_name = 'stage%d' % (i + 1)
        network_name = configuration.get_defaults(config, ['network'], verbose=True)[0]
        if not args.share_variables:
            forward_fn = tf.make_template('%s/%s' % (base_name, network_name), getattr(nets, network_name)) 

        # intermediate stages
        if i < len(stages_configs) - 1:
            decode_fn = tf.make_template('%s/decode' % base_name, nets.get_detection_outputs_with_groups)
            loss_fn = partial(loss_utils.get_odgi_loss, loss_base_name=base_name)
        # final stage
        else:
            decode_fn = tf.make_template('%s/decode' % base_name, nets.get_detection_outputs)
            loss_fn = partial(loss_utils.get_standard_loss, loss_base_name=base_name)

        forward_pass = partial(nets.forward, forward_fn=forward_fn, decode_fn=decode_fn)
        stages.append((base_name, network_name, forward_pass, config, loss_fn))

        with open(os.path.join(base_config["log_dir"], '%s_config.pkl' % base_name), 'wb') as f:
            pickle.dump(config, f)
        if with_summaries:
            with tf.name_scope('%s_config_summary' % base_name):
                viz.add_text_summaries(config) 
        

    ########################################################################## Build the graph
    print('\nTrain Graph:')
    with tf.name_scope('train'):
        with tf.name_scope('inputs'):
            inputs, _ = graph_manager.get_inputs(mode='train', verbose=args.verbose, **stages[0][3])    

        for i in range(base_config['num_gpus']): 
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('dev%d' % i):
                    verbose = args.verbose * (i == 0)
                    stage_inputs = inputs[i]

                    ### Main graph #######
                    for s, (name, network_name, forward_pass, stage_config, loss_fn) in enumerate(stages):
                        ### Transition from next stage
                        if s > 0:
                            with tf.name_scope('stage_transition'):
                                print((' > %s' if verbose == 1 else ' \033[33m> %s\033[0m') % 'Stage transition')
                                stage_inputs = stage_transition(
                                    stage_inputs, stage_outputs, 'train', stage_config, 
                                    add_summaries=with_summaries * (i == 0), verbose=verbose)

                        ### Feed forward
                        with tf.name_scope(name):
                            if verbose > 0:
                                print((' > %s/%s' if verbose == 1 else ' \033[33m> %s/%s\033[0m') % (
                                    name, network_name))

                            with tf.name_scope('feed_forward'):
                                stage_outputs = forward_pass(
                                    stage_inputs['image'], stage_config, is_training=True, verbose=verbose) 

                            if verbose > 0:
                                print((' > %s' if verbose == 1 else ' \033[33m> %s\033[0m') % 'Collecting losses')
                            with tf.name_scope('losses'):
                                graph_manager.add_losses_to_graph(
                                    loss_fn, stage_inputs, stage_outputs, stage_config, is_chief=i == 0, verbose=verbose)

                        ### Summaries
                        if (i == 0) and with_summaries:
                            print((' > %s' if verbose == 1 else ' \033[33m> %s\033[0m') % 'Adding summaries')
                            graph_manager.add_summaries(
                                stage_inputs, stage_outputs, mode='train', family="train_%s" % name, **stage_config)
                    #######################

        # Training Objective
        print('\nLosses:')
        with tf.name_scope('losses'):
            if False:
                stage1_losses = graph_manager.get_total_loss(
                    splits=['stage1'], with_summaries=with_summaries, verbose=args.verbose)
                assert len(stage1_losses) == 1
                stage1_loss = [x[0] for x in stage1_losses]
                losses = graph_manager.get_total_loss(with_summaries=with_summaries, verbose=args.verbose)
                assert len(losses) == 1
                full_loss = [x[0] for x in losses]
            else:
                losses = graph_manager.get_total_loss(splits=[x[0] for x in stages],
                                                      shared_scope=shared_scope,
                                                      with_summaries=with_summaries,
                                                      verbose=args.verbose)
                assert len(losses) == 2
                full_loss = [x[0] for x in losses]


        # Train op    
        with tf.name_scope('train_op'):   
            if False:
                global_step, train_ops = graph_manager.get_train_op(stage1_losses, verbose=args.verbose, **base_config)
                train_stage1_op = train_ops[0]
                _, train_ops = graph_manager.get_train_op(losses, verbose=args.verbose, **base_config)
                train_op = train_ops[0]
            else:
                global_step, train_ops = graph_manager.get_train_op(losses, verbose=args.verbose, **base_config)
                assert len(train_ops) == len(losses)
                train_stage1_op = train_ops[0]
                train_stage2_op = train_ops[1]



    ############################### Eval
    with tf.name_scope('eval'):  
        eval_split_placehoder = tf.placeholder_with_default(True, (), 'choose_eval_split')
        ### TODO (multi-GPU evaluation)
        # tf.reshape operations in `stage_transition` do not handle case of 0-dims Tensors 
        # that may happen when splitting the inputs tensors across devices.
        # To avoid these cases, we run evaluation on one device
        # Note: Thsi is not a problem during training as drop_remainder is turned on
        base_config['num_gpus'] = 1
        stage1_config['num_gpus'] = 1
        ### TODO

        eval_inputs, eval_initializer = tf.cond(
            eval_split_placehoder,
            true_fn=lambda: graph_manager.get_inputs(mode='test', verbose=False, **stages[0][3]),
            false_fn=lambda: graph_manager.get_inputs(mode='val', verbose=False, **stages[0][3]),
            name='eval_inputs')

        for i in range(base_config['num_gpus']):     
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('dev%d' % i):
                    stage_inputs = eval_inputs[i]
                    tf.add_to_collection('inference_image_ids', eval_inputs[i]['im_id'])
                    tf.add_to_collection('inference_num_boxes', eval_inputs[i]['num_boxes'])
                    tf.add_to_collection('inference_gt_bbs', eval_inputs[i]['bounding_boxes'])

                    for s, (name, _, forward_pass, stage_config, _) in enumerate(stages):                    
                        if s > 0:
                            stage_inputs = stage_transition(
                                stage_inputs, stage_outputs, 'test', stage_config, verbose=verbose)

                        # stage 1
                        if s == 1:
                            tf.add_to_collection('stage1_pred_bbs', stage_outputs['bounding_boxes'])
                            tf.add_to_collection('stage1_pred_confidences', stage_outputs['detection_scores'])
                            tf.add_to_collection('stage1_kept_out_boxes', stage_outputs['kept_out_filter'])
                            crop_boxes = stage_outputs['crop_boxes']

                        stage_outputs = forward_pass(
                            stage_inputs['image'], stage_config, is_training=False, verbose=verbose)

                        # stage 2 (final)
                        if s == 1:                    
                            stage_outputs = format_final_boxes(stage_outputs, crop_boxes)
                            tf.add_to_collection('stage2_pred_bbs', stage_outputs['bounding_boxes'])
                            tf.add_to_collection('stage2_pred_confidences', stage_outputs['detection_scores'])

        # gather predictions across gpus
        with tf.name_scope('gather'):
            eval_outputs = [tf.concat(tf.get_collection(key), axis=0) for key in [
                'inference_image_ids', 'inference_num_boxes', 'inference_gt_bbs', 
                'stage2_pred_bbs', 'stage2_pred_confidences',
                'stage1_pred_bbs', 'stage1_pred_confidences', 'stage1_kept_out_boxes']]        

        # eval functions
        validation_results_path = os.path.join(base_config["log_dir"], 'val_output.txt')
        test_results_path = os.path.join(base_config["log_dir"], 'test_output.txt')

        run_eval = partial(graph_manager.run_eval, eval_split_placehoder=eval_split_placehoder,
                           eval_initializer=eval_initializer, eval_outputs=eval_outputs, configuration=base_config)
        eval_validation = partial(run_eval, mode='val', results_path=validation_results_path)
        eval_test = partial(run_eval, mode='test', results_path=test_results_path)

    
    ########################################################################## Start Session            
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        
    print('number of parameters', total_parameters) 
    print('total graph size: %.2f MB' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6))
    log_run()            
    
    try:        
        with graph_manager.get_monitored_training_session(**base_config) as sess:
            # Initialize from pretrained weights for MobileNet architectures
            configuration.start_from_pretrained(sess)
            
            # Start training
            print('\nStart training:')
            start_time = time.time()
            global_step_ = 0
            train_stage2 = False
            
            try:
                while 1:
                    # Determine whether to start training second stage
                    if not train_stage2:
                        num_epochs = global_step_ // base_config["train_num_iters_per_epoch"]
                        if num_epochs >= args.stage2_starting_epoch:
                            print(('   Epoch %d: %s' if verbose == 1 else '\033[33m   Epoch %d: %s\033[0m') % (
                                num_epochs, 'start training stage 2'))
                            train_stage2 = True
                            
                    # Train       
                    if train_stage2:
                        if False:
                            global_step_, full_loss_, _ = sess.run([
                                global_step, full_loss, train_op])
                        else:
                            global_step_, full_loss_, _, _ = sess.run([
                                global_step, full_loss, train_stage1_op, train_stage2_op])
                    else:
                        global_step_, full_loss_, _ = sess.run([
                            global_step, full_loss[:-1], train_stage1_op])

                    # Display
                    if (global_step_ - 1) % args.display_loss_every_n_steps == 0:
                        viz.display_loss(global_step_, full_loss_, start_time,
                                         base_config["train_num_samples_per_iter"], 
                                         base_config["train_num_samples"])

                    # Evaluate on validation set
                    if (base_config["save_evaluation_steps"] is not None and (global_step_ > 1)
                        and global_step_  % base_config["save_evaluation_steps"] == 0):
                        eval_validation(sess, global_step_)
                        log_run()
            except tf.errors.OutOfRangeError: # End of training
                pass              

            # Evaluate on the validation and test set 
            eval_validation(sess, global_step_)
            eval_test(sess, global_step_)
            log_run()

    except KeyboardInterrupt:          # Keyboard interrupted
        print('\nInterrupted at step %d' % global_step_)
        log_run()