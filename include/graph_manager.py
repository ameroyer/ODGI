import os
from datetime import datetime
import tensorflow as tf

from .configuration import get_defaults
from . import eval_utils
from . import tf_inputs
from . import viz   

    
def generate_log_dir(configuration, verbose=1):
    """Generate  log directory based on the current system time 
    otherwise use the `fixed_log_dir` entry in the config dictionnary if given
    
    Args:
        configuration: the config dictionary 
        verbose: Verbosity level
    
    Returns:
        Nothing, but add a `log_dir` entry to the input dictionary
    """
    if "fixed_log_dir" not in configuration:
        base_log_dir, exp_name = get_defaults(configuration, ["base_log_dir", "exp_name"], verbose=verbose)
        configuration["log_dir"] = os.path.join(base_log_dir, exp_name, datetime.now().strftime("%m-%d_%H-%M"))
    else:
        configuration["log_dir"] = configuration["fixed_log_dir"]
    if not os.path.exists(configuration["log_dir"]):
        os.makedirs(configuration["log_dir"])
        
    
def get_monitored_training_session(with_ready_op=False,
                                   log_dir=None,
                                   allow_soft_placement=True,
                                   log_device_placement=False,
                                   verbose=True,
                                   **kwargs):
    """Returns a monitored training session object with the specified global configuration.
    Args:
        with_ready_op: Whether to add ready operations to the graph
        log_dir: log directory. 
        log_device_placement: Whether to log the Tensorflow device placement
        allow_soft_placement: Whether to allow Tensorflow soft device placement
        verbose: Controls verbosity level
        **kwargs: Additional configuration options, will be queried for:
            gpu_mem_frac. Defauts to 1
            max_to_keep. Number of checkpoints to keep save, defaults to 1
            save_checkpoints_step. Frequency for checkpoint saving 
            save_summaries_steps. Frequency of summary saving
        
    Returns:
        A tf.train.MonitoredTrainingSeccion object.
    """
    # Kwargs
    assert  log_dir is not None
    gpu_mem_frac, max_to_keep, save_checkpoint_steps = get_defaults(
        kwargs, ['gpu_mem_frac', 'max_to_keep', 'save_checkpoint_steps'], verbose=verbose)
    
    # GPU config
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac),
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
            
    # Summary hooks
    hooks = []
    collection = tf.get_collection('outputs')
    if len(collection) > 0:
        save_summaries_steps = get_defaults(kwargs, ['save_summaries_steps'], verbose=verbose)[0]
        assert save_summaries_steps is not None
        hooks.append(tf.train.SummarySaverHook(
            save_steps=save_summaries_steps, output_dir=log_dir, summary_op=tf.summary.merge(collection)))
    else:
        print('    \033[31mWarning:\033[0m No summaries found in collection "outputs"')        
    try:
        hooks.append(tf.train.SummarySaverHook(
            save_steps=1e6, output_dir=log_dir, summary_op=tf.summary.merge_all(key='config')))
    except ValueError:
        print('    \033[31mWarning:\033[0m No summaries found in collection "config"') 
        
    # Model saving hooks    
    if save_checkpoint_steps is not None:
        print('    saving checkpoint in \033[36m%s\033[0m' % log_dir)
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        hooks.append(tf.train.CheckpointSaverHook(log_dir, saver=saver, save_steps=save_checkpoint_steps))
            
    # Scaffold      
    init_iterator_op = tf.get_collection('iterator_init')
    local_init_op = tf.group(tf.local_variables_initializer(), *init_iterator_op)
    scaffold = tf.train.Scaffold(
        local_init_op=local_init_op,
        ready_op=None if with_ready_op else tf.constant([]),
        ready_for_local_init_op=None if with_ready_op else tf.constant([]))
    
    # Session object
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, checkpoint_dir=log_dir, config=config)
    return tf.train.MonitoredSession(session_creator=session_creator, hooks=hooks)


def get_inputs(mode='train',
               feature_keys=None,
               image_folder='',
               image_format=None,
               image_size=-1,
               grid_offsets=None,
               verbose=0,
               **kwargs):
    """ Returns a dataset iterator on the initial input.
    
    Args:
        mode: one of `train` or `test`. Defaults to `train`.
        feature_keys: List of feature keys present in the TFrecords
        image_folder: path to the image folder. Can contain a format %s that will be replace by `mode`
        image_format: used to infer the dataset and loading format
        image_size: Image size
        grid_offsets: Precomputed grid offsets
        verbose: verbosity
        **kwargs: Additional configuration options, will be queried for:
            {train,test}_tf_records. if mode is train, resp. test.
            {train,test}_max_num_bbs. if mode is train, resp. test. Number of ground-truth bounding boxes. Defaults to 1
            batch_size. Batch size per device. Defaults to 16
            num_epochs: Number of epochs to run
            num_gpus. Defaults to 1
            shuffle_buffer. size of the shuffle buffer. Defaults to 1
            num_threads. For parallel read. Defaults to 8
            prefetch_capacity. Defaults to 1
            subset. takes subset of the dataset if strictly positive. Defaults to -1
            data_augmentation_threshold. Defaults to 0.5 (train)
            with_groups: whether to precompute groups based on the grid
            with_classification: whether to load classes
        
    Returns:
        A tf.data.Dataset iterator (and its initializer, if mode is test or val)
    """
    ### Kwargs
    assert '%s_tfrecords' % mode in kwargs
    assert '%s_max_num_bbs' % mode in kwargs
    (num_threads, prefetch_capacity, batch_size, num_devices, with_groups, with_classes) = get_defaults(
        kwargs, ['num_threads', 'prefetch_capacity', 'batch_size', 'num_gpus', 
                 'with_groups', 'with_classification'], verbose=verbose)
    num_classes = get_defaults(kwargs, ['num_classes'], verbose=verbose)[0] if with_classes else None
    tfrecords_path = kwargs['%s_tfrecords' % mode]
    max_num_bbs = kwargs['%s_max_num_bbs' % mode]
    
    ## Set args
    if mode == 'train':
        shuffle_buffer, data_augmentation_threshold, num_epochs = get_defaults(
            kwargs, ['shuffle_buffer', 'data_augmentation_threshold', 'num_epochs'], verbose=verbose)
        drop_remainder = True
        make_initializable_iterator = False
    elif mode in ['val', 'test']:  
        shuffle_buffer = 1
        num_epochs = 1
        data_augmentation_threshold = 0.
        drop_remainder = False
        make_initializable_iterator = True
    else:
        raise NotImplementedError("Unknown mode for `get_inputs`:", mode)
        
    try:
        image_folder = image_folder % mode
    except TypeError:
        pass
        
    return tf_inputs.get_tf_dataset(
        tfrecords_path,    
        feature_keys,
        image_format,
        max_num_bbs,
        with_groups=with_groups,
        with_classes=with_classes,
        num_classes=num_classes,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        num_epochs=num_epochs,
        image_size=image_size,
        image_folder=image_folder,
        data_augmentation_threshold=data_augmentation_threshold,
        grid_offsets=grid_offsets,
        num_devices=num_devices,
        num_threads=num_threads,
        shuffle_buffer=shuffle_buffer,
        prefetch_capacity=prefetch_capacity,
        make_initializable_iterator=make_initializable_iterator,
        verbose=verbose)        
    

def get_stage2_inputs(inputs,
                      crop_boxes,
                      mode='train',
                      image_folder='',
                      image_format=None,
                      image_size=-1,
                      grid_offsets=None,
                      verbose=False,
                      **kwargs):
    """ Extract patches to create .
    
    Args:
        inputs, a dictionnary of inputs
        crop_boxes, a (batch_size * num_boxes, 4) tensor of crops
        mode: one of `train`, `val` or `test`. Defaults to `train`.
        image_folder: path to the image folder. Can contain a format %s that will be replace by `mode`
        image_format: used to infer the dataset and loading format
        image_size: Image sizes
        grid_offsets: Grid offsets
        verbose: verbosity
        **kwargs: Configuration options
        
    Kwargs:
        (test_)batch_size. if mode is train, resp. val. Defaults to 32
        shuffle_buffer. Defaults to 100
        num_threads. For parallel read. Defaults to 8
        
    Returns:
        A list with `num_gpus` element, each being a dictionary of inputs.
    """
    assert 'batch_size' in kwargs
    assert 'previous_batch_size' in kwargs
    assert image_size > 0
    assert mode in ['train', 'val', 'test']
    assert len(crop_boxes.get_shape()) == 3
    intersection_ratio_threshold = get_defaults(kwargs, ['patch_intersection_ratio_threshold'], verbose=verbose)[0]
    previous_batch_size = kwargs['previous_batch_size']
    batch_size = kwargs['batch_size']
    
    ## Train: Accumulate crops into queue
    if mode == 'train':
        (shuffle_buffer, num_threads) = get_defaults(kwargs, ['shuffle_buffer', 'num_threads'], verbose=verbose) 
        use_queue = (batch_size is not None)
    ## Eval: Pass the output directly to the next stage, sequential execution
    else:    
        num_crops = get_defaults(kwargs, ['test_num_crops'], verbose=verbose)[0]
        use_queue = False
        shuffle_buffer = 1
        num_threads = 1
        
    try:
        image_folder = image_folder % mode
    except TypeError:
        pass
    
    return tf_inputs.get_next_stage_inputs(inputs, 
                                           crop_boxes,
                                           batch_size=batch_size,
                                           previous_batch_size=previous_batch_size,
                                           image_folder=image_folder,
                                           image_format=image_format,
                                           image_size=image_size,
                                           grid_offsets=grid_offsets,
                                           intersection_ratio_threshold=intersection_ratio_threshold,
                                           shuffle_buffer=shuffle_buffer,
                                           num_threads=num_threads,
                                           use_queue=use_queue,
                                           verbose=verbose)
        
    
def add_losses_to_graph(loss_fn, inputs, outputs, configuration, is_chief=False, verbose=0):
    """Add losses to graph collections.
    
    Args:
        loss_fn: Loss function. Should have signature f: (dict, dict, is_chief, **kwargs) -> tuple of losses and names
        inputs: inputs dictionary
        outputs: outputs dictionary
        configuration: configuration dictionary
        is_chief: Whether the current process is chief or not
        verbose: Verbosity level
    """
    losses = loss_fn(inputs, outputs, is_chief=is_chief, verbose=is_chief * verbose, **configuration)    
    for key, loss in losses:
        if not key.endswith('_loss') and is_chief:
            print('\033[31mWarning:\033[0m %s will be ignored. Losses name should end with "_loss"' % key)
        tf.add_to_collection(key, loss)
        
        
def get_total_loss(splits=[''], collection='outputs', with_summaries=True, verbose=0):
    """Retrieve the total loss over all collections and all devices.
    All collections ending with '_loss' will be taken as a loss function
    
    Args:
        splits: Create separate losses (given to different optimizers) for the given scopes.
            By just sum all the losses present in the graph for all the trainable variables
        collection: collection to add loss summaries to
        with_summaries: Whether to add summaries to the graph
        verbose: Verbosity level
        
    Returns:
        A list of tuples (tensor containing the loss, list of variables to optimize)
    """      
    losses = []
    for split in splits:
        full_loss = 0.
        loss_collections = [x for x in tf.get_default_graph().get_all_collection_keys() if 
                            x.endswith('_loss') and x.startswith(split)]
        ## sum losses
        for key in loss_collections:
            collected = tf.get_collection(key)
            loss = tf.add_n(collected) / float(len(collected))
            full_loss += loss
            if with_summaries:
                base_name = key.split('_', 1)[0]
                tf.summary.scalar(key, loss, collections=[collection], family='train_%s' % base_name)

        ## Add regularization loss if any
        reg_losses = tf.losses.get_regularization_losses()
        if len(reg_losses):
            regularization_loss = tf.add_n(reg_losses)
            full_loss += regularization_loss
            if with_summaries:
                tf.summary.scalar('%sregularization_loss' % split, regularization_loss, collections=[collection])

        ## Summary for the total loss in the current scope
        if with_summaries:
            tf.summary.scalar('%stotal_loss' % split, full_loss, collections=[collection]) 
            
        ## Add losses and corresponding variables
        train_vars = tf.trainable_variables(scope=split)
        losses.append((full_loss, train_vars, split))
        if verbose == 2:
            print('    in %s scope:' % (split if split else "global"))
            if len(loss_collections):
                print('\n'.join(["        *%s*: %s tensors" % (x, len(tf.get_collection(x)))
                                 for x in loss_collections]))
            else:
                print('        \033[31mWarning:\033[0m No losses found with base name', split)
            print('        Trainable variables: [%s]' % ', '.join(list(map(lambda x: x.name, train_vars))))
    return losses


def get_train_op(full_losses, 
                 verbose=True,
                 **kwargs):
    """Return the global step and graph operation (training, update batch norm and moving average)
    
    Args:
        full_loss: the total loss tensor
        
    Kwargs:
        optimizer. Defaults to ADAM
        learning_rate: Defaults to 0.001
        with_ema. Exponential moving average. Defaults to False
        lr_decay_steps. If optimizer si MOMENTUM
        lr_decay_rate. If optimizer si MOMENTUM
        momentum. If optimizer si MOMENTUM
        
    Returns:
        global step tensor
        train operation
    """
    if verbose == 1:
        print(' > Build train operation')
    elif verbose == 2:
        print(' \033[31m> Build train operation\033[0m')
        
    # Master global step
    global_step = tf.train.get_or_create_global_step()    
    global_step_op = tf.assign_add(global_step, 1)
    
    # Define optimizer
    optimizer, learning_rate = get_defaults(kwargs, ['optimizer', 'learning_rate'], verbose=verbose)
    if verbose:
        print('    Using optimizer %s with learning rate %.2e' % (optimizer, learning_rate))
        
    # Optimizer    
    if optimizer == 'MOMENTUM':
        learning_rate_decay_steps, learning_rate_decay_rate, momentum = get_defaults(
            kwargs, ['lr_decay_steps', 'lr_decay_rate', 'momentum'], verbose=verbose)
        lr = tf.train.exponential_decay(learning_rate, 
                                        global_step, 
                                        learning_rate_decay_steps, 
                                        learning_rate_decay_rate,
                                        staircase=False)        
        get_optimizer_op = lambda learning_rate=lr, momentum=momentum: tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=momentum)
    elif optimizer == 'ADAM':
        beta1 = get_defaults(kwargs, ['beta1'], verbose=verbose)[0]
        get_optimizer_op = lambda learning_rate=learning_rate, beta1=beta1: tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1, epsilon=1e-4)
    else:
        raise NotImplementedError(optimizer_type)
    
    # Train op for each split
    train_ops = []
    for full_loss, var_list, scope in full_losses:
        update_ops = [x for x in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if scope in x.name]
        print('   ', len(update_ops), 'update operations found in %s scope' % (scope if scope else "global"))
        with tf.control_dependencies(update_ops):
            train_op = get_optimizer_op().minimize(
                full_loss, var_list=var_list, colocate_gradients_with_ops=True)
        train_ops.append(train_op)
    
    # Return
    return global_step_op, train_ops


def add_summaries(inputs, 
                  outputs, 
                  mode='train',
                  family=None,
                  display_inputs=True,
                  verbose=True,
                  **kwargs):
    """Add summaries to the graph
    
    Args:
        inputs dictionary
        outputs dictionary
        mode: one of `train` or `test`
        
    Kwargs:    
        gt_bbs_key: Key to ground-truth bounding boxes in the input dictionnary
        num_summaires: Number of image summaries per batch
        summary_confidence_thresholds: threhsolds at which to plot output bounding boxes
    """
    assert mode in ['train', 'test']
    num_summaries, summary_confidence_thresholds = get_defaults(
        kwargs, ['num_summaries', 'summary_confidence_thresholds'], verbose=verbose)
    del kwargs
    collection = 'outputs' if mode == 'train' else 'evaluation'
    
    # Image summaries
    viz.add_image_summaries(inputs,
                            outputs, 
                            num_summaries,
                            confidence_thresholds=summary_confidence_thresholds,
                            collection=collection,
                            family=family,
                            display_inputs=display_inputs)
    
    
def run_eval(sess, 
             global_step_,
             eval_split_placehoder,
             eval_initializer,
             eval_outputs, 
             mode, 
             results_path,
             configuration):
    """Run evaluation in a given session
    
    Args:
        sess: Current session
        eval_outputs: Tensor outputs to be parsed. See `eval_utils` functions.
        mode: Whether to run on the validation or test split
        global_step_: Current global step, for display purpose
        results_path: Path to a file where to log results
    """
    assert mode in ['val', 'test']
    feed_dict = {eval_split_placehoder: mode == 'test'}
    with open(results_path, 'w') as f:
        f.write('%s results at step %d\n' % (mode, global_step_))
    sess.run(eval_initializer, feed_dict=feed_dict)
    
    try:
        while 1:   
            out_ = sess.run(eval_outputs,  feed_dict=feed_dict)
            eval_utils.append_detection_outputs(results_path, *out_, **configuration)
    except tf.errors.OutOfRangeError:
        pass
    
    eval_aps, eval_aps_thresholds, num_images = eval_utils.detect_eval(results_path, **configuration)
    print('\revaluated %d %s images at step %d:' % (num_images, mode, global_step_), ' - '.join(
        'map@%.2f = %.5f' % (thresh, sum(x[t] for x in eval_aps.values()) / len(eval_aps))
        for t, thresh in enumerate(eval_aps_thresholds)))