import os
import re
import numpy as np
from math import ceil
from datetime import datetime
import tensorflow as tf

from defaults import defaults_dict
import tf_inputs
import tf_utils
import viz

############################################################ Configuration
def get_defaults(kwargs, defaults, verbose=0):
    """ Set default for agument not in kwargs and print the default value if chosen
    
    Args:
        kwargs: A dict of kwargs
        defaults: A list of (key, default_value) pairs. Set key to default_value iff not in kwargs.
        verbose: Whether to print the default
        
    Returns:
        The list of variables in defaults set to the correct value
    """
    output = []
    for key in defaults:
        if key in kwargs:
            v = kwargs[key]
        elif key in defaults_dict:
            v = defaults_dict[key]
            if verbose: print('    with default `%s` = %s' % (key, v))
        else:
            raise IndexError('\033[31mError\033[0m: No default value found for parameter \033[31m%s\033[0m' % key)
        output.append(v)
    return output
   

def finalize_configuration(configuration, verbose=2):
    """ Finalize and print the given configuration object.
    
    Args:
        configuration dictionnary
        verbose: verbosity mode (0 - low, 1 - verbose with colored output, 2 - simple verbose)
    """
    assert 'feature_keys' in configuration
    assert 'image_folder' in configuration
    
    ## Pre-compute number of steps per epoch for loss display
    for split in ['train', 'test']:
        configuration['%s_num_samples' % split] = configuration['%s_num_samples' % split]
        configuration['%s_num_samples_per_iter' % split] = configuration['batch_size']
        if split == 'train':
            configuration['%s_num_samples_per_iter' % split] *= configuration['num_gpus']
        configuration['%s_num_iters_per_epoch' % split] = ceil(configuration['%s_num_samples' % split] /
                                                               configuration['%s_num_samples_per_iter' % split])
        print('%d %s samples (%d iters)' % (
            configuration['%s_num_samples' % split], split, configuration['%s_num_iters_per_epoch' % split]))
    
    ## Print final config (except for grid_offsets)
    if verbose == 2: 
        print('\nConfig:')
        print('\n'.join('   \033[96m%s:\033[0m %s' % (k, v) for k, v in sorted(configuration.items()) if k != 'grid_offsets'))
    elif verbose == 1:
        print('\nConfig:')
        print('\n'.join('  *%s*: %s' % (k, v) for k, v in sorted(configuration.items()) if k != 'grid_offsets'))


def finalize_grid_offsets(configuration, verbose=2):    
    """Compute the current number of cells and grid offset in the given configuration
    
    Args:
        configuration dictionnary   
    """
    network, image_size = get_defaults(configuration, ['network', 'image_size'])
    if network == 'tiny_yolo_v2':        
        configuration['num_cells'] = get_num_cells(image_size, 5)
    elif network == 'yolo_v2':        
        configuration['num_cells'] = get_num_cells(image_size, 5)
    elif network == 'mobilenet':
        configuration['num_cells'] = get_num_cells(image_size, 5)        
    else:
        ### TODO
        configuration['num_cells'] = get_num_cells(image_size, 5)  
        #raise NotImplementedError('Uknown network architecture', network)
    configuration['grid_offsets'] = precompute_grid_offsets(configuration['num_cells'])
    if verbose:
        print("   using grid size", configuration['num_cells'])   


def get_num_cells(image_size, num_layers):
    """Return the number of cells a fully convolutional output given the initial image size.
    
    Args:
        image_size: Integer specifying the input image square size
        num_layers: Number of downsize-steps layers
        
    Returns:
        a 2D array containing the number of cells on each axis
    """
    num_cells = np.array([image_size, image_size])
    for _ in range(num_layers):
        num_cells = np.ceil(num_cells / 2.)
    return num_cells.astype(np.int32)


def precompute_grid_offsets(num_cells):
    """Precompute the grid cells top left corner coordinates as a numpy array
    
    Args:
        num_cells a 2D array indicating the number of cells along each dimension
        
    Returns:
        A (num_cells[0], num_cells[1], 1, 2) numpy array
    """
    offsets_x = np.arange(num_cells[0])
    offsets_y = np.arange(num_cells[1])
    offsets = np.meshgrid(offsets_x, offsets_y)
    offsets = np.expand_dims(np.stack(offsets, axis=-1), axis=-2)
    return offsets

    
############################################################ Monitored Session   
def generate_log_dir(configuration, verbose=1):
    """Generate automatic log directory based on the current system time or use the `fixed_log_dir` entry if given
    
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
                                   model_path=None,
                                   log_dir=None,
                                   log_device_placement=False,
                                   allow_soft_placement=True,
                                   verbose=True,
                                   **kwargs):
    """Returns a monitored training session object with the specified global configuration.
    Args:
        with_ready_op: Whether to add ready operations to the graph
        model_path: If not None, restore weights from the given ckpt
        log_dir: log directory. 
        log_device_placement: Whether to log the Tensorflow device placement
        allow_soft_placement: Whether to allow Tensorflow soft device placement
        verbose: Controls verbosity level
        **kwargs: Additional configuration options, will be queried for:
            gpu_mem_frac. Defauts to 1
            max_to_keep. Number of checkpoints to keep save, defaults to 1
            save_checkpoints_sec. Defaults to 600 (save checkpoint every 10mn)
            log_globalstep_steps. Defaults to 500
            save_summaries_steps. If `outputs` summaries exist. Defaults to 100
            save_text_summaries_steps. If `outputs_text` summaries exist. Defaults to 100
            restore_scope. If `model_path` is not None. Defaults to None
            restore_replace_to. If `model_path` is not None. Defaults to ''
            restore_to_replace. If `model_path` is not None. Defaults to ''
        
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
    collection = tf.get_collection('outputs_summaries')
    if len(collection) > 0:
        save_summaries_steps = get_defaults(kwargs, ['save_summaries_steps'], verbose=verbose)[0]
        hooks.append(tf.train.SummarySaverHook(
            save_steps=save_summaries_steps, output_dir=log_dir, summary_op=tf.summary.merge(collection)))
    else:
        print('    \033[31mWarning:\033[0m No summaries found in collection "outputs_summaries"') 
        
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


############################################################ Inputs tf.data.Dataset  
def get_inputs(mode='train',
               shard_index=0,
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
    assert image_size > 0
    assert feature_keys is not None  
    assert mode in ['train', 'val', 'test']
    (num_threads, prefetch_capacity, batch_size, with_groups, with_classes) = get_defaults(kwargs, [
        'num_threads', 'prefetch_capacity', 'batch_size', 'with_groups', 'with_classification'], verbose=verbose)
    num_classes = get_defaults(kwargs, ['num_classes'], verbose=verbose)[0] if with_classes else None
    
    ## Set args
    if mode == 'train':
        assert 'train_tfrecords' in kwargs
        tfrecords_path = kwargs['train_tfrecords']
        assert 'train_max_num_bbs' in kwargs
        max_num_bbs = kwargs['train_max_num_bbs']
        shuffle_buffer, data_augmentation_threshold, num_epochs, num_shards = get_defaults(
            kwargs, ['shuffle_buffer', 'data_augmentation_threshold', 'num_epochs', 'num_gpus'], verbose=verbose)
        make_initializable_iterator = False
    else:  
        assert '%s_tfrecords' % mode in kwargs
        tfrecords_path = kwargs['%s_tfrecords' % mode]
        assert '%s_max_num_bbs' % mode in kwargs
        max_num_bbs = kwargs['%s_max_num_bbs' % mode]
        shuffle_buffer = 1
        data_augmentation_threshold = 0.
        num_epochs = 1
        shard_index = 0
        num_shards = 1
        make_initializable_iterator = True
        
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
        num_epochs=num_epochs,
        image_size=image_size,
        image_folder=image_folder,
        data_augmentation_threshold=data_augmentation_threshold,
        grid_offsets=grid_offsets,
        num_shards=num_shards,
        shard_index=shard_index,
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
    full_image_size, intersection_ratio_threshold = get_defaults(kwargs, [
        'full_image_size', 'patch_intersection_ratio_threshold'], verbose=verbose)
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
                                           full_image_size=full_image_size,
                                           grid_offsets=grid_offsets,
                                           intersection_ratio_threshold=intersection_ratio_threshold,
                                           shuffle_buffer=shuffle_buffer,
                                           num_threads=num_threads,
                                           use_queue=use_queue,
                                           verbose=verbose)
        
    
############################################################ Training operation
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
        
        
def get_total_loss(collection='outputs_summaries', add_summaries=True, splits=[''], verbose=0):
    """Retrieve the total loss over all collections and all devices.
    All collections ending with '_loss' will be taken as a loss function
    
    Args:
        collection: Summaries collection
        add_summaries: Whether to add summaries to the graph
        splits: Create separate losses (given to different optimizers) for the given scopes. The default just sum all the 
            losses present in the graph for all the trainable variables
        verbose: Verbosity level
        
    Returns:
        A list of tuples (tensor containing the loss, list of variables to optimize)
    """
    if verbose == 1:
        print(' > Collecting losses%s' % ('' if splits == [''] else (' (scopes: %s)' % ', '.join(splits))))
    elif verbose == 2:
        print(' \033[31m> Collecting losses\033[0m%s' % (
            '' if splits == [''] else (' (scopes: %s)' % ', '.join(splits))))    
    losses = []
    for split in splits:
        ## Collect losses from  `*_loss' collections
        full_loss = 0.
        loss_collections = [x for x in tf.get_default_graph().get_all_collection_keys() if 
                            x.endswith('_loss') and x.startswith(split)]
        for key in loss_collections:
            collected = tf.get_collection(key)
            loss = tf.add_n(collected) / float(len(collected))
            full_loss += loss
            if add_summaries:
                base_name = key.split('_', 1)[0]
                tf.summary.scalar(key, loss, collections=[collection], family='train_%s' % base_name)

        ## Add regularization loss if any
        reg_losses = tf.losses.get_regularization_losses()
        if len(reg_losses):
            regularization_loss = tf.add_n(reg_losses)
            full_loss += regularization_loss
            if add_summaries:
                tf.summary.scalar('%sregularization_loss' % split, regularization_loss, collections=[collection])

        ## Summary
        if add_summaries:
            tf.summary.scalar('%stotal_loss' % split, full_loss, collections=[collection]) 
            
        train_vars = tf.trainable_variables(scope=split)
        losses.append((full_loss, train_vars, split))
        if verbose == 2:
            print('\n    Scope %s' % split)
            print('\n'.join(["        *%s*: %s tensors" % (x, len(tf.get_collection(x)))  
                             for x in tf.get_default_graph().get_all_collection_keys() if x.endswith('_loss')]))
            print('       Trainable variables: ', ', '.join(list(map(lambda x: x.name, train_vars))))
            
    # Return
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
        
    # Optimizer    
    global_step = tf.train.get_or_create_global_step()    
    optimizer, learning_rate = get_defaults(kwargs, ['optimizer', 'learning_rate'], verbose=verbose)
    if verbose:
        print('    Using optimizer %s with learning rate %.2e' % (optimizer, learning_rate))
    if optimizer == 'MOMENTUM':
        learning_rate_decay_steps, learning_rate_decay_rate = get_defaults(
            kwargs, ['lr_decay_steps', 'lr_decay_rate', 'momentum'], verbose=verbose)
        lr = tf.train.exponential_decay(
            learning_rate, global_step, learning_rate_decay_steps, learning_rate_decay_rate, staircase=False)
        tf.summary.scalar('learning_rate', lr, collections=['outputs'])        
        get_optimizer_op = lambda learning_rate=lr, momentum=momentum: tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=momentum)
    elif optimizer == 'ADAM':
        beta1 = get_defaults(kwargs, ['beta1'], verbose=verbose)[0]
        get_optimizer_op = lambda learning_rate=learning_rate, beta1=beta1: tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1)
    else:
        raise NotImplementedError(optimizer_type)
    
    # Train op for each stage
    train_ops = []
    for full_loss, var_list, scope in full_losses:
        update_ops = [x for x in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if scope in x.name]
        print('   ', len(update_ops), 'update operations found in scope', scope)
        with tf.control_dependencies(update_ops):
            train_op = get_optimizer_op().minimize(full_loss, var_list=var_list, colocate_gradients_with_ops=True)
        train_ops.append(train_op)
        
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #print('   ', len(update_ops), 'update operations found in scope', scope)
    #with tf.control_dependencies(update_ops):
        #train_ops = [get_optimizer_op().minimize(full_loss, var_list=var_list, colocate_gradients_with_ops=True) 
        #             for full_loss, var_list in full_losses]
    
    # Collect update_ops for batch norm
    
    # Return
    global_step_op = tf.assign_add(global_step, 1)
    return global_step_op, train_ops


############################################################ Train Summaries
def add_summaries(inputs, 
                  outputs, 
                  mode='train',
                  family=None,
                  display_inputs=True,
                  verbose=True,
                  **kwargs):
    """Add summaries
    
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
    collection = 'outputs_summaries' if mode == 'train' else 'evaluation'
    
    # Image summaries
    viz.add_image_summaries(inputs,
                            outputs, 
                            num_summaries,
                            confidence_thresholds=summary_confidence_thresholds,
                            collection=collection,
                            family=family,
                            display_inputs=display_inputs)
    
    
############################################################ USELESS 
def extract_config(event_file):
    """Extract config dictionnary for a Tensorboard event file"""
    import ast
    def parse_config_line(l, config):
        key, value = l.split(' = ', 1)
        key = key.replace('"', '').lower()
        try:
            config[key] = ast.literal_eval(value)
        except(ValueError, SyntaxError):
            config[key] = value
        
    configs = {}
    for event in tf.train.summary_iterator(event_file):
        vals = event.summary.value
        # Find the configuration summaries
        if len(vals) and 'config_summary' in vals[0].tag:
            for val in vals:
                # Get content
                header = val.tag.split('/')
                if header[-1] == 'configuration':
                    base_name = header[-2]
                    obj = val.tensor.string_val
                    assert len(obj) == 1
                    config = {}
                    # Parse
                    for line in obj[0].decode("utf-8").split('\n\t')[1:]:
                        parse_config_line(line, config)
                    configs[base_name] = config
            return configs