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


def finalize_grid_offsets(configuration, finalize_retrieval_top_n=True):    
    """Compute the current number of cells and grid offset in the given configuration
    
    Args:
        configuration dictionnary   
    """
    num_filters, top_retrieval_n, num_boxes, image_size = get_defaults(
        configuration, ['num_filters', 'retrieval_top_n', 'num_boxes', 'image_size'])
    configuration['num_cells'] = get_num_cells(image_size, len(num_filters) - 2)
    configuration['grid_offsets'] = precompute_grid_offsets(configuration['num_cells'])
    if finalize_retrieval_top_n:
        configuration['retrieval_top_n'] = min(
            top_retrieval_n, num_boxes * configuration['num_cells'][0] * configuration['num_cells'][1])
    print("grid size", configuration['num_cells'])   
    

def finalize_configuration(configuration, verbose=2):
    """ Finalize and print the given configuration object.
    
    Args:
        configuration dictionnary
        verbose: verbosity mode (0 - low, 1 - verbose with colored output, 2 - simple verbose)
    """
    assert 'feature_keys' in configuration
    assert 'image_folder' in configuration
    
    ## Pre-compute number of steps
    configuration['train_num_samples'] = (configuration['subset'] if configuration['subset'] > 0 
                                          else configuration['train_num_samples'])
    configuration['train_num_samples_per_iter'] = configuration['num_gpus'] * configuration['batch_size']
    configuration['train_num_iters_per_epoch'] = ceil(configuration['train_num_samples'] /
                                                      configuration['train_num_samples_per_iter'])
    configuration['test_num_samples'] = (configuration['subset'] if configuration['subset'] > 0 
                                          else configuration['test_num_samples'])
    configuration['test_num_samples_per_iter'] = configuration['num_gpus'] * configuration['test_batch_size']
    configuration['test_num_iters_per_epoch'] = ceil(configuration['test_num_samples'] /
                                                    configuration['test_num_samples_per_iter'])
    configuration['last_test_batch_size'] = configuration['test_num_samples'] % configuration['test_num_samples_per_iter']

    if "num_epochs" in configuration and not "num_steps" in configuration:
        configuration["num_steps"] = (configuration["num_epochs"] + 1) * configuration['train_num_iters_per_epoch']
        
    ## Print added information
    if 'num_steps' in configuration:
        print("%d training steps" % configuration["num_steps"])
        if configuration["num_steps"] > 0:
            print("...which means %d epochs" % (configuration["num_steps"] // configuration['train_num_iters_per_epoch']))
    print('%d training samples (%d iters)' % (configuration['train_num_samples'], 
                                              configuration['train_num_iters_per_epoch']))
    print('%d validation samples (%d iters)' % (configuration['test_num_samples'], 
                                               configuration['test_num_iters_per_epoch']))    
    
    ## Print final config
    if verbose == 1: 
        print('\n\033[41mConfig:\033[0m')
        print('\n'.join('\033[96m%s:\033[0m %s' % (k, v) for k, v in sorted(configuration.items()) if k != 'grid_offsets'))
    elif verbose > 1:
        print('\nConfig:')
        print('\n'.join('  *%s*: %s' % (k, v) for k, v in sorted(configuration.items()) if k != 'grid_offsets'))
    
    
def get_defaults(kwargs, defaults, verbose=False):
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
            raise IndexError('\033[31mError\033[30: No default value found for parameter %s' % key)
        output.append(v)
    return output


def load_metadata(filename):
    """Load and parse the metadata file.
    
    Args: 
        filename: path to the metadata file
        
    Returns:
        A dictionnary of the metadata values
    """ 
    metadata = {}
    with open(filename, 'r') as f:    
        for line in f.read().splitlines():
            key, values = line.split('\t', 1)
            if key in ['data_classes', 'feature_keys']:
                metadata[key] = values.split(',')
            elif key.endswith('tfrecords') or key == 'image_folder':
                metadata[key] = values
            else:
                metadata[key] = int(values)
        return metadata


def get_num_cells(image_size, num_layers):
    """Return the number of cells in the tiny YOLO v2 output given the initial image size.
    
    Args:
        image_size: Integer specifying the input image square size
        num_layers: Number of convolutional layers
        
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

    
def generate_log_dir(configuration, verbose=True):
    """Generate automatic log directory or use the `fixed_log_dir` entry if given
    
    Args:
        configuration: the config dictionary 
    
    Returns:
        Nothing, but add a `log_dir` entry to the input dictionary
    """
    if "fixed_log_dir" not in configuration:
        base_log_dir, exp_name = get_defaults(configuration, ["base_log_dir", "exp_name"], verbose=verbose)
        configuration["log_dir"] = os.path.join(base_log_dir, exp_name, datetime.now().strftime("%m-%d_%H-%M"))
    else:
        configuration["log_dir"] = configuration["fixed_log_dir"]
        
    
def get_monitored_training_session(with_ready_op=False,
                                   model_path=None,
                                   log_dir=None,
                                   log_device_placement=False,
                                   verbose=True,
                                   **kwargs):
    """Returns a monitored training session object with the specified global configuration.
    Args:
        with_ready_op: Whether to add ready operations to the graph
        model_path: If not None, restore weights from the given ckpt
        log_dir: log directory
        verbose: Controls verbosity level
        **kwargs: Configuration options
        
    Kwargs:
        gpu_mem_frac. Defauts to 1
        num_steps. Defaults to -1 (no limit)
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
    gpu_mem_frac, num_steps, max_to_keep, save_checkpoint_secs, log_globalstep_steps = get_defaults(
        kwargs, ['gpu_mem_frac', 'num_steps', 'max_to_keep', 'save_checkpoint_secs', 'log_globalstep_steps'], verbose=verbose)
    
    # GPU config
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac),
        log_device_placement=log_device_placement,
        allow_soft_placement=True)
        
    # Stop at given number of iterations
    hooks = ([] if num_steps <= 0 else [tf.train.StopAtStepHook(num_steps=num_steps)])
    
    # Summary hooks
    collection = tf.get_collection('outputs')
    if len(collection) > 0:
        save_summaries_steps = get_defaults(kwargs, ['save_summaries_steps'], verbose=verbose)[0]
        hooks.append(tf.train.SummarySaverHook(
            save_steps=save_summaries_steps, output_dir=log_dir, summary_op=tf.summary.merge(collection)))
    else:
        print('    \033[31mWarning:\033[0m No summaries found in collection "outputs"')        
    try:
        hooks.append(tf.train.SummarySaverHook(
            save_steps=1e6, output_dir=log_dir, summary_op=tf.summary.merge_all(key='config')))
    except ValueError:
        print('    \033[31mWarning:\033[0m No summaries found in collection "config"')
    
    # Scaffold    
    if model_path is  None:
        def init_fn(scaffold, sess):
            del scaffold, sess
            return
    else:
        print('    \033[34mRestoring:\033[0m', 'from', model_path)
        restore_scope, restore_replace_to, restore_to_replace,  = get_defaults(
        kwargs, ['restore_scope', 'restore_replace_to', 'restore_to_replace'], verbose=verbose)        
        variables_to_restore = tf.model_variables(scope=restore_scope)
        variables_to_restore = {re.sub(restore_to_replace, restore_replace_to, v.op.name): v
                                for v in variables_to_restore}
        print('    \033[34mRestoring:\033[0m', ', '.join(sorted(list(variables_to_restore.keys()))))
        saver = tf.train.Saver(variables_to_restore, name='restore_saver')
        
        def init_fn(scaffold,sess):
            del scaffold
            saver.restore(sess, model_path)
                
    scaffold = tf.train.Scaffold(
        init_fn=init_fn, 
        ready_op=None if with_ready_op else tf.constant([]),
        ready_for_local_init_op=None if with_ready_op else tf.constant([]),
        saver=tf.train.Saver(max_to_keep=max_to_keep))
    
    # Session object
    return tf.train.MonitoredTrainingSession(
            checkpoint_dir=log_dir,
            config=config,
            scaffold=scaffold,
            hooks=hooks,
            save_checkpoint_secs=save_checkpoint_secs,
            log_step_count_steps=log_globalstep_steps)


def get_inputs(mode='train',
               num_classes=None,
               feature_keys=None,
               image_folder='',
               image_size=-1,
               grid_offsets=None,
               verbose=False,
               **kwargs):
    """ Returns the input pipeline with the given global configuration.
    
    Args:
        mode: one of `train` or `test`. Defaults to `train`.
        num_classes: Number of classes, used to infer the dataset
        feature_keys: List of feature keys present in the TFrecords
        image_folder: path to the image folder. Can contain a format %s that will be replace by `mode`
        image_size: Image sizes
        grid_offsets: Grid offsets
        verbose: verbosity
        **kwargs: Configuration options
        
    Kwargs:
        {train,test}_tf_records. if mode is train, resp. test.
        {train,test}_max_num_bbs. if mode is train, resp. test. Defaults to 1
        (test_)batch_size. if mode is train, resp. test. Defaults to 32
        {train, test}_max_num_merged_bbs. if mode is train, resp. test, and 
            clustered bounding boxes in TFrecord. Defaults to 1
        num_gpus. Defaults to 1
        shuffle_buffer. Defaults to 100
        num_threads. For parallel read. Defaults to 8
        prefetch_capacity. Defaults to 1.
        subset. takes subset of the dataset if strictly positive. Defaults to -1
        data_augmentation_threshold. Defaults to 0.5
        num_boxes: if target_bounding_boxes in record or clustering_fn is not None. Defaults to 1
        
    Returns:
        A list with `num_gpus` element, each being a dictionary of inputs.
    """
    ### Kwargs
    assert image_size > 0
    assert feature_keys is not None  
    assert mode in ['train', 'test']
    assert num_classes in [1, 6, 9, 15, 20, 80]
    (num_gpus, shuffle_buffer, num_threads, prefetch_capacity, subset, data_augmentation_threshold, with_groups,
     with_classes) = get_defaults(kwargs, [
        'num_gpus', 'shuffle_buffer', 'num_threads', 'prefetch_capacity', 'subset', 
        'data_augmentation_threshold', 'with_groups', 'with_classification'], verbose=verbose)    
    assert num_gpus > 0
    assert shuffle_buffer > 0
    
    ## Train vs validation
    pad_with_dummies = 0
    if mode == 'train':
        assert 'train_tfrecords' in kwargs
        tfrecords_path = kwargs['train_tfrecords']
        assert 'train_max_num_bbs' in kwargs
        max_num_bbs = kwargs['train_max_num_bbs']
        batch_size = get_defaults(kwargs, ['batch_size'], verbose=verbose)[0]
    else:    
        assert 'test_tfrecords' in kwargs
        tfrecords_path = kwargs['test_tfrecords']
        assert 'test_max_num_bbs' in kwargs
        max_num_bbs = kwargs['test_max_num_bbs']
        batch_size, last_batch_size = get_defaults(kwargs, ['test_batch_size', 'last_test_batch_size'], verbose=verbose)        
        if last_batch_size > 0: 
            pad_with_dummies = batch_size * num_gpus - last_batch_size
        shuffle_buffer = 1
        data_augmentation_threshold = 0.
        
    try:
        image_folder = image_folder % mode
    except TypeError:
        pass
    if verbose == 1:
        print('    pad \x1b[32m%s\x1b[0m inputs with \x1b[32m%d\x1b[0m dummy samples' % (mode, pad_with_dummies))
    elif verbose > 1:
        print('    pad %s inputs with %d dummy samples' % (mode, pad_with_dummies))
        
    return tf_inputs.get_tf_dataset(
        tfrecords_path,    
        feature_keys,
        max_num_bbs,
        num_classes,
        with_groups=with_groups,
        with_classes=with_classes,
        batch_size=batch_size,
        image_size=image_size,
        image_folder=image_folder,
        data_augmentation_threshold=data_augmentation_threshold,
        grid_offsets=grid_offsets,
        subset=subset,
        num_splits=num_gpus,
        num_threads=num_threads,
        shuffle_buffer=shuffle_buffer,
        prefetch_capacity=prefetch_capacity,
        pad_with_dummies=pad_with_dummies,
        verbose=verbose)        
    

def get_stage2_inputs(inputs,
                      crop_boxes,
                      mode='train',
                      num_classes=None,
                      image_folder='',
                      image_size=-1,
                      grid_offsets=None,
                      verbose=False,
                      **kwargs):
    """ Returns the input pipeline with the given global configuration.
    
    Args:
        inputs, a dictionnary of inputs
        crop_boxes, a (batch_size * num_boxes, 4) tensor of crops
        mode: one of `train` or `test`. Defaults to `train`.
        num_classes: Number of classes, used to infer the dataset
        image_folder: path to the image folder. Can contain a format %s that will be replace by `mode`
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
    assert image_size > 0
    assert mode in ['train', 'test']
    assert num_classes in [6, 9, 15]
    assert len(crop_boxes.get_shape()) == 3
    full_image_size, intersection_ratio_threshold = get_defaults(kwargs, [
        'full_image_size', 'patch_intersection_ratio_threshold'], verbose=verbose)
    
    ## Train: Accumulate crops into queue
    if mode == 'train':
        (batch_size, shuffle_buffer, num_threads) = get_defaults(
            kwargs, ['batch_size', 'shuffle_buffer', 'num_threads'], verbose=verbose)    
        use_queue = True
    ## Val: Pass the output directly
    else:    
        test_batch_size, num_crops = get_defaults(kwargs, ['test_batch_size', 'test_num_crops'], verbose=verbose)
        batch_size = num_crops * test_batch_size
        use_queue = False
        shuffle_buffer = 1
        num_threads = 1
        
    try:
        image_folder = image_folder % mode
    except TypeError:
        pass
    
    return tf_inputs.get_next_stage_inputs(inputs, 
                                           crop_boxes,
                                           image_folder=image_folder,
                                           batch_size=batch_size,
                                           num_classes=num_classes,
                                           image_size=image_size,
                                           full_image_size=full_image_size,
                                           grid_offsets=grid_offsets,
                                           intersection_ratio_threshold=intersection_ratio_threshold,
                                           shuffle_buffer=shuffle_buffer,
                                           num_threads=num_threads,
                                           use_queue=use_queue,
                                           verbose=verbose)
        
    
def add_losses_to_graph(loss_fn, inputs, outputs, configuration, is_chief=False, verbose=False):
    """Add losses to graph collections.
    
    Args:
        loss_fn: Loss function. Should have signature f: (dict, dict, is_chief, **kwargs)
        inputs: inputs dictionary
        outputs: outputs dictionary
        configuration: configuration dictionary
        is_chief: Whether the current process is chief or not
    """
    losses = loss_fn(inputs, outputs, is_chief=is_chief, verbose=verbose, **configuration)
    
    for key, loss in losses:
        if not key.endswith('_loss'):
            print('\033[31mWarning:\033[0m %s will be ignored. Losses name should end with "_loss"' % key)
        tf.add_to_collection(key, loss)
        
        
def get_total_loss(collection='outputs', add_summaries=True, splits=['']):
    """Retrieve the total loss over all collections and all devices.
    All collections ending with '_loss' will be taken as a loss function
    
    Args:
        collection: Summaries collection
        add_summaries: Whether to add summaries to the graph
        splits: Create separate losses (given to different optimizers) for the given scopes. The default just sum all the 
            losses present in the graph for all the trainable variables
        
    Returns:
        A list of tuples (tensor containing the loss, list of variables to optimize)
    """
    print('    Collect losses%s' % ('' if splits == [''] else (' (scopes: %s)' % ','.join(splits))))
    losses = []
    for split in splits:
        ## Collect losses from  `*_loss' collections
        full_loss = 0.
        for key in [x for x in tf.get_default_graph().get_all_collection_keys() if 
                    x.endswith('_loss') and x.startswith(split)]:
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
            
        losses.append((full_loss, tf.trainable_variables(scope=split)))
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
    print('  > Build train operation')
    optimizer, learning_rate = get_defaults(kwargs, ['optimizer', 'learning_rate'], verbose=verbose)
    global_step = tf.train.get_or_create_global_step()    
    print('  > Using optimizer %s with learning rate %.2e' % (optimizer, learning_rate))
        
    # Train op       
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
    
    train_ops = [get_optimizer_op().minimize(
        full_loss, var_list=var_list, colocate_gradients_with_ops=True) for full_loss, var_list in full_losses]
    global_step_op = tf.assign_add(global_step, 1)
    
    # Update op for batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('  >', len(update_ops), 'update operations found')
    
    # Return
    final_op = tf.group(global_step_op, *train_ops, *update_ops) 
    return global_step, final_op


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
    collection = 'outputs' if mode == 'train' else 'evaluation'
    
    # Image summaries
    viz.add_image_summaries(inputs,
                            outputs, 
                            num_summaries,
                            confidence_thresholds=summary_confidence_thresholds,
                            collection=collection,
                            family=family,
                            display_inputs=display_inputs)
    
    
def add_metrics_to_graph(eval_fn, inputs, outputs, metrics_to_norms, clear_ops, update_ops, configuration, 
                         device=0, verbose=True):
    """Add metrics update operations.
    
    Args:
        eval_fn: Eval function. Should have signature f: (dict, dict, **kwargs)
        inputs: inputs dictionary
        outputs: outputs dictionary
        metrics_to_norms: Dictionnary mapping keys of a metric to the key of the denominator to use to normalize it
        clear_op: List to store the clear emtrics operations
        update_ops: List to store the metrics update operations
        configuration: configuration dictionary
        device: Device name, used to create independeng variable scope for each metric
    """
    new_metrics = eval_fn(inputs, outputs, verbose=verbose, **configuration)
    
    for key, norm_key, new_metric in new_metrics:    
        # Create variable on each device
        with tf.variable_scope('metrics_dev_%d' % device):
            metric = tf.get_variable('%s_running' % key, shape=(), initializer=tf.zeros_initializer(),
                                     trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, key])
        metrics_to_norms[key] = norm_key
        # Clear operation
        clear_ops.append(tf.assign(metric, tf.constant(0., dtype=tf.float32, shape=())))
        # Update operation
        update_ops.append(tf.assign_add(metric, new_metric))    
    
    
def get_eval_op(metrics_to_norms, output_values=False): 
    """ Return evaluation summary operation.
    
    Args:
        metrics_keys_to_norms: Maps the name of a metrics collections (key) to another (value) that will be used to 
            normalize the key collection
    """
    metrics = {}
    for metric_key, norm_key in metrics_to_norms.items():
        metric = tf.add_n(tf.get_collection(metric_key))
        if norm_key is None:
            tf.summary.scalar(metric_key, metric, collections=['evaluation'])                
        else:
            norm = tf.maximum(1., tf.add_n(tf.get_collection(norm_key)))
            tf.summary.scalar(metric_key, metric / norm, collections=['evaluation'])
            metrics[metric_key] = metric / norm
    eval_summary_op = tf.summary.merge_all(key='evaluation')
    if output_values:
        return eval_summary_op, metrics
    else:
        return eval_summary_op
    
    
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