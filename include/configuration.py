from collections import defaultdict
import numpy as np

_defaults_dict = {
    # Architecture 
    "network": 'tiny_yolo_v2',                             # Backbone
    "num_boxes": 1,                                        # Number of boxes per cell to predict
    "with_groups": False,                                  # if True, use grouped instances
    "with_offsets": False,                                 # If True, and with_groups is True, use learned offsets
    "offsets_margin": 0.025,                               # Additional margins to train the offsets
    "with_classification": False,                          # If True, output class scores
    # Inputs
    "batch_size": 12,                                     
    "image_size": 1024,                                    # Input Image Size
    "shuffle_buffer": 2000,                                # Shuffle buffer size
    "data_augmentation_threshold": 0.5,                    # Data augmentation (flip left/right) ratio
    "num_threads": 4,                                      # Number of parallel readers for the dataset map operation
    "prefetch_capacity": 1,                                # prefetch capacity for the dataset object
    # Training Setting
    "learning_rate": 1e-3,                                 # Initial learning rate
    "num_epochs": 100,                                     # Number of training epochs
    "num_gpus": 1,                                         # Number of gpus to use
    "gpu_mem_frac": 1.0,                                   # Memory usage per gpu
    "optimizer": "ADAM",                                   # 'ADAM' or 'MOMENTUM'
    "beta1": 0.9,                                          # If using ADAM optimizer
    "lr_decay_rate": 0.995,                                # If using Momentum optimizer
    "lr_decay_steps": 1,                                   # If using Momentum optimizer
    "momentum": 0.9,                                       # If using momentum optimizer
    # Loss Function
    "centers_localization_loss_weight": 1.0,   
    "scales_localization_loss_weight": 1.0,            
    "classification_loss_weight": 1.0,
    "confidence_loss_weight": 5.0,
    "noobj_confidence_loss_weight": 1.0,
    "group_classification_loss_weight": 1.0,
    "offsets_loss_weight": 1.0,
    # Patch Extraction (training)
    "train_num_crops": 10,                                 # Maximum number of crops per image to predict (train)
    "train_patch_confidence_threshold": 0.0,               # Only keep boxes above this threshold for patch extraction
    "train_patch_nms_threshold": 1.0,                      # IoU threshold for non-maximum suppression during patch extraction
    "patch_intersection_ratio_threshold": 0.33,            # Only keep gt box in the pach if at least this ratio is visible
    # Patch Extraction (inference)
    "test_num_crops": 5,                                   # Maximum number of crops per image to predict (eval)
    "test_patch_nms_threshold": 0.25,                      # IoU threshold for non-maximum suppression during patch extraction
    "test_patch_confidence_threshold": 0.25,               # Only keep boxes above this threshold for patch extraction
    "test_patch_strong_confidence_threshold": 0.75,        # boxes considered 'single' and above this threshold -> no patch
    # Summary and Outputs
    "base_log_dir": "./run_logs",                          # Base log directory
    "max_to_keep": 1,                                      # maximum number of checkpoints to keep
    "save_checkpoint_steps": 2000,                         # Save checkpoints at the given frequency (in seconds)
    "summary_confidence_thresholds": [0.5],                # Plot output boxes cut at the given thresholds  
    "num_summaries": 3,                                    # Number of image to display in summaries
    # Evaluation metrics
    "retrieval_confidence_threshold": 0.,                  # Only keep boxes above this threshold for evaluation
    "retrieval_iou_threshold": [0.5, 0.75],                # Evaluate at these IoU retrieval threshold
    "retrieval_nms_threshold": 0.5,                        # IoU threshold for the Non-maximum suppression during evaluation
}


def get_defaults(kwargs, args, verbose=0):
    """ Set default for agument not in kwargs and print the default value if chosen
    
    Args:
        kwargs: A dict of kwargs
        args: A list of strings. If args is not in kwargs, set it to its default value.
        verbose: Whether to print the default
        
    Returns:
        The list of variables in defaults set to the correct value
    """
    global _defaults_dict
    output = []
    for key in args:
        if key in kwargs:
            v = kwargs[key]
        elif key in _defaults_dict:
            v = _defaults_dict[key]
            if verbose: print('    with default `%s` = %s' % (key, v))
        else:
            raise IndexError('\033[31mError\033[0m: No default value found for parameter \033[31m%s\033[0m' % key)
        output.append(v)
    return output


def build_base_parser(parser):
    """Base parser for common line arguments"""
    parser.add_argument('data', type=str, help='Dataset.', choices=[
        'vedai_fold%02d' % i for i in range(1, 11)] + ['sdd'])
    parser.add_argument('--network', type=str, default="tiny_yolo_v2", help='Architecture."',
                        choices=['tiny_yolo_v2', 'yolo_v2'])
    parser.add_argument('--image_size', default=1024, type=int, help='Size of input images')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs workers to use')
    parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--display_loss_every_n_steps', type=int, default=250, help='Print the loss at every given step')
    parser.add_argument('--save_evaluation_steps', type=int, help='Evaluate validation set at every given step')
    parser.add_argument('--save_summaries_steps', type=int, help='Save summaries tensorboards at every given step')
    parser.add_argument('--verbose', type=int, default=2, help='Extra verbosity')
  
    
def load_metadata(filename):
    """Load and parse the metadata file for a given dataset.
    
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
    
        
def build_base_config_from_args(args, verbose=0):
    """Build the base configuration from the command line arguments"""    
    global _defaults_dict
    
    ## Set initial configuration based on dataset
    configuration = {}
    if args.data.startswith('vedai'):
        configuration['setting'] = args.data
        configuration['exp_name'] = args.data
        configuration['save_summaries_steps'] = args.save_summaries_steps
        configuration['save_evaluation_steps'] = 500 if args.save_evaluation_steps is None else args.save_evaluation_steps
        configuration['num_epochs'] = _defaults_dict['num_epochs'] if args.num_epochs is None else args.num_epochs
        configuration['image_format'] = 'vedai'
        # [Final inference] Cross-validated hyperparameters for ODGI 512-256
        configuration['test_num_crops'] = 3
        configuration['test_patch_nms_threshold'] = 0.25
        configuration['test_patch_confidence_threshold'] = 0.1
        configuration['test_patch_strong_confidence_threshold'] = 0.8
    elif args.data == 'sdd':
        configuration['setting'] = 'sdd'
        configuration['exp_name'] = 'sdd'
        configuration['save_summaries_steps'] = args.save_summaries_steps 
        configuration['save_evaluation_steps'] = 500 if args.save_evaluation_steps is None else args.save_evaluation_steps
        configuration['num_epochs'] = _defaults_dict['num_epochs'] if args.num_epochs is None else args.num_epochs
        configuration['image_format'] = 'sdd'
        # [Final inference] Cross-validated hyperparameters for ODGI 512-256
        configuration['test_num_crops'] = 6
        configuration['test_patch_nms_threshold'] = 0.25
        configuration['test_patch_confidence_threshold'] = 0.1
        configuration['test_patch_strong_confidence_threshold'] = 0.6
    else:
        raise ValueError("unknown data", args.data)
    
    ## ODGI: Choose number of crops during training as to max capacity of the device
    if args.network == 'tiny_yolo_v2':
        configuration['train_num_crops'] = 10
    elif args.network == 'yolo_v2':
        configuration['train_num_crops'] = 6
    elif args.network == 'mobilenet':
        configuration['train_num_crops'] = 10
    
    ## Metadata
    tfrecords_path = 'Data/metadata_%s.txt'
    metadata = load_metadata(tfrecords_path % configuration['setting'])
    configuration.update(metadata)
    if 'data_classes' in configuration:
        configuration['num_classes'] = len(configuration['data_classes'])
    assert 'feature_keys' in configuration
    assert 'image_folder' in configuration
        
    ## Network 
    configuration['network'] = args.network

    ## GPUs
    configuration['num_gpus'] = args.num_gpus
    configuration['gpu_mem_frac'] = max(0., min(1., args.gpu_mem_frac))

    ## Training
    configuration['batch_size'] = args.batch_size
    configuration['learning_rate'] = args.learning_rate
    
    ## Pre-compute number of steps per epoch for loss display
    for split in ['train', 'val', 'test']:
        configuration['%s_num_samples' % split] = configuration['%s_num_samples' % split]
        configuration['%s_num_samples_per_iter' % split] = configuration['batch_size'] * configuration['num_gpus']
        configuration['%s_num_iters_per_epoch' % split] = int(np.ceil(
            configuration['%s_num_samples' % split] / configuration['%s_num_samples_per_iter' % split]))
        print('%d %s samples (%d iters per epoch)' % (
            configuration['%s_num_samples' % split], split, configuration['%s_num_iters_per_epoch' % split]))
        
    ## Print final config (except for grid_offsets)
    if verbose == 2: 
        print('\nConfig:')
        print('\n'.join('   \033[96m%s:\033[0m %s' % (k, v)
                        for k, v in sorted(configuration.items()) 
                        if k != 'grid_offsets'))
    elif verbose == 1:
        print('\nConfig:')
        print('\n'.join('  *%s*: %s' % (k, v) 
                        for k, v in sorted(configuration.items()) 
                        if k != 'grid_offsets'))
    
    ## Return
    return configuration


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
        raise NotImplementedError('Unknown network architecture', network)
    configuration['grid_offsets'] = precompute_grid_offsets(configuration['num_cells'])
    if verbose: print("   using grid size", configuration['num_cells'])   


def get_num_cells(image_size, num_layers):
    """Return the number of grid cells output by a fully convolutional 
        network with `num_layers` down-sampling given the initial image size.
    
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