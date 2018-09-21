##### Default option
defaults_dict = {
    # Base 
    "batch_size": 16,                                     
    "base_name": "net",                                    # Base name for variable scopes
    "exp_name": "exp",                                     # Base name for current run's directory
    "network": 'tiny-yolov2',                              # One of tiny-yolov2 or yolov2
    "num_boxes": 1,                                        # Number of boxes per cell to predict
    "with_groups": False,                                  # if True, use grouped instances
    "with_group_flags": False, 
    "with_offsets": False,                                 # If True, use learned offsets
    "with_classification": False,                          # If True, output class scores 
    # Data
    "data_augmentation_threshold": 0.5,                    # Data augmentation (flip lr) ratio
    "image_size": 512,                                     # Input Image Size
    "full_image_size": 1024,                               # image size to load for extracting the crops
    "shuffle_buffer": 2000,                                # Shuffle buffer size in the main dataset object
    "num_threads": 8,                                      # Number of parallel readers for the input queues
    "prefetch_capacity": 1,                                # prefetch capacity for the main dataset object
    "subset": -1,                                          # If > 0, select a subset of the dataset
    # Training Setting
    "learning_rate": 1e-3,
    "num_epochs": 100,                                     # Number of training epochs
    "num_gpus": 1,                                         # Number of gpus to use
    "gpu_mem_frac": 1.0,                                   # memory usage per gpu
    "optimizer": "ADAM",                                   # 'ADAM' or 'MOMENTUM'
    "beta1": 0.9,                                          # If using ADAM optimizer
    "lr_decay_rate": 0.995,                                # If using Momentum optimizer
    "lr_decay_steps": 1,                                   # If using Momentum optimizer
    "momentum": 0.9,                                       # If using momentum optimizer
    "normalizer_decay": 0.9,                               # Batch norm decay
    "weight_decay": 0.,
    # Loss Function
    "assignment_reward_fn": "iou",                         # function for assigning gt to best predictor
    "target_conf_fn": "iou",                               # function for to compute the confidence scores ground-truth 
    "offsets_margin": 0.025,                               # Additional margins to learn the offsets
    "centers_localization_loss_weight": 1.0,   
    "scales_localization_loss_weight": 1.0,            
    "classification_loss_weight": 1.0,
    "confidence_loss_weight": 5.0,
    "noobj_confidence_loss_weight": 1.0,
    "group_classification_loss_weight": 1.,
    "offsets_loss_weight": 1.0,
    # Patch Extraction (train time)
    "train_num_crops": 5,                                 # Maximum number of crops per image to predict (train)
    "train_patch_confidence_threshold": 0.0,               # Only keep boxes above this threshold for patch extraction
    "train_patch_nms_threshold": 1.0,                      # IoU threshold for non-maximum suppression during patch extraction
    "patch_intersection_ratio_threshold": 0.33,            # Only keep gt box in the pach if at least this ratio is visible
    # Patch Extraction (eval time)
    "test_num_crops": 5,                                   # Maximum number of crops per image to predict (eval)
    "test_patch_nms_threshold": 0.25,                      # IoU threshold for non-maximum suppression during patch extraction
    "test_patch_confidence_threshold": 0.25,               # Only keep boxes above this threshold for patch extraction
    "test_patch_strong_confidence_threshold": 0.75,        # boxes considered 'single' and above this threshold -> no patch
    # Evaluation and Outputs
    "base_log_dir": "./log",                               # Base log directory
    "log_globalstep_steps": 100,
    "max_to_keep": 1,                                      # maximum number of checkpoints to keep
    "num_summaries": 3,                                    # Number of image to display in summaries
    "retrieval_confidence_threshold": 0.,                  # Only keep boxes above this threshold for evaluation
    "retrieval_iou_threshold": [0.5, 0.75],                # Evaluate at these retrieval threshold
    "retrieval_nms_threshold": 0.5,                        # IoU threshold for the Non-maximum suppression during evaluation
    "save_checkpoint_secs": 3600,                          # Save checkpoints at the given frequency (in seconds)
    "save_evaluation_steps": 1000,
    # Load pretrained model
    "restore_replace_to": '',                              # string to replace to in variable names to restore
    "restore_scope": None,                                 # If a checkpoint path is given, restore from the given scope
    "restore_to_replace": '',                              # string to replace in variable names to restore
    "summary_confidence_thresholds": [0.5],                # Plot output boxes cut at the given thresholds  
}

def build_base_parser(parser):
    """Add common arguments to the base parser"""
    parser.add_argument('data', type=str, help='Dataset. One of "vedai", "stanford" or "dota"')
    parser.add_argument('--network', type=str, default="tiny-yolov2", help='Architecture. One of "tiny-yolov2" or "yolov2"')
    parser.add_argument('--size', default=1024, type=int, help='Size of input images')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs workers to use')
    parser.add_argument('--gpu_mem_frac', type=float, default=1., help='Memory fraction to use for each GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--display_loss_very_n_steps', type=int, default=200, help='Print the loss at every given step')
    parser.add_argument('--save_evaluation_steps', type=int, help='Print the loss at every given step')
    parser.add_argument('--summaries', action='store_true', help='Save Tensorboard summaries while training')
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
            elif key.endswith('tfrecords') or key in ['image_format', 'image_folder']:
                metadata[key] = values
            else:
                metadata[key] = int(values)
        return metadata
    
    
def build_base_config_from_args(args):
    """Build the base configuration from the command line argument"""
    
    ## Set dataset
    configuration = {}
    configuration['network'] = args.network
    if args.data == 'vedai':
        configuration['setting'] = 'vedai'
        configuration['exp_name'] = 'vedai'
        configuration['save_summaries_steps'] = 100
        configuration['save_evaluation_steps'] = 250 if args.save_evaluation_steps is None else args.save_evaluation_steps
        configuration['num_epochs'] = 1000 if args.num_epochs is None else args.num_epochs
    elif args.data == 'stanford':
        configuration['setting'] = 'sdd'
        configuration['exp_name'] = 'sdd'
        configuration['save_summaries_steps'] = 200 
        configuration['save_evaluation_steps'] = 500 if args.save_evaluation_steps is None else args.save_evaluation_steps
        configuration['num_epochs'] = 120 if args.num_epochs is None else args.num_epochs
    elif args.data == 'dota':
        configuration['setting'] = 'dota'
        configuration['exp_name'] = 'dota'
        configuration['save_summaries_steps'] = 500
        configuration['save_evaluation_steps'] = 1000 if args.save_evaluation_steps is None else args.save_evaluation_steps
        configuration['num_epochs'] = 100 if args.num_epochs is None else args.num_epochs

    ## Metadata
    tfrecords_path = 'Data/metadata_%s.txt'
    metadata = load_metadata(tfrecords_path % configuration['setting'])
    configuration.update(metadata)
    configuration['num_classes'] = len(configuration['data_classes'])

    ## GPUs
    configuration['num_gpus'] = args.num_gpus                                 
    configuration['gpu_mem_frac'] = max(0., min(1., args.gpu_mem_frac))

    ## Training
    configuration['batch_size'] = args.batch_size
    configuration['learning_rate'] = args.learning_rate
    
    ## Return
    return configuration