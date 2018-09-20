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
    "train_patch_confidence_threshold": 0.5,               # Only keep boxes above this threshold for patch extraction
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