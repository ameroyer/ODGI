defaults_dict = {
    "assignment_reward_fn": "iou",                         # function for assigning gt to best predictor
    "base_log_dir": "./log",                               # Base log directory
    "beta1": 0.9,                                          # If using Momentum optimizer
    "base_name": "net",
    "batch_size": 16,                                      # training batch size
    "centers_localization_loss_weight": 1.0,               
    "classification_loss_weight": 1.0,
    "confidence_loss_weight": 5.0,
    "data_augmentation_threshold": 0.5,                    # Data augmentation (flip lr) ratio
    "exp_name": "exp",
    "full_image_size": 1024,                               # image size to load for extracting the crops
    "gpu_mem_frac": 1.0,                                   # memory usage per gpu
    "group_classification_loss_weight": 1.,
    "image_size": 256,                                     # Image size
    "learning_rate": 1e-3,
    "log_globalstep_steps": 100,
    "lr_decay_rate": 0.995,                                # If using Momentum optimizer
    "lr_decay_steps": 1,                                   # If using Momentum optimizer
    "max_to_keep": 1,                                      # maximum number of checkpoints to keep
    "momentum": 0.9,                                       # If using momentum optimizer
    "network": 'tiny-yolov2',                              # One of tiny-yolov2 or yolov2
    "noobj_confidence_loss_weight": 1.0,
    "normalizer_decay": 0.9,                               # Batch norm decay
    "num_boxes": 1,                                        # Number of boxes per cell to predict
    "num_epochs": 100,                                     # Number of training epochs
    "num_gpus": 1,                                         # Number of gpus to use
    "num_summaries": 3,                                    # Number of summaries to display
    "num_threads": 8,                                      # Number of parallel readers for the input queues
    "offsets_loss_weight": 1.0,
    "offsets_margin": 0.025,                                # Margins to learn the additional offsets
    "optimizer": "ADAM",                                   # 'ADAM' or 'MOMENTUM'
    "patch_intersection_ratio_threshold": 0.3,             # Only keep gt box in the pach if at least this ratio is visible
    "patch_nms_threshold": 0.25,                           # IoU threshold for non-maximum suppression during patch extraction
    "prefetch_capacity": 1,                                # prefetch capacity for the main dataset object
    "restore_replace_to": '',                              # string to replace to in variable names to restore
    "restore_scope": None,                                 # If a checkpoint path is given, restore from the given scope
    "restore_to_replace": '',                              # string to replace in variable names to restore
    "retrieval_confidence_threshold": 0.,                  # Only keep boxes above this threshold for evaluation
    "retrieval_intersection_threshold": [0.25, 0.5, 0.75], # Evaluate at these retrieval threshold
    "retrieval_nms_threshold": 0.5,                        # IoU threshold for the Non-maximum suppression during evaluation
    "retrieval_top_n": 1024,                               # Look-up at most these number of top predictions for evaluation
    "save_checkpoint_secs": 3600,                          # Save checkpoints at the given frequency (in seconds)
    "save_evaluation_steps": 1000,
    "scales_localization_loss_weight": 1.0,
    "shuffle_buffer": 1,                                   # Shuffle buffer size in the main dataset object
    "subset": -1,                                          # If > 0, select a subset of the dataset
    "summary_confidence_thresholds": [0.5],                # Plot output boxes cut at the given thresholds  
    "target_conf_fn": "iou",                               # Function for determining the confidence scores ground-truth 
    "test_num_crops": 5,                                    # Maximum number of crops per image to predict (val)
    "test_patch_confidence_threshold": 0.25,                 # Only keep boxes above this threshold for patch extraction
    "test_patch_strong_confidence_threshold": 0.75,         # boxes considered 'single' and above this threshold -> no patch
    "train_num_crops": 5,                                  # Maximum number of crops per image to predict (train)
    "train_patch_confidence_threshold": 0.15,               # Only keep boxes above this threshold for patch extraction
    "weight_decay": 0.,
    "with_classification": False,                          # If True, output class scores 
    "with_groups": False,                                   # if False, do not compute group_bounding_boxes
    "with_group_flags": False,
    "with_offsets": False                                  # If True, output offsets to match the gt box + offsets_margin
}