cfg = dict(
    model='pvt_small',  # You can change to pvt_tiny, pvt_medium, or pvt_large
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/food_classification',
    
    # Training parameters
    batch_size=32,  # Adjust based on your GPU memory
    epochs=100,     # Number of training epochs
    
    # Dataset parameters
    data_set='FOOD',
    data_path='/content/food_dataset',  # Your dataset path
    input_size=224,
    
    # Optimizer parameters
    opt='adamw',
    lr=5e-4,
    weight_decay=0.05,
    
    # Augmentation parameters
    color_jitter=0.4,
    aa='rand-m9-mstd0.5-inc1',
    reprob=0.25,
    remode='pixel',
    recount=1,
    train_interpolation='bicubic',
    
    # Mixup/Cutmix parameters
    mixup=0.8,
    cutmix=1.0,
    mixup_prob=1.0,
    mixup_switch_prob=0.5,
    mixup_mode='batch',
    smoothing=0.1,
    
    # Other parameters
    repeated_aug=True,
    num_workers=2,  # Adjust based on your CPU cores
    pin_mem=True,
    
    # Learning rate schedule
    sched='cosine',
    warmup_epochs=5,
    min_lr=1e-5,
    decay_epochs=30,
    warmup_lr=1e-6,
)