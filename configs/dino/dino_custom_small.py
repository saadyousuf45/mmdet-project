_base_ = './dino-5scale_swin-l_8xb2-12e_coco.py'

# Modify model with Swin-Small backbone
model = dict(
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,  # Swin-Small uses 96 embedding dimensions
        depths=[2, 2, 18, 2],  # Swin-Small structure
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        drop_path_rate=0.2,
        patch_norm=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./checkpoints/swin_small_patch4_window7_224.pth',
            prefix='backbone'
        )
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],  # ✅ Set as a list to match Swin-Small's output
        kernel_size=1,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5
    ),
    bbox_head=dict(num_classes=162)  # Match number of classes
)

# Dataset configuration
data_root = '/media/saadi/Drive/Thesis_Folder/Thesis_2022/mmdetection/data/Age_Gender_Attire/'
metainfo = dict(classes=open('predefined_classes.txt').read().splitlines())

# Reduce GPU memory usage
train_dataloader = dict(
    batch_size=1,  # Reduce batch size
    num_workers=1,  # Fewer workers save memory
    dataset=dict(
        data_root=data_root,
        ann_file='train_annotations.json',
        data_prefix=dict(img='Image/'),
        metainfo=metainfo
    )
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader

# Reduce memory usage
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),  # Reduce image size
    dict(type='PackDetInputs')
]

# Use mixed precision (FP16)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)  


# Gradient accumulation (simulates larger batch size)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    accumulative_counts=2  # Simulate batch_size=4 without using more memory
)

# Evaluators - Compute mAP after every epoch
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'train_annotations.json',
    metric='bbox', # Computes mAP (AP@IoU=0.50:0.95)
    classwise=True, 
    format_only=False
)

test_evaluator = val_evaluator

# Hooks for logging, checkpoint saving, and validation
default_hooks = dict(
    checkpoint=dict(interval=1),  # Save checkpoint every epoch
    logger=dict(type='LoggerHook', interval=50)  # Log every 50 iterations
)

# Use ValLoop for evaluation (correct format)
val_cfg = dict(type='ValLoop')

