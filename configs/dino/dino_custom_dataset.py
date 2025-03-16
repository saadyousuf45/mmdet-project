_base_ = './dino-5scale_swin-l_8xb2-12e_coco.py'

# Modify model with correct Swin-Large backbone and pretrained weights
model = dict(
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,  # Swin-Large uses 192 dimensions
        depths=[2, 2, 18, 2],  # Swin-Large architecture
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        drop_path_rate=0.2,
        patch_norm=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./checkpoints/swin_large_patch4_window7_224_22kto1k.pth',
            prefix='backbone'
        )
    ),
    bbox_head=dict(num_classes=162)  # Update number of classes
)

# Dataset configuration
data_root = '/media/saadi/Drive/Thesis_Folder/Thesis_2022/mmdetection/data/Age_Gender_Attire/'
metainfo = dict(classes=open('predefined_classes.txt').read().splitlines())

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train_annotations.json',
        data_prefix=dict(img='Image/'),
        metainfo=metainfo
    )
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader

val_evaluator = dict(ann_file=data_root + 'train_annotations.json')
test_evaluator = val_evaluator

# Training schedule (customize if needed)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

