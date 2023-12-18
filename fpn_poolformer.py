norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth'
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='mmcls.PoolFormer',
        arch='s12',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth',
            prefix='backbone.'),
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        out_indices=(0, 2, 4, 6),
        frozen_stages=0),
    
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.6),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.4)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
log_config = dict(
    interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/root/mmsegmentation-0.29.0/tools/work_dirs/poolformer_upernet_aux-fapn/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=400)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
ataset_type = 'PascalVOCDataset'
data_root = '/home/sys004/segmentation'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (480, 480)
crop_size = (480, 480)
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=1,
    train=dict(
        type='PascalVOCDataset',
        data_root='/root/dataset',
        img_dir='image',
        ann_dir='label',
        split='/root/dataset/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(480, 480), ratio_range=(1.0, 1.0)),
            dict(type='RandomFlip', prob=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='PascalVOCDataset',
        data_root='/root/mmsegmentation-0.29.0/data',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassPNG',
        split='/root/mmsegmentation-0.29.0/data/ImageSets/Segmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 480),
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        img_scale=(480, 480),
                        ratio_range=(1.0, 1.0)),
                    dict(type='RandomFlip', prob=0.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='PascalVOCDataset',
        data_root='/root/mmsegmentation-0.29.0/data',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassPNG',
        split=
        '/root/mmsegmentation-0.29.0/data/ImageSets/Segmentation/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 480),
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        img_scale=(336, 496),
                        ratio_range=(1.0, 1.0)),
                    dict(type='RandomFlip', prob=0.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
work_dir = './work_dirs/2_fpn_poolformer_s12_8x4_512x512_40k_ade20k'
gpu_ids = [0]
auto_resume = False
