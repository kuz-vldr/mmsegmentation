# pract_work/configs/baseline_unet.py

_base_ = ['../../configs/_base_/models/fcn_unet_s5-d16.py']

data_root = 'dataset/train_dataset_for_students'
crop_size = (512, 512)

model = dict(
    backbone=dict(in_channels=3),
    decode_head=dict(num_classes=3),
    test_cfg=dict(mode='whole')
)

train_dataloader = dict(
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img/train', seg_map_path='labels/new_train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomCrop', crop_size=crop_size),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs')
        ],
        reduce_zero_label=False
    )
)

val_dataloader = dict(
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ],
        reduce_zero_label=False
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
# --- Оптимизатор ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# --- Scheduler ---
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-4, begin=0, end=40000, by_epoch=False)
]

# --- Циклы ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')


work_dir = './work_dirs/baseline_unet'