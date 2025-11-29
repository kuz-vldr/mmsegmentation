dataset_type = 'CatDogDataset'

data_root = 'pract_work/dataset/train_dataset_for_students'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img/train',
            seg_map_path='labels/new_train'
        ),
        pipeline=train_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png'
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img/val',
            seg_map_path='labels/val'
        ),
        pipeline=test_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png'
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
       type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img/test',
            seg_map_path='labels/test'
        ),
        pipeline=test_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png'
    )
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])