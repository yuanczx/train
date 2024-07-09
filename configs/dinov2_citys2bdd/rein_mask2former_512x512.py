_base_ = [
    "../_base_/datasets/bdd100k_512x512.py",
    "../_base_/datasets/cityscapes_512x512.py",
    "../_base_/models/rein_dinov2_mask2former.py",
    "../_base_/schedules/schedule_40k.py",
    "../_base_/default_runtime.py"
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_cityscapes}},
) 

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset={{_base_.val_bdd}},
)

test_dataloader = val_dataloader

val_evaluator = dict(
    dataset_keys=['bdd'],
    iou_metrics=['mIoU'],
    type='DGIoUMetric'
)

test_evaluator = val_evaluator

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]

work_dir = './work_dirs/rein_citys2bdd'