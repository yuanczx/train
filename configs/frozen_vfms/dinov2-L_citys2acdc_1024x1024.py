# dataset config
_base_ = [
    "../_base_/datasets/dg_citys2acdc_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/models/dinov2_mask2former_1024x1024.py",
]
model = dict(type="FrozenBackboneEncoderDecoder")

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=4096,
    ),
    dict(type="RandomCrop", crop_size={{_base_.crop_size}}, cat_max_ratio=0.75), # type: ignore
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(batch_size=8, num_workers=1, dataset=dict(pipeline=train_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "query_embed": embed_multi,
            "query_feat": embed_multi,
            "level_embed": embed_multi,
            "norm": dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=10000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=5000, max_keep_ckpts=8
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
work_dir="./work_dirs/frozen_dinov2-citys2acdc_1024x1024"