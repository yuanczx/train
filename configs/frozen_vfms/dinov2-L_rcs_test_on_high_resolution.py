_base_ = [
    "./dinov2-L_mask2former.py"
]
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root='data/cityscapes/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        2048,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='CityscapesDataset'),
            dict(
                data_prefix=dict(
                    img_path='images/10k/val',
                    seg_map_path='labels/sem_seg/masks/val'),
                data_root='data/bdd100k/',
                img_suffix='.jpg',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1280,
                        720,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='.png',
                type='CityscapesDataset'),
            dict(
                data_prefix=dict(
                    img_path='half/val_img', seg_map_path='half/val_label'),
                data_root='data/mapillary/',
                img_suffix='.jpg',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        2048,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='.png',
                type='CityscapesDataset'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataloader = test_dataloader
train_dataloader=dict(
    dataset=dict(
        type="RCSDataset",
        rare_class_sampling=dict(min_pixels=3000, class_temp=100, min_crop_ratio=0.5)
        )
    )
