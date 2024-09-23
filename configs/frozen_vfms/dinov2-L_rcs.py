_base_ = [
    "./dinov2-L_mask2former.py"
]

train_dataloader=dict(
    dataset=dict(
        type="RCSDataset",
        rare_class_sampling=dict(min_pixels=3000, class_temp=100, min_crop_ratio=0.5)
        )
    )
