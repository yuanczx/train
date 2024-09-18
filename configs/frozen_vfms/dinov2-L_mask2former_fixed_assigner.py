_base_=["./dinov2-L_mask2former.py"]

model=dict(
    decode_head=dict(
        train_cfg=dict(
            assigner=dict(
                _delete_=True,
                type="IdentityAssigner",
                num_cls=19
                )
            )
        )
    )