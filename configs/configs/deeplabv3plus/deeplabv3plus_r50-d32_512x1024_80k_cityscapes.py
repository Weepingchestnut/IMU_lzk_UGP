_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        depth=50,
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2)
    ),
    decode_head=dict(
        channels=512,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=256
    )
)

data = dict(
    samples_per_gpu=8
)
