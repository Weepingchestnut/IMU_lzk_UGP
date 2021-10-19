_base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    backbone=dict(
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        multi_grid=(1, 2, 4)
    ),
    decode_head=dict(
        channels=512,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=256,
        sampler=dict(type='OHEMPixelSampler', min_kept=100000)
    )
)
data = dict(
    samples_per_gpu=8
)
