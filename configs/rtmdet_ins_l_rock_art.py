# ========================================================
#  Config: RTMDet-Ins-L (Large Instance Segmentation)
#
#  【配置说明】
#  这是一个完全展开的独立配置文件，无需 _base_ 依赖。
#  包含了 RTMDet-Ins-L 的所有核心组件：CSPNeXt 骨干、PAFPN、两阶段增强策略。
#
#  【关键功能】
#  1. save_best: 自动保存验证集 mAP 最高的权重。
#  2. 80 Epochs: 适配微调任务 (前 70 轮强增强，后 10 轮弱增强)。
#  3. AdamW + CosineLR: RTMDet 标准优化策略。
# ========================================================

# --------------------------------------------------------
# 1. 基础环境与钩子配置 (Runtime)
# --------------------------------------------------------
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 【核心功能】保存最优权重配置
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,                 # 每 5 轮保存一次常规权重
        max_keep_ckpts=3,           # 最多保留 3 个常规权重，节省空间
        save_best='coco/bbox_mAP',  # 监控指标：bbox mAP (也可以设为 'coco/segm_mAP')
        rule='greater'              # 规则：越大越好
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, show=False)
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer',alpha=0.3)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'

# 【预训练权重】RTMDet-Ins-L 官方 COCO 权重
load_from = 'checkpoints/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'
resume = False

# --------------------------------------------------------
# 2. 类别与数据集定义
# --------------------------------------------------------
class_name = (
    'Sheep_Goat',       # ID 1
    'Anthropomorph',    # ID 2
    'Symbol',           # ID 3
    'Cattle_Bovine',    # ID 4
    'Deer_Cervid',      # ID 5
    'Horse_Equid',      # ID 6
    'Face_Mask',        # ID 7
    'Rider',            # ID 8
    'Camel',            # ID 9
    'Utensil_Instrument', # ID 10
    'Feline'            # ID 11
)
num_classes = 11
metainfo = dict(classes=class_name, palette=[(220, 20, 60)] * num_classes)

# --------------------------------------------------------
# 3. 模型结构定义 (RTMDet-Ins-L)
# --------------------------------------------------------
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=load_from)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetInsHead',
        num_classes=num_classes, # 修改为 11
        in_channels=256,
        stacked_convs=2,
        pred_kernel_size=1,
        feat_channels=256,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN'),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-06, reduction='mean')),
    train_cfg=dict(
        assigner=dict(
            type='DynamicSoftLabelAssigner',
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5),
)

# --------------------------------------------------------
# 4. 数据增强流水线 (Pipelines)
# --------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    # RTMDet 特有的缓存 Mosaic
    dict(type='CachedMosaic', img_scale=(1024, 1024), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(2048, 2048),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    # 强增强混合 MixUp
    dict(
        type='CachedMixUp',
        img_scale=(1024, 1024),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

# 第二阶段弱增强 (训练末期切换)
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# --------------------------------------------------------
# 5. Dataloader 配置
# --------------------------------------------------------
train_dataloader = dict(
    batch_size=4,  # RTMDet 显存占用较大，建议从 4 开始，如果显存够大可改为 8
    num_workers=4, # 建议设为 2 或 4
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=True),
    sampler=dict(type='ClassAwareSampler'),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='data/rock_art/',
        ann_file='annotations/train_11.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None,
        metainfo=metainfo)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/rock_art/',
        ann_file='annotations/val_11.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
        metainfo=metainfo)
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/rock_art/annotations/val_11.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None,
    classwise=True,
outfile_prefix='./work_dirs/rtmdet_ins_results'
)
test_evaluator = val_evaluator

# --------------------------------------------------------
# 6. 训练循环与优化器策略
# --------------------------------------------------------
max_epochs = 80         # 总轮数
stage2_num_epochs = 10  # 最后10轮切换弱增强
base_lr = 0.0005        # 学习率 (Batch=4时建议0.0005，Batch=8时建议0.001)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 学习率策略 (Cosine Annealing)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 优化器 (AdamW)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 自定义钩子 (用于在最后阶段切换 Pipeline)
custom_hooks = [
    dict(
        type='NumClassCheckHook'),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

auto_scale_lr = dict(enable=False, base_batch_size=16)