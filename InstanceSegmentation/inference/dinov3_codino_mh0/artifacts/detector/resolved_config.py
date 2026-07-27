dataset_type = 'CocoDataset'
classes = ('foreground', )
data_root = 'data/dataset1027_tentative_combined/'
ann_root = 'data/dataset1027_tentative_combined_processed/annotations/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='UnifiedPhotometricAug'),
    dict(type='UnifiedLetterboxResize', target_size=(736, 1280), pad_val=128),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'letterbox_meta'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=28,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        classes=('foreground', ),
        ann_file=
        '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/distillation_workspace_20260724/workspaces/06_video_pseudo_data/results/teacher_pseudo_v1/dataset/annotations_train_gt_plus_video_pseudo.json',
        img_prefix=
        '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/distillation_workspace_20260724/workspaces/06_video_pseudo_data/results/teacher_pseudo_v1/dataset/images_root',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='UnifiedPhotometricAug'),
            dict(
                type='UnifiedLetterboxResize',
                target_size=(736, 1280),
                pad_val=128),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'letterbox_meta'))
        ],
        filter_empty_gt=False),
    val=dict(
        type='CocoDataset',
        classes=('foreground', ),
        ann_file=
        '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/distillation_workspace_20260724/workspaces/06_video_pseudo_data/results/teacher_pseudo_v1/dataset/annotations_val_gt_prefixed.json',
        img_prefix=
        '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/distillation_workspace_20260724/workspaces/06_video_pseudo_data/results/teacher_pseudo_v1/dataset/images_root',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 736),
                flip=False,
                transforms=[
                    dict(
                        type='UnifiedLetterboxResize',
                        target_size=(736, 1280),
                        pad_val=128),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'letterbox_meta'))
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('foreground', ),
        ann_file=
        '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/distillation_workspace_20260724/workspaces/06_video_pseudo_data/results/teacher_pseudo_v1/dataset/annotations_test_gt_prefixed.json',
        img_prefix=
        '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/distillation_workspace_20260724/workspaces/06_video_pseudo_data/results/teacher_pseudo_v1/dataset/images_root',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 736),
                flip=False,
                transforms=[
                    dict(
                        type='UnifiedLetterboxResize',
                        target_size=(736, 1280),
                        pad_val=128),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'letterbox_meta'))
                ])
        ]),
    train_dataloader=dict(pin_memory=True, prefetch_factor=4))
evaluation = dict(
    interval=1,
    metric=['bbox', 'segm'],
    recall_iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    recall_max_dets=100,
    save_best='segm_mAP',
    rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        momentum=0.0001,
        resume_from=None,
        priority=49),
    dict(type='StopAfterEpochHook', stop_after_epoch=10, priority='LOWEST'),
    dict(
        type='TorchCompileHook',
        priority='HIGH',
        compile_modules='backbone,neck',
        compile_mode='reduce-overhead',
        compile_fullgraph=False,
        compile_dynamic=False,
        compile_cudagraphs=False),
    dict(
        type='EpochXidRecoveryCheckpointHook',
        priority='NORMAL',
        interval=500,
        max_keep=2,
        prefix='recovery_iter'),
    dict(type='LrGroupLoggerHook', priority=60, interval=50)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
pretrained = None
num_dec_layer = 6
lambda_2 = 2.0
model = dict(
    type='TrainableSCBalancedCoDETR',
    backbone=dict(
        type='DINOv3ViTSPlus',
        weights=
        '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        pretrained=True,
        layers_to_use=1,
        embed_dim=384,
        patch_size=16),
    neck=dict(
        type='SFP',
        in_channels=[384],
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
    rpn_head=None,
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        out_channels=192,
        featmap_strides=[4, 8, 16, 32],
        finest_scale=56),
    mask_head=dict(
        type='TrainableExplicit112SimpleRefineMaskHead',
        num_convs_instance=1,
        num_convs_semantic=2,
        conv_in_channels_instance=192,
        conv_in_channels_semantic=192,
        conv_kernel_size_instance=3,
        conv_kernel_size_semantic=3,
        conv_out_channels_instance=192,
        conv_out_channels_semantic=192,
        conv_cfg=None,
        norm_cfg=dict(type='LN2d'),
        fusion_type='MultiBranchFusionAvg',
        dilations=[1, 3, 5],
        semantic_out_stride=4,
        stage_num_classes=[1, 1, 1],
        stage_sup_size=[14, 28, 56],
        pre_upsample_last_stage=False,
        upsample_cfg=dict(type='bilinear', scale_factor=2),
        loss_weight=15.96,
        loss_cfg=dict(
            type='BARCrossEntropyLoss',
            stage_instance_loss_weight=[
                0.3333333333333333, 0.6666666666666666, 1.0
            ],
            boundary_width=2,
            start_stage=1)),
    mask_iou_head=dict(
        type='MaskIoUHead',
        num_convs=2,
        num_fcs=1,
        roi_feat_size=14,
        in_channels=192,
        conv_out_channels=192,
        fc_out_channels=768,
        num_classes=1,
        score_use_sigmoid=True,
        norm_cfg=dict(type='LN2d'),
        loss_iou=dict(type='MSELoss', loss_weight=6.0)),
    query_head=dict(
        type='CoDINOHead',
        num_query=100,
        num_classes=1,
        num_feature_levels=3,
        in_channels=256,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        mixed_selection=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=50)),
        transformer=dict(
            type='CoDinoTransformer',
            with_pos_coord=True,
            with_coord_feat=False,
            num_co_heads=0,
            num_feature_levels=3,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=1,
                with_cp=0,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=3,
                        dropout=0.0,
                        num_points=4),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=3,
                            dropout=0.0,
                            num_points=4)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            two_stage_num_proposals=100),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[],
    bbox_head=[],
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_thr_binary=0.5,
            mask_size=28,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        dict(
            max_per_img=100,
            nms=dict(type='soft_nms', iou_threshold=0.8),
            mask_thr_binary=0.5),
        dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.5),
            mask_thr_binary=0.5,
            max_per_img=1000)
    ],
    query_level_indices=(1, 2, 3),
    neck_level_channels=(128, 256, 256, 256, 256),
    query_channels=256,
    mask_channels=192)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=143,
    warmup_ratio=0.01,
    min_lr=1e-07,
    by_epoch=False)
runner = dict(type='RecoverableEpochBasedRunner', max_epochs=10)
optimizer_config = dict(
    grad_clip=dict(max_norm=0.1, norm_type=2), grad_norm_log_interval=0)
optimizer = dict(
    type='AdamW',
    lr=9e-05,
    weight_decay=0.01,
    constructor='DINOv3HeadBoostLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=1.0, head_lr_mult=1.5))
work_dir = '/home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/runs/codino_inst/mh0_dino_norot_video_pseudo_cos10_b28_20260724'
seed = 42
gpu_ids = range(0, 2)
device = 'cuda'
codino_training_mode = 'dino_only'
fp16 = dict(loss_scale='dynamic')
letterbox_size = (736, 1280)
matrix_runtime = dict(
    bypass_p2=True,
    encoder_levels=('P3', 'P4', 'P5'),
    encoded_levels=('P3', 'P4', 'P5'),
    p2_lite=False,
    neck_channels=256,
    neck_level_channels=[128, 256, 256, 256, 256],
    query_channels=256,
    mask_channels=192)
architecture_name = 'SC-BALANCED+MH0-DINO-ONLY+NO-ROTATION'
experiment_contract = dict(
    initialization=
    'DINOv3 pretrained backbone with randomly initialized task heads',
    training='epochs 0-3 within a ten-epoch cosine schedule',
    auxiliary_heads=False,
    rotation_augmentation=False,
    control='ordinary DINO-only losses')
custom_imports = dict(
    imports=['mmdet.datasets.pipelines.unified_letterbox_aug'],
    allow_failed_imports=False)
