import os

_dinov3_pretrained = os.environ.get(
    'DINOV3_PRETRAINED_WEIGHTS',
    'checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')

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
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale':
            [(480, 2400), (512, 2400), (544, 2400), (576, 2400), (608, 2400),
             (640, 2400), (672, 2400), (704, 2400), (736, 2400), (768, 2400),
             (800, 2400), (832, 2400), (864, 2400), (896, 2400), (928, 2400),
             (960, 2400), (992, 2400), (1024, 2400), (1056, 2400),
             (1088, 2400), (1120, 2400), (1152, 2400), (1184, 2400),
             (1216, 2400), (1248, 2400), (1280, 2400), (1312, 2400),
             (1344, 2400), (1376, 2400), (1408, 2400), (1440, 2400),
             (1472, 2400), (1504, 2400), (1536, 2400)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (384, 600),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(480, 2400), (512, 2400), (544, 2400),
                                    (576, 2400), (608, 2400), (640, 2400),
                                    (672, 2400), (704, 2400), (736, 2400),
                                    (768, 2400), (800, 2400), (832, 2400),
                                    (864, 2400), (896, 2400), (928, 2400),
                                    (960, 2400), (992, 2400), (1024, 2400),
                                    (1056, 2400), (1088, 2400), (1120, 2400),
                                    (1152, 2400), (1184, 2400), (1216, 2400),
                                    (1248, 2400), (1280, 2400), (1312, 2400),
                                    (1344, 2400), (1376, 2400), (1408, 2400),
                                    (1440, 2400), (1472, 2400), (1504, 2400),
                                    (1536, 2400)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
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
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=('foreground', ),
        ann_file=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_train_cc.json',
        img_prefix=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='UnifiedRandomRotate',
                prob=0.2,
                angle_range=(-180, 180),
                expand=True,
                pad_val=128),
            dict(type='UnifiedPhotometricAug'),
            dict(
                type='UnifiedLetterboxResize',
                target_size=(720, 1280),
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
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        filter_empty_gt=False),
    val=dict(
        type='CocoDataset',
        classes=('foreground', ),
        ann_file=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_val_cc.json',
        img_prefix=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 720),
                flip=False,
                transforms=[
                    dict(
                        type='UnifiedLetterboxResize',
                        target_size=(720, 1280),
                        pad_val=128),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('foreground', ),
        ann_file=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_val_cc.json',
        img_prefix=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 720),
                flip=False,
                transforms=[
                    dict(
                        type='UnifiedLetterboxResize',
                        target_size=(720, 1280),
                        pad_val=128),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    train_eval=dict(
        type='CocoDataset',
        classes=('foreground', ),
        ann_file=
        'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/annotations_train_eval_500_seed42.json',
        img_prefix=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 720),
                flip=False,
                transforms=[
                    dict(
                        type='UnifiedLetterboxResize',
                        target_size=(720, 1280),
                        pad_val=128),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    train_eval_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        persistent_workers=False))
evaluation = dict(
    interval=2424,
    metric=['bbox', 'segm'],
    recall_iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    recall_max_dets=100,
    by_epoch=False)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='codino-dinov3-smoke',
                dir=
                'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019',
                mode='online',
                config=dict(
                    data_root=
                    'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
                    train_json=
                    'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_train_cc.json',
                    val_json=
                    'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_val_cc.json',
                    dino_weights=_dinov3_pretrained,
                    epochs=6,
                    max_iters=None,
                    samples_per_gpu=8,
                    workers_per_gpu=2,
                    num_query=100,
                    num_dn_queries=50,
                    amp='fp16',
                    train_height=720,
                    train_width=1280,
                    lr=1e-05,
                    lr_policy='cosine',
                    lr_by_iter=True,
                    eval_iter_interval=2424,
                    load_from=
                    'checkpoints/codino/detector/pretrain_epoch_10_reference.pth',
                    resume_from=None,
                    clean_eval_root=
                    'data/0423_eval_sod_clean_sdjs362_sdam161_cc_v1',
                    clean_eval_json=
                    'data/0423_eval_sod_clean_sdjs362_sdam161_cc_v1/annotations_eval_cc.json',
                    data_preflight_report=
                    'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/data_preflight_report.json',
                    freeze_backbone=False,
                    head_lr_mult=2.0,
                    layer_decay_rate=0.8,
                    weight_decay=0.01,
                    wandb_val_overlay_count=6,
                    wandb_val_overlay_score_thr=0.3,
                    train_eval_count=500,
                    train_eval_interval=1,
                    recall_iou_thrs=[
                        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
                    ],
                    recall_max_dets=100,
                    seed=42),
                name=
                'dinov3-codino-inst-0423-lrrestart-from0414ep10-b8-bb1e5-head2e5-cosine-freq4-20260514_112019',
                tags=[
                    'full0423', 'lr-restart', 'from0414ep10', 'b8', 'bb1e-5',
                    'head2e-5', 'cosine', 'freq4', 'clean-preflight'
                ]),
            interval=50,
            log_artifact=True,
            out_suffix=('.log.json', '.log', '.py', '.json'),
            define_metric_cfg=dict({
                'loss': 'min',
                'bbox_mAP': 'max',
                'segm_mAP': 'max',
                'train/bbox_mAP': 'max',
                'train/segm_mAP': 'max',
                'val/bbox_mAP': 'max',
                'val/segm_mAP': 'max'
            }))
    ])
custom_hooks = [
    dict(type='LrGroupLoggerHook', priority=60, interval=50),
    dict(
        type='WandbDebugArtifactsHook',
        priority='LOWEST',
        artifact_name='codino-debug-inputs',
        aliases=['latest', 'seed42'],
        metadata=dict(
            samples_per_gpu=8,
            epochs=6,
            num_query=100,
            freeze_backbone=False,
            head_lr_mult=2.0,
            train_json=
            'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_train_cc.json'
        ),
        files=[
            'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/resolved_config.py',
            'training_reference/train_dinov3_codino_inst_0423_clean_freq4.py',
            'training_reference/unified_letterbox_aug.py',
            'training_reference/make_codino_train20k_subset.py',
            'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_train_cc.json',
            'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/annotations_train_eval_500_seed42.json',
            'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_val_cc.json',
            'data/0423_eval_sod_clean_sdjs362_sdam161_cc_v1/annotations_eval_cc.json',
            'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/dataset_stats.json',
            'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/run_metadata.json',
            'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/data_preflight_report.json'
        ]),
    dict(
        type='WandbMetricMirrorHook',
        priority=80,
        interval=2424,
        by_epoch=False),
    dict(
        type='WandbValOverlayHook',
        priority='VERY_LOW',
        val_json=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_val_cc.json',
        data_root=
        'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
        num_images=6,
        score_thr=0.3,
        interval=1,
        seed=42),
    dict(
        type='TrainSubsetEvalHook',
        priority=70,
        dataset_cfg=dict(
            type='CocoDataset',
            classes=('foreground', ),
            ann_file=
            'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1/annotations_val_cc.json',
            img_prefix=
            'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(1280, 720),
                    flip=False,
                    transforms=[
                        dict(
                            type='UnifiedLetterboxResize',
                            target_size=(720, 1280),
                            pad_val=128),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]),
        dataloader_cfg=dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=False,
            shuffle=False,
            persistent_workers=False),
        eval_cfg=dict(
            interval=2424,
            metric=['bbox', 'segm'],
            recall_iou_thrs=[
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
            ],
            recall_max_dets=100,
            by_epoch=False),
        interval=2424,
        by_epoch=False,
        prefix='',
        raw_metrics=True),
    dict(
        type='TrainSubsetEvalHook',
        priority=75,
        dataset_cfg=dict(
            type='CocoDataset',
            classes=('foreground', ),
            ann_file=
            'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019/annotations_train_eval_500_seed42.json',
            img_prefix=
            'data/0423_train_old_eval_plus_new_nonrecommended_cc_v1',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(1280, 720),
                    flip=False,
                    transforms=[
                        dict(
                            type='UnifiedLetterboxResize',
                            target_size=(720, 1280),
                            pad_val=128),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]),
        dataloader_cfg=dict(
            samples_per_gpu=2,
            workers_per_gpu=2,
            dist=False,
            shuffle=False,
            persistent_workers=False),
        eval_cfg=dict(
            interval=2424,
            metric=['bbox', 'segm'],
            recall_iou_thrs=[
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
            ],
            recall_max_dets=100,
            by_epoch=False),
        interval=2424,
        by_epoch=False,
        prefix='train')
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/codino/detector/pretrain_epoch_10_reference.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
pretrained = None
num_dec_layer = 6
lambda_2 = 2.0
model = dict(
    type='CoDETR',
    backbone=dict(
        type='DINOv3ViT',
        img_size=1536,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layers_to_use=1,
        pretrained=True,
        weights=_dinov3_pretrained
    ),
    neck=dict(
        type='SFP',
        in_channels=[1024],
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0),
        loss_bbox=dict(type='L1Loss', loss_weight=12.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32, 64],
        finest_scale=56),
    mask_head=dict(
        type='SimpleRefineMaskHead',
        num_convs_instance=1,
        num_convs_semantic=2,
        conv_in_channels_instance=256,
        conv_in_channels_semantic=256,
        conv_kernel_size_instance=3,
        conv_kernel_size_semantic=3,
        conv_out_channels_instance=256,
        conv_out_channels_semantic=256,
        conv_cfg=None,
        norm_cfg=dict(type='LN2d'),
        fusion_type='MultiBranchFusionAvg',
        dilations=[1, 3, 5],
        semantic_out_stride=4,
        stage_num_classes=[1, 1, 1, 1],
        stage_sup_size=[14, 28, 56, 112],
        pre_upsample_last_stage=False,
        upsample_cfg=dict(type='bilinear', scale_factor=2),
        loss_weight=15.96,
        loss_cfg=dict(
            type='BARCrossEntropyLoss',
            stage_instance_loss_weight=[0.5, 0.75, 0.75, 1.0],
            boundary_width=2,
            start_stage=1)),
    mask_iou_head=dict(
        type='MaskIoUHead',
        num_convs=2,
        num_fcs=1,
        roi_feat_size=14,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        num_classes=1,
        score_use_sigmoid=True,
        norm_cfg=dict(type='LN2d'),
        loss_iou=dict(type='MSELoss', loss_weight=6.0)),
    query_head=dict(
        type='CoDINOHead',
        num_query=100,
        num_classes=1,
        num_feature_levels=5,
        in_channels=2048,
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
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
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
                            num_levels=5,
                            dropout=0.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
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
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='GN', num_groups=32),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=12.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=120.0)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=1,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0))
    ],
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
            debug=False),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
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
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        dict(
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.8),
            mask_thr_binary=0.5),
        dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.5),
            mask_thr_binary=0.5,
            max_per_img=1000),
        dict(
            rpn=dict(
                nms_pre=8000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.9),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                mask_thr_binary=0.5,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=1000)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='soft_nms', iou_threshold=0.6),
            max_per_img=100)
    ])
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    min_lr=1e-07,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=6)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer = dict(
    type='AdamW',
    lr=1e-05,
    weight_decay=0.01,
    constructor='DINOv3HeadBoostLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.8, head_lr_mult=2.0))
work_dir = 'output/training_reference/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019'
seed = 42
gpu_ids = range(0, 1)
device = 'cuda'
fp16 = dict(loss_scale='dynamic')
letterbox_size = (720, 1280)
