"""Inference-only EVA02 ViT-L Cascade Mask R-CNN LazyConfig.

This is the model portion of the validated training configuration. Dataset,
optimizer, and trainer declarations are deliberately absent because they do
not belong to the inference-family replacement unit.
"""

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    CascadeROIHeads,
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
)


model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# Cascade R-CNN replaces the single-stage box head owned by the base config.
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]
model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(
            thresholds=[th],
            labels=[0, 1],
            allow_low_quality_matches=False,
        )
        for th in [0.5, 0.6, 0.7]
    ],
)

# Exact inference-relevant EVA02 ViT-L 1280 settings.
model.backbone.net.img_size = 1280
model.backbone.square_pad = 1280
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4 * 2 / 3
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.3
model.backbone.net.window_block_indexes = (
    list(range(0, 2))
    + list(range(3, 5))
    + list(range(6, 8))
    + list(range(9, 11))
    + list(range(12, 14))
    + list(range(15, 17))
    + list(range(18, 20))
    + list(range(21, 23))
)
