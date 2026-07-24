"""Inference-only Detectron LazyConfig for DINOv3 Cascade.

Training data, optimizer, and scheduler configuration are deliberately absent.
The family model builder replaces the ViT backbone and proposal generator
before instantiation; this file owns only the base mask model and Cascade ROI
head structure that survive into inference.
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

[model.roi_heads.pop(key) for key in ("box_head", "box_predictor", "proposal_matcher")]
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
        for w1, w2 in ((10, 5), (20, 10), (30, 15))
    ],
    proposal_matchers=[
        L(Matcher)(
            thresholds=[threshold],
            labels=[0, 1],
            allow_low_quality_matches=False,
        )
        for threshold in (0.5, 0.6, 0.7)
    ],
)
