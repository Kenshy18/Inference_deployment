"""Semantic contract for the 14/16/18/20-point Production candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PROFILE_ID = "polygon14_keyframe_v1"


@dataclass(frozen=True)
class Polygon14CandidateConfig:
    profile_id: str = PROFILE_ID
    vertices_per_component: int = 14
    vertex_fallbacks: tuple[int, ...] = (14, 16, 18, 20)

    spatial_recall_floor: float = 0.97
    spatial_iou_floor: float = 0.95
    spatial_dense_vertices: int = 64
    spatial_coverage_quantile: float = 0.65
    spatial_maximum_intersection_radius: float = 0.20
    spatial_intersection_regularization: float = 0.01

    temporal_recall_floor: float = 0.97
    temporal_optimizer: str = "new_production_v1_dp_unchanged"
    temporal_target_kind: str = "soft_keyframe_interval"
    pair_vote_mode: str = "per_key_iou_under_exact_recall"
    pair_vote_sweeps: int = 2
    exact_validation: str = "selected_edges_and_final_dense_output"

    topology_constraint: str = "simple_polygon_hard_constraint"
    topology_dp_mode: str = "lazy_selected_edge_split"
    topology_pair_vote_mode: str = "best_iou_valid_trial_or_dp_fallback"
    topology_interpolation_check: str = "every_integer_output_frame"

    reference_mask: str = "tracked_source_mask_from_polygon_stage_input_sqlite"
    output_schema: str = "unchanged"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["vertex_fallbacks"] = list(self.vertex_fallbacks)
        return payload


CANDIDATE = Polygon14CandidateConfig()
