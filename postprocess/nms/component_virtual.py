"""Role-aware virtual-component NMS Production candidate v3.

Each foreground connected component is treated as a temporary NMS object,
while the canonical output remains one detection per original owner.  Pair
semantics are intentionally asymmetric:

* main vs main: configurable legacy-box or adaptive-mask NMS;
* island vs island: the same configurable NMS, losing island only;
* island vs another owner's main: 80%/50% subordinate test, island only;
* components from the same owner: never compared.

This prevents a small high-score island from deleting a legitimate main
instance, while retaining legacy behaviour for whole-instance duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mask_geometry import association_geometry
from .mask_adaptive import AdaptiveMaskNms
from .components import (
    _component_coverage,
    _descendants,
    _geometry,
    _net_area,
    _with_removed_components,
    fill_holes_and_remove_tiny_islands,
)


@dataclass(frozen=True)
class VirtualComponentNmsDiagnostics:
    input_detections: int = 0
    holes_filled: int = 0
    tiny_islands_removed: int = 0
    main_components: int = 0
    island_components: int = 0
    main_main_pairs: int = 0
    main_owners_suppressed: int = 0
    island_island_pairs: int = 0
    island_island_suppressed: int = 0
    island_main_pairs: int = 0
    island_main_suppressed: int = 0
    output_detections: int = 0

    def as_dict(self) -> dict[str, int]:
        return {field: int(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class _VirtualComponent:
    owner_index: int
    root: int
    is_main: bool
    score: float
    source_detection_id: Any
    area: float
    detection: dict[str, Any]


def _component_detection(
    detection: dict[str, Any],
    geometry: Any,
    root: int,
) -> dict[str, Any]:
    """Return a canonical detection containing exactly one component tree."""
    indices = _descendants(root, geometry.parents)
    removed = set(geometry.foreground) - {root}
    # Removing another foreground root also removes its descendants.  Contours
    # outside the selected component tree cannot survive this construction.
    component = _with_removed_components(detection, geometry, removed)
    retained = [
        geometry.polygons[index].astype(float).tolist() for index in sorted(indices)
    ]
    xs = [float(point[0]) for polygon in retained for point in polygon]
    ys = [float(point[1]) for polygon in retained for point in polygon]
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    component = dict(component)
    component["polygons"] = retained
    component["segmentation"] = retained
    component["bbox_xyxy"] = bbox
    component["bbox"] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    component.pop("_bbox_area", None)
    component.pop("_mask_area", None)
    return component


def _virtual_components(
    detections: list[dict[str, Any]],
) -> tuple[list[Any], list[_VirtualComponent], list[_VirtualComponent]]:
    geometries = [_geometry(detection) for detection in detections]
    mains: list[_VirtualComponent] = []
    islands: list[_VirtualComponent] = []
    for owner_index, (detection, geometry) in enumerate(
        zip(detections, geometries, strict=True)
    ):
        score = float(detection.get("score") or 0.0)
        source_id = detection.get("source_detection_id", owner_index)
        if geometry is None:
            bbox = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1])
            )
            mains.append(
                _VirtualComponent(
                    owner_index=owner_index,
                    root=-1,
                    is_main=True,
                    score=score,
                    source_detection_id=source_id,
                    area=area,
                    detection=detection,
                )
            )
            continue
        for root in geometry.foreground:
            component_indices = _descendants(root, geometry.parents)
            is_main = root == geometry.largest_foreground
            # Canonical detections carry a raster-derived bbox whose half-pixel
            # convention is not recoverable from contour vertices alone.  When
            # there is only one foreground component, virtualisation must be a
            # no-op so main-main NMS is exactly legacy Production.  Rebuilding
            # that bbox shifted borderline IoU values (for example 0.1043 to
            # 0.0991) across a legacy threshold and retained nested duplicates.
            component_detection = (
                detection
                if is_main and len(geometry.foreground) == 1
                else _component_detection(detection, geometry, root)
            )
            virtual = _VirtualComponent(
                owner_index=owner_index,
                root=root,
                is_main=is_main,
                score=score,
                source_detection_id=source_id,
                area=_net_area(geometry, component_indices),
                detection=component_detection,
            )
            (mains if virtual.is_main else islands).append(virtual)
    return geometries, mains, islands


@dataclass(frozen=True)
class VirtualComponentNms:
    """Opt-in unified component-aware NMS candidate.

    Hole filling and <=1% owner-relative island deletion run before virtual
    component NMS.  ``comparison_policy`` selects either frozen legacy-box or
    adaptive-mask comparisons for main-main and island-island pairs.  The
    final SQLite schema is unaffected because all virtual metadata is
    transient.
    """

    name: str = "virtual_component_nms_candidate_v3"
    fill_all_holes: bool = True
    unconditional_owner_ratio_max: float = 0.01
    island_other_coverage_min: float = 0.80
    island_to_other_area_max: float = 0.50
    legacy_iou_threshold: float = 0.20
    legacy_small_iou_threshold: float = 0.10
    legacy_tiny_iou_threshold: float = 0.05
    comparison_policy: str = "legacy_bbox"
    mask_iou_threshold: float = 0.20
    mask_small_iou_threshold: float = 0.10
    mask_tiny_iou_threshold: float = 0.05
    mask_small_area: float = 5000.0
    mask_tiny_area: float = 2000.0
    mask_containment_coverage_min: float = 0.80
    mask_contain_ratio_max: float = 8.0
    mask_small_contain_ratio_max: float = 5.0
    mask_tiny_contain_ratio_max: float = 5.0

    def _legacy(self) -> AdaptiveNms:
        # Historical comparison support is loaded only by archived experiments;
        # the Production path has no dependency on the bbox policy.
        from .adaptive import AdaptiveNms

        return AdaptiveNms(
            iou_threshold=self.legacy_iou_threshold,
            small_iou_threshold=self.legacy_small_iou_threshold,
            tiny_iou_threshold=self.legacy_tiny_iou_threshold,
        )

    def _comparison(self) -> AdaptiveNms | AdaptiveMaskNms:
        if self.comparison_policy == "legacy_bbox":
            return self._legacy()
        if self.comparison_policy == "adaptive_mask":
            return AdaptiveMaskNms(
                iou_threshold=self.mask_iou_threshold,
                small_iou_threshold=self.mask_small_iou_threshold,
                tiny_iou_threshold=self.mask_tiny_iou_threshold,
                small_area=self.mask_small_area,
                tiny_area=self.mask_tiny_area,
                containment_coverage_min=self.mask_containment_coverage_min,
                contain_ratio_max=self.mask_contain_ratio_max,
                small_contain_ratio_max=self.mask_small_contain_ratio_max,
                tiny_contain_ratio_max=self.mask_tiny_contain_ratio_max,
            )
        raise ValueError(f"unsupported comparison policy: {self.comparison_policy}")

    @staticmethod
    def _pair_decision(
        policy: AdaptiveNms | AdaptiveMaskNms,
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> tuple[str | None, dict[str, float]]:
        if isinstance(policy, AdaptiveMaskNms):
            metrics = policy.pair_metrics(first, second)
            threshold_area = policy.pair_threshold_area(first, second)
            iou_threshold, contain_ratio_max = policy.thresholds_for_area(
                threshold_area
            )
            return policy.suppression_reason_from_metrics(
                metrics,
                threshold_area=threshold_area,
            ), {
                "mask_iou": metrics.iou,
                "smaller_coverage": metrics.smaller_coverage,
                "smaller_to_larger_area_ratio": (metrics.smaller_to_larger_area_ratio),
                "threshold_area": threshold_area,
                "iou_threshold": iou_threshold,
                "contain_ratio_max": contain_ratio_max,
            }
        return policy.pair_suppression_reason(first, second), {}

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retained, _, _ = self.apply_with_trace(detections)
        return retained

    def apply_with_diagnostics(
        self, detections: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], VirtualComponentNmsDiagnostics]:
        retained, diagnostics, _ = self.apply_with_trace(detections)
        return retained, diagnostics

    def apply_with_trace(
        self, detections: list[dict[str, Any]]
    ) -> tuple[
        list[dict[str, Any]],
        VirtualComponentNmsDiagnostics,
        list[dict[str, Any]],
    ]:
        if not detections:
            return [], VirtualComponentNmsDiagnostics(), []

        preprocessed, topology = fill_holes_and_remove_tiny_islands(
            detections,
            fill_all_holes=self.fill_all_holes,
            unconditional_owner_ratio_max=self.unconditional_owner_ratio_max,
        )
        geometries, mains, islands = _virtual_components(preprocessed)
        comparison = self._comparison()
        trace: list[dict[str, Any]] = []

        main_order = sorted(
            mains,
            key=lambda item: (-item.score, item.owner_index, item.root),
        )
        suppressed_owners: set[int] = set()
        retained_mains: list[_VirtualComponent] = []
        main_main_pairs = 0
        for position, winner in enumerate(main_order):
            if winner.owner_index in suppressed_owners:
                continue
            retained_mains.append(winner)
            for loser in main_order[position + 1 :]:
                if loser.owner_index in suppressed_owners:
                    continue
                main_main_pairs += 1
                reason, overlap = self._pair_decision(
                    comparison, winner.detection, loser.detection
                )
                if reason is None:
                    continue
                suppressed_owners.add(loser.owner_index)
                trace.append(
                    {
                        "reason": (
                            "main_main_mask_nms"
                            if self.comparison_policy == "adaptive_mask"
                            else "main_main_legacy_nms"
                        ),
                        "suppression_reason": reason,
                        "legacy_reason": (
                            reason if self.comparison_policy == "legacy_bbox" else None
                        ),
                        "winner_owner": winner.owner_index,
                        "winner_source_detection_id": winner.source_detection_id,
                        "loser_owner": loser.owner_index,
                        "loser_source_detection_id": loser.source_detection_id,
                        **overlap,
                    }
                )

        surviving_islands = [
            item for item in islands if item.owner_index not in suppressed_owners
        ]
        island_order = sorted(
            surviving_islands,
            key=lambda item: (-item.score, item.owner_index, item.root),
        )
        removed_islands: set[tuple[int, int]] = set()
        island_island_pairs = 0
        island_island_suppressed = 0
        for position, winner in enumerate(island_order):
            winner_key = (winner.owner_index, winner.root)
            if winner_key in removed_islands:
                continue
            for loser in island_order[position + 1 :]:
                loser_key = (loser.owner_index, loser.root)
                if (
                    loser.owner_index == winner.owner_index
                    or loser_key in removed_islands
                ):
                    continue
                island_island_pairs += 1
                reason, overlap = self._pair_decision(
                    comparison, winner.detection, loser.detection
                )
                if reason is None:
                    continue
                removed_islands.add(loser_key)
                island_island_suppressed += 1
                trace.append(
                    {
                        "reason": (
                            "island_island_mask_nms"
                            if self.comparison_policy == "adaptive_mask"
                            else "island_island_legacy_nms"
                        ),
                        "suppression_reason": reason,
                        "legacy_reason": (
                            reason if self.comparison_policy == "legacy_bbox" else None
                        ),
                        "winner_owner": winner.owner_index,
                        "winner_source_detection_id": winner.source_detection_id,
                        "winner_root": winner.root,
                        "loser_owner": loser.owner_index,
                        "loser_source_detection_id": loser.source_detection_id,
                        "loser_root": loser.root,
                        **overlap,
                    }
                )

        retained_main_by_owner = {
            item.owner_index: item
            for item in retained_mains
            if item.owner_index not in suppressed_owners
        }
        island_main_pairs = 0
        island_main_suppressed = 0
        for island in island_order:
            island_key = (island.owner_index, island.root)
            if island_key in removed_islands:
                continue
            owner_geometry = geometries[island.owner_index]
            if owner_geometry is None or island.area <= 0.0:
                continue
            for main in retained_main_by_owner.values():
                if main.owner_index == island.owner_index or main.area <= 0.0:
                    continue
                island_main_pairs += 1
                area_ratio = island.area / main.area
                if area_ratio > self.island_to_other_area_max:
                    continue
                other_geometry = geometries[main.owner_index]
                if other_geometry is None:
                    continue
                coverage = _component_coverage(
                    owner_geometry,
                    island.root,
                    other_geometry,
                    other_root=main.root,
                )
                if coverage < self.island_other_coverage_min:
                    continue
                removed_islands.add(island_key)
                island_main_suppressed += 1
                trace.append(
                    {
                        "reason": "island_subordinate_to_main",
                        "island_owner": island.owner_index,
                        "island_source_detection_id": island.source_detection_id,
                        "island_root": island.root,
                        "main_owner": main.owner_index,
                        "main_source_detection_id": main.source_detection_id,
                        "coverage": coverage,
                        "island_to_main_area_ratio": area_ratio,
                    }
                )
                break

        retained: list[dict[str, Any]] = []
        for main in retained_mains:
            owner_index = main.owner_index
            if owner_index in suppressed_owners:
                continue
            geometry = geometries[owner_index]
            if geometry is None:
                retained.append(preprocessed[owner_index])
                continue
            removed_roots = {
                root
                for candidate_owner, root in removed_islands
                if candidate_owner == owner_index
            }
            original = detections[owner_index]
            before = preprocessed[owner_index]
            after = _with_removed_components(before, geometry, removed_roots)
            if after is not original:
                # Hole fill, <=1% cleanup and component-level suppression are
                # final-mask topology corrections.  They must not perturb the
                # otherwise fixed temporal association: a small area change
                # can flip one greedy match and propagate to hundreds of DP
                # frames.  Track with the immutable raw AI geometry while
                # exposing only the cleaned public geometry downstream.
                after = {**after, **association_geometry(original)}
            retained.append(after)

        diagnostics = VirtualComponentNmsDiagnostics(
            input_detections=len(detections),
            holes_filled=topology.holes_filled,
            tiny_islands_removed=topology.tiny_islands_removed,
            main_components=len(mains),
            island_components=len(islands),
            main_main_pairs=main_main_pairs,
            main_owners_suppressed=len(suppressed_owners),
            island_island_pairs=island_island_pairs,
            island_island_suppressed=island_island_suppressed,
            island_main_pairs=island_main_pairs,
            island_main_suppressed=island_main_suppressed,
            output_detections=len(retained),
        )
        return retained, diagnostics, trace


DEFAULT_VIRTUAL_COMPONENT_NMS = VirtualComponentNms()


@dataclass(frozen=True)
class VirtualComponentMaskNms(VirtualComponentNms):
    """Virtual-component candidate with adaptive exact-mask comparisons."""

    name: str = "virtual_component_mask_nms_candidate_v4"
    comparison_policy: str = "adaptive_mask"


DEFAULT_VIRTUAL_COMPONENT_MASK_NMS = VirtualComponentMaskNms()


@dataclass(frozen=True)
class ProductionVirtualComponentNms(VirtualComponentNms):
    """Promoted exact-mask/virtual-component implementation."""

    name: str = "production_virtual_component_mask_nms_v1"
    comparison_policy: str = "adaptive_mask"


DEFAULT_PRODUCTION_NMS = ProductionVirtualComponentNms()
