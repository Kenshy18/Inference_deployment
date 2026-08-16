# Production candidate: component-aware mask NMS v2

Frozen for A/B evaluation on 2026-08-13. It is opt-in; legacy Production stays
the default.

## Fixed order

1. Fill every mask hole.
2. Remove a secondary foreground component when its area is at most 1% of the
   owner's largest foreground component.
3. Run stable greedy NMS over the complete remaining instance masks. Bboxes are
   broad-phase only; suppression requires symmetric raster Mask IoU >= 0.70.
4. For surviving instances only, treat every secondary component as a virtual
   island. Remove only that island when at least 80% is covered by another
   survivor's largest foreground component and the island is at most 50% of
   that covering main component.
5. Rebuild `polygons`, `segmentation`, `bbox_xyxy`, and `bbox`, then track.
   When step 4 changed a survivor, only its cleanup-before bbox and mask area
   are carried as private association hints. Tracking uses those hints for ID
   association while persisting the cleaned public geometry. This prevents a
   tiny local cleanup from flipping greedy assignment; the hints are discarded
   before SQLite and do not change its schema.

All component decisions in step 4 use the same immutable survivor snapshot and
are applied simultaneously. Islands are not compared with other islands in v2.
An instance suppressed in step 3 is not rescued because it owned an island.

## Contract and activation

No source inference SQLite is changed and no schema field is added. The policy
is registered as:

```json
{
  "id": "nms",
  "implementation": "nms.component_aware_mask_candidate_v2",
  "options": {
    "mask_iou_threshold": 0.70,
    "fill_all_holes": true,
    "unconditional_owner_ratio_max": 0.01,
    "island_other_coverage_min": 0.80,
    "island_to_other_area_max": 0.50
  }
}
```

Implementation: `postprocess/nms/component_aware.py` and
`postprocess/nms/components.py`.

## 2026-08-13 evaluation status

The candidate remains opt-in. Across nine archived V3 runs (477,691 frames,
431,815 detections), the final direct four-arm audit retained 431,256
detections and performed 49 hole fills, 125 <=1% island removals, and 12
survivor-island removals. The private association hint was emitted for only
those 12 changed detections (1,357 extra JSON bytes across all candidate
outputs). All direct retention decisions remained identical to the pre-hint
candidate.

Dual-geometry tracking removed the observed island-cleanup assignment flip
without changing public masks or SQLite schema. However, the fixed downstream
KPI audit exposed a separate unresolved failure: Mask-IoU 0.70 can retain
same-class nested scale variants whose mutual IoU is low, producing duplicate
tracks and a local polygon-IoU regression. Therefore this candidate must not
replace legacy Production unconditionally until that duplicate/track-split
case is resolved and manually reviewed.
