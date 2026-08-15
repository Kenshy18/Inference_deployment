# Production candidate: island cleanup v1

Frozen on 2026-08-13. This is a candidate and remains disabled by default.

## Rule

All holes are filled unconditionally before NMS.

A secondary foreground connected component is removed before NMS when either:

1. its net area is at most 1% of the owner's largest foreground component; or
2. at least 90% of it is covered by another raw instance and its net area is
   at most 30% of that covering instance.

The rule intentionally does not use temporal persistence, screen-edge contact,
or whether the covering instance survives NMS. Those are separate NMS/tracking
problems. Raw inference SQLite is immutable; only the downstream working mask
is changed.

## V3 audit

Source: `output/instance_mask_topology_20260806/topology.sqlite`.

- secondary foreground components: 151
- unconditional <=1% components: 123
- redundant-with-other components: 73
- additional redundant components above 1%: 9
- union removed by candidate v1: 132 (87.4%)
- retained: 19 (12.6%)

The 90% coverage threshold is a raster-tolerant interpretation of containment.
Exact/99.9% containment rejected visibly redundant cases with small contour
alignment differences.

## Activation for evaluation

Set the `nms.adaptive` stage option:

```json
{
  "island_cleanup_policy": "production_candidate_v1"
}
```

Implementation: `postprocess/nms/components.py`, function
`remove_redundant_islands_candidate_v1`.
