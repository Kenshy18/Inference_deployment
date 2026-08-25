# Fixed-cardinality Catmull--Rom DP validation (2026-08-25)

## Contract

- Minimum per-frame Recall: `0.97` (hard edge and final-audit constraint)
- Topology: no invalid interpolated frame
- Key count: solve the requested cardinality directly; if infeasible, choose
  the smallest feasible count above it
- IoU/key count: Pareto trade-off selected by the requested interval
- Post-DP key insertion: disabled
- Editable geometry: Catmull--Rom interpolation points `P` only; handles remain
  derived by the frozen `1/6` conversion

## Representative 300-frame tracks

The cases below are deliberately different: female track 26 is a stable
stream, while male track 48 is a difficult non-rigid stream. All rows had zero
Recall violations and zero topology failures.

| Track | Target | Keys | Effective interval | Mean IoU | q05 IoU | Minimum IoU | Maximum area ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| female 26 | 1 | 300 | 1.000 | 0.9927 | 0.9913 | 0.9887 | 1.0030 |
| female 26 | 2 | 150 | 2.000 | 0.9877 | 0.9787 | 0.9711 | 1.0192 |
| female 26 | 3 | 100 | 3.000 | 0.9832 | 0.9717 | 0.9579 | 1.0331 |
| female 26 | 4 | 75 | 4.000 | 0.9783 | 0.9623 | 0.9437 | 1.0558 |
| female 26 | 5 | 60 | 5.000 | 0.9725 | 0.9437 | 0.8599 | 1.1304 |
| female 26 | 6 | 50 | 6.000 | 0.9678 | 0.9332 | 0.8592 | 1.1308 |
| male 48 | 1 | 300 | 1.000 | 0.9909 | 0.9882 | 0.9787 | 1.0035 |
| male 48 | 2 | 150 | 2.000 | 0.9765 | 0.9533 | 0.9161 | 1.0519 |
| male 48 | 3 | 100 | 3.000 | 0.9640 | 0.9359 | 0.9032 | 1.0713 |
| male 48 | 4 | 75 | 4.000 | 0.9484 | 0.9033 | 0.8749 | 1.1300 |
| male 48 | 5 | 60 | 5.000 | 0.9264 | 0.8583 | 0.7551 | 1.2835 |
| male 48 | 6 | 50 | 6.000 | 0.9001 | 0.8146 | 0.7679 | 1.2792 |

The difficult track now exposes the intended trade-off instead of silently
moving target 6 back toward target 3. The preceding penalty/rescue route used
87 keys (effective interval 3.448) for male track 48 at target 6. The new route
uses exactly 50 keys (6.000); its lower IoU is the explicit price of that
sparser Pareto point rather than a hidden target override.

## End-to-end smoke

Source: the existing 300-row KPI track fixture used by prior Production curve
validation.

- Target 3: 100 keys, interval 3.000, minimum Recall 0.97045, mean IoU 0.98540,
  q05 IoU 0.97397, 125.84 emitted FPS
- Target 6: 50 keys, interval 6.000, minimum Recall 0.97002, mean IoU 0.97562,
  q05 IoU 0.95907, 68.12 emitted FPS
- Recall violations: 0
- Topology failures: 0

Artifacts:

- `output/catmull_rom_curve_fixed_cardinality_i3_smoke_20260825`
- `output/catmull_rom_curve_fixed_cardinality_smoke_v2_20260825`

The native fixed-cardinality decode itself took about 3 ms on a 300-frame,
four-state graph. Sparse-target runtime remains dominated by exact interval
raster evaluation and point refinement, not by the cardinality dimension.
