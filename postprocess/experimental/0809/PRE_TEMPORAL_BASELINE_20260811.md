# Pre-temporal baseline (2026-08-11)

This is the frozen experimental baseline immediately before adding temporal,
movement, area-acceleration, or local-tail constraints.

## Configuration

- profile: `production_candidate_best_v4`
- shape mode: polygon for all three genital classes
- requested key interval: 8 (penalty target, not a hard key-count constraint)
- hard quality constraint: exact per-frame Recall >= 0.97
- DP: CUDA dense-graph screening plus native DP
- selected key positions/count: fixed after DP
- post-DP shape refinement: one alpha per selected key on the line from the
  selected best-v4 vector to the Production pair-vote vector
- refinement objective: exact adjacent-interval IoU sum only
- refinement constraint: exact Recall >= 0.97 on every affected frame
- coordinate schedule: one forward and one backward sweep
- alpha grid: 1/16 coarse, then 1/128 local refinement
- temporal, movement, area, smoothness, and local-tail penalties: disabled
- post-decode expansion repair during the comparison: disabled

Candidate states are class-specific:

- female: raw, C02_125, G02, G04, A06, F3_P1, D6_P1, F3_Q75_P1
- male: raw, C02_125, G02_H3, G04_H3, A06_K3, F3_P1, D6_R5_P1
- joint: raw, C02_125, G02, G04, A06, VF8_P1, VB8_P1

## Canonical full run

Root:

`output/pre_temporal_baseline_best_v4_pair_vote_i8_20260811`

The rerun exactly matched the earlier full run: 0 changed keys, 0 changed
vertices, and identical metrics on all 24,501 evaluated rows.

| metric | value |
|---|---:|
| video frames | 23,510 |
| evaluated mask rows | 24,501 |
| selected keys before safety export | 3,973 |
| effective interval | 6.456072 |
| mean IoU | 0.888143 |
| minimum / q01 / q05 IoU | 0.235654 / 0.532742 / 0.704593 |
| mean / minimum Recall | 0.986185 / 0.941303 |
| q001 / q01 / q05 Recall | 0.970007 / 0.970069 / 0.970527 |
| exact Recall violations | 4 known CUDA/CPU boundary cases |
| mean / q95 / maximum area-to-source ratio | 1.115699 / 1.393939 / 4.243508 |
| optimization wall time | 165.051 s |
| video throughput | 142.441 FPS |
| mask-row throughput | 148.445 rows/s |

Class metrics:

| class | rows | keys | effective interval | mean IoU | q01 IoU | q05 IoU | exact Recall violations | wall | pair-vote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| female | 9,130 | 1,260 | 7.6081 | 0.923303 | 0.687601 | 0.799543 | 2 | 146.35 s | 52.15 s |
| male | 6,499 | 989 | 6.8567 | 0.911324 | 0.579821 | 0.759243 | 0 | 164.94 s | 66.96 s |
| joint | 8,872 | 1,724 | 5.3982 | 0.834979 | 0.432822 | 0.620362 | 2 | 135.52 s | 47.03 s |

The three class processes run concurrently; their wall times must not be
summed.  The slowest class determines most of the 165.05-second profile wall.

## Software-facing SQLite

The keyframe-primary V3 export is:

`output/pre_temporal_baseline_best_v4_pair_vote_i8_20260811/software_review/12月KPI動画_時間制約前ベースライン_best_v4_pairvote_目標8_Recall097.sqlite`

The exact CPU export audit handled the four boundary cases by inserting three
raw keys and replacing one existing key.  The safe export therefore contains
3,976 authoritative polygon keyframes across 101 segments.

- schema: `video-mask-integrated-result` v3
- contract revision: 5
- compatibility profile: `keyframe-primary-v3`
- schema fingerprint: unchanged
- SQLite integrity: `ok`
- foreign-key errors: 0
- size: 270,901,248 bytes
- SHA-256: `fc3e52d7b296bc7f502a9fff8daf9fe0a14599d5da3798b1e9374c6eb67730a2`

The exporter also emits a legacy experimental evaluation block against the
original inference-label identities.  That block is not comparable to the
Phase-2 classwise source after tracking/reclassification and must not be used
as the quality result.  The exact per-row metrics above are authoritative.

## Remaining caveat

The CUDA proxy metadata reports 79 `infeasible_streams`, while the final exact
CPU raster audit finds only four violating rows/streams and the software export
repairs those four.  This proxy count is diagnostic metadata, not the final
quality denominator.  It should be reconciled before Production promotion,
but it does not change the exact metrics or emitted geometry.
