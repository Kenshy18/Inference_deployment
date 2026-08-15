# new_production benchmark (2026-08-12)

SQLite polygon geometry only; no video frame was decoded.

| target | actual | keys | mean IoU | min / q01 / q05 IoU | ref | optimized | speedup | 20 min estimate | byte parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1.201 | 20,903 | 0.976522 | 0.334749 / 0.853097 / 0.944237 | 147.24s | 126.66s | 14.0% | 3:14 | yes |
| 3 | 2.980 | 8,487 | 0.945203 | 0.238432 / 0.692546 / 0.803414 | 160.28s | 127.48s | 20.5% | 3:15 | yes |
| 6 | 5.460 | 4,679 | 0.902577 | 0.238323 / 0.469187 / 0.650679 | 158.35s | 125.15s | 21.0% | 3:11 | yes |

The 20-minute estimate assumes the same 30000/1001 fps and mask density.
It covers this polygon keyframe optimization stage, not inference or overlay.
The exact CPU audit still sees the frozen baseline's known CUDA/OpenCV boundary cases; the optimized engine neither adds nor removes them.
