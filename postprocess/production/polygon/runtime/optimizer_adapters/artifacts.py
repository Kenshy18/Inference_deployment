"""SQLite and JSON artifact adapters for the Production optimizer."""

from __future__ import annotations

from types import ModuleType


def install_artifact_adapters(
    module: ModuleType,
    original_json_dumps,
    original_load_rows,
) -> None:
    def compact_big_list_dumps(obj, *args, **kwargs):
        if isinstance(obj, list) and kwargs.get("indent") is not None:
            kwargs = dict(kwargs)
            kwargs.pop("indent", None)
            kwargs.setdefault("separators", (",", ":"))
        return original_json_dumps(obj, *args, **kwargs)

    def polygon_cache_key(path):
        try:
            return str(module.Path(path).resolve())
        except OSError:
            return str(module.Path(path))

    def cached_load_rows(sqlite_path):
        rows = original_load_rows(sqlite_path)
        cache = getattr(module, "_polygon_loaded_rows_cache", None)
        if cache is None:
            cache = {}
            module._polygon_loaded_rows_cache = cache
        cache[polygon_cache_key(sqlite_path)] = rows
        return rows

    def cached_evaluate_union_exact(union_rows, tracked_sqlite, output_dir):
        output_dir = module.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_lookup = {
            (int(row["frame"]), str(row["track_id"])): row for row in union_rows
        }
        cache = getattr(module, "_polygon_loaded_rows_cache", {})
        rows = cache.get(polygon_cache_key(tracked_sqlite))
        if rows is None:
            rows = original_load_rows(tracked_sqlite)
            cache[polygon_cache_key(tracked_sqlite)] = rows
            module._polygon_loaded_rows_cache = cache

        result_rows = []
        for row in rows:
            pred = pred_lookup.get((int(row.frame), str(row.track_id)))
            if pred is None:
                continue
            pred_polys = [
                module.np.asarray(poly, dtype=module.np.float32).reshape(-1, 2)
                for poly in pred["polygons"]
            ]
            metrics = module.compute_exact_metrics_from_polygons(
                row.polygons, pred_polys
            )
            weighted_error = float(module.compute_weighted_error(metrics))
            result_rows.append(
                {
                    "frame": int(row.frame),
                    "track_id": str(row.track_id),
                    "run_id": int(pred.get("run_id", -1)),
                    "has_keyframe": int(pred.get("has_keyframe", 0)),
                    "gt_area": float(metrics["gt_area"]),
                    "pred_area": float(metrics["pred_area"]),
                    "intersection": float(metrics["intersection"]),
                    "union": float(metrics["union"]),
                    "recall": float(metrics["recall"]),
                    "precision": float(metrics["precision"]),
                    "iou": float(metrics["iou"]),
                    "weighted_error": weighted_error,
                }
            )

        sorted_result_rows = sorted(
            result_rows,
            key=lambda item: (int(item["frame"]), int(str(item["track_id"]))),
        )
        metrics_csv = output_dir / "keyframe_exact_metrics.csv"
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = module.csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "track_id",
                    "run_id",
                    "has_keyframe",
                    "gt_area",
                    "pred_area",
                    "intersection",
                    "union",
                    "recall",
                    "precision",
                    "iou",
                    "weighted_error",
                ],
            )
            writer.writeheader()
            writer.writerows(sorted_result_rows)
        summary = {
            "input_tracked_sqlite": str(tracked_sqlite),
            "optimized": module.aggregate_exact_rows(sorted_result_rows),
        }
        (output_dir / "summary.json").write_text(
            module.json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    def fast_union_rows_to_pred_sqlite(union_rows, output_sqlite):
        output_sqlite = module.Path(output_sqlite)
        output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if output_sqlite.exists():
            output_sqlite.unlink()
        conn = module.sqlite3.connect(str(output_sqlite))
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)"
            )
            cur.executemany(
                "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                (
                    (
                        int(row["frame"]),
                        str(row["track_id"]),
                        module.json.dumps(row["polygons"], ensure_ascii=False),
                    )
                    for row in union_rows
                ),
            )
            conn.commit()
        finally:
            conn.close()

    module.json.dumps = compact_big_list_dumps
    module.load_rows = cached_load_rows
    module.evaluate_union_exact = cached_evaluate_union_exact
    module.union_rows_to_pred_sqlite = fast_union_rows_to_pred_sqlite


__all__ = ("install_artifact_adapters",)
