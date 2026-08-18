"""Media probing, normalization, and coordinate conversion for orchestration."""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from fractions import Fraction
from typing import Any

from .contracts import validate_inference_sqlite
from .rescale_result_sqlite import (
    VideoGeometry,
    rescale_inference_sqlite_for_postprocess,
    rescale_result_sqlite,
)
from .runner_support import (
    BUNDLED_FFMPEG,
    INTERLACED_FIELD_ORDERS,
    REPOSITORY_ROOT,
    OrchestrationError,
    _atomic_json,
    _utc_now,
)


class RunnerMediaMixin:
    """Own video metadata, progressive proxies, and SQLite coordinates."""

    @staticmethod
    def _probe_video(ffprobe: Path, path: Path) -> VideoGeometry:
        """Read display geometry and the exact decodable frame count.

        OpenCV derives ``CAP_PROP_FRAME_COUNT`` from container duration.  MKV
        files whose audio starts slightly before video can therefore report
        one phantom frame.  Container ``nb_frames`` and packet counts can also
        include undecodable samples.  FFprobe's counted decoded frames are
        authoritative for the frame-preserving proxy/deinterlace contract.
        """

        def run_probe(*extra: str) -> dict[str, Any]:
            completed = subprocess.run(
                [
                    str(ffprobe),
                    "-v",
                    "error",
                    *extra,
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    (
                        "stream=width,height,avg_frame_rate,r_frame_rate,"
                        "start_time,duration,nb_frames,nb_read_frames:"
                        "format=start_time,duration"
                    ),
                    "-of",
                    "json",
                    str(path),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "unknown FFprobe error"
                raise OrchestrationError(f"failed to probe video: {path}: {detail}")
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise OrchestrationError(
                    f"FFprobe returned invalid JSON for {path}: {completed.stdout.strip()}"
                ) from exc

        def positive_float(value: Any) -> float | None:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if math.isfinite(parsed) and parsed > 0.0 else None

        def positive_int(value: Any) -> int | None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed > 0 else None

        payload = run_probe()
        variable_rate_mismatch: tuple[float, float] | None = None
        try:
            stream = payload["streams"][0]
            average_rate_text = str(stream.get("avg_frame_rate") or "0/1")
            nominal_rate_text = str(stream.get("r_frame_rate") or "0/1")
            rate_text = str(
                stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
            )
            rate = float(Fraction(rate_text))
            average_rate = float(Fraction(average_rate_text))
            nominal_rate = float(Fraction(nominal_rate_text))
            if (
                average_rate > 0.0
                and nominal_rate > 0.0
                and abs(average_rate - nominal_rate)
                > max(0.02, average_rate * 0.05)
            ):
                variable_rate_mismatch = (average_rate, nominal_rate)
            declared_count = positive_int(stream.get("nb_frames"))
            stream_duration = positive_float(stream.get("duration"))
            if stream_duration is None:
                format_data = payload.get("format") or {}
                format_start = float(format_data.get("start_time") or 0.0)
                format_duration = positive_float(format_data.get("duration"))
                stream_start = float(stream.get("start_time") or format_start)
                if format_duration is not None:
                    stream_duration = max(
                        0.0,
                        format_start + format_duration - stream_start,
                    )
            estimated_count = (
                None if stream_duration is None else stream_duration * rate
            )

            # Most MP4/MOV files expose a trustworthy nb_frames value.  Avoid
            # decoding a multi-hour source merely to repeat that metadata.  A
            # large disagreement with stream timing is the signature of the
            # malformed edit lists that produced phantom frames in practice.
            # Duration-derived counts are only a cheap consistency check.  A
            # percentage tolerance is unsafe for long videos: at 24 fps a
            # 15-minute MP4 can be wrong by twenty frames and still fall
            # inside a 0.1% window.  Such files exist in the deployment
            # corpus (the edit list advertises 21626 frames while FFprobe can
            # decode 21602).  Permit only timestamp rounding noise here; any
            # larger disagreement pays for one exact decoded-frame probe.
            tolerance = None if estimated_count is None else 1.5
            if declared_count is not None and (
                estimated_count is None
                or abs(declared_count - estimated_count) <= tolerance
            ):
                frame_count = declared_count
            elif (
                declared_count is None
                and estimated_count is not None
                and abs(estimated_count - round(estimated_count)) <= 0.1
            ):
                # Matroska commonly omits nb_frames.  Its format duration may
                # begin at a negative audio timestamp, hence the stream-start
                # correction above before accepting this exact integer.
                frame_count = int(round(estimated_count))
            else:
                counted_payload = run_probe("-count_frames")
                counted_stream = counted_payload["streams"][0]
                frame_count = positive_int(counted_stream.get("nb_read_frames"))
                if frame_count is None:
                    raise ValueError("FFprobe did not return nb_read_frames")
            geometry = VideoGeometry(
                width=int(stream["width"]),
                height=int(stream["height"]),
                fps=rate,
                frame_count=frame_count,
            )
        except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise OrchestrationError(
                f"video has invalid FFprobe geometry: {path}: {payload!r}"
            ) from exc
        if (
            min(geometry.width, geometry.height, geometry.frame_count) <= 0
            or geometry.fps <= 0
        ):
            raise OrchestrationError(f"video has invalid geometry: {path}: {geometry}")
        if variable_rate_mismatch is not None:
            average_rate, nominal_rate = variable_rate_mismatch
            raise OrchestrationError(
                "variable-frame-rate input is not supported because inference "
                "and overlay frame timestamps must remain one-to-one; transcode "
                "the video to a constant frame rate before processing: "
                f"average={average_rate:.6f}, nominal={nominal_rate:.6f}: {path}"
            )
        return geometry

    @staticmethod
    def _uses_1080p_proxy(geometry: VideoGeometry) -> bool:
        """Return whether a 16:9 source needs the canonical 1080p workspace.

        Model preprocessing remains unchanged.  The proxy only fixes the pixel
        coordinate system used by inference outputs and every postprocessing
        stage.  Results are rescaled back to the source geometry at publication.
        """

        return geometry.width * 9 == geometry.height * 16 and (
            geometry.width,
            geometry.height,
        ) != (1920, 1080)

    def _prepare_analysis_video(self) -> None:
        """Use one 1080p analysis/postprocess space for non-1080p 16:9 video."""

        _ffmpeg, ffprobe = self._ffmpeg_tools()
        self.original_geometry = self._probe_video(ffprobe, self.config.input_video)
        self.analysis_geometry = (
            self.original_geometry
            if self.processing_video == self.config.input_video
            else self._probe_video(ffprobe, self.processing_video)
        )
        self.inference_video = self.processing_video
        needs_proxy = self._uses_1080p_proxy(self.original_geometry) and (
            self.config.postprocess.enabled
            or (
                self.config.inference.enabled
                and self.original_geometry.width > 1920
                and self.original_geometry.height > 1080
            )
        )
        if not needs_proxy:
            self.analysis_video = self.processing_video
            return
        ffmpeg, _ffprobe = self._ffmpeg_tools()
        if not self._can_resume_stage(
            "analysis_proxy", {"analysis_proxy_video": self.proxy_video_path}
        ):
            self.proxy_video_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(ffmpeg),
                "-hide_banner",
                "-y",
                "-i",
                str(self.processing_video),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "scale=1920:1080:flags=lanczos,format=yuv420p",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "15",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(self.proxy_video_path),
            ]
            self._execute("analysis_proxy", command, cpu_only=True)
        self.analysis_geometry = self._probe_video(ffprobe, self.proxy_video_path)
        if (
            self.analysis_geometry.width != 1920
            or self.analysis_geometry.height != 1080
        ):
            raise OrchestrationError(
                f"analysis proxy has unexpected size: {self.analysis_geometry}"
            )
        if self.analysis_geometry.frame_count != self.original_geometry.frame_count:
            raise OrchestrationError(
                "analysis proxy changed frame count: "
                f"source={self.original_geometry.frame_count}, "
                f"proxy={self.analysis_geometry.frame_count}"
            )
        if abs(self.analysis_geometry.fps - self.original_geometry.fps) > 1e-3:
            raise OrchestrationError(
                "analysis proxy changed fps: "
                f"source={self.original_geometry.fps}, "
                f"proxy={self.analysis_geometry.fps}"
            )
        self.analysis_video = self.proxy_video_path
        # Downscale large sources before inference as before.  Small 16:9
        # sources keep their original pixels for inference; only the emitted
        # SQLite coordinates are enlarged for postprocessing.
        if self.original_geometry.width > 1920 and self.original_geometry.height > 1080:
            self.inference_video = self.proxy_video_path
        self._publish_artifacts(
            {"analysis_proxy_video": self.proxy_video_path},
            validation={
                "analysis_proxy_video": {
                    "source": str(self.config.input_video),
                    "source_width": self.original_geometry.width,
                    "source_height": self.original_geometry.height,
                    "proxy_width": self.analysis_geometry.width,
                    "proxy_height": self.analysis_geometry.height,
                    "frame_count": self.analysis_geometry.frame_count,
                    "fps": self.analysis_geometry.fps,
                }
            },
        )

    def _sqlite_frame_bounds(self, source: Path) -> tuple[int, int] | None:
        """Return the materialized frame domain without revalidating the SQLite."""

        resolved = Path(source).expanduser().resolve()
        cached = self._sqlite_frame_bounds_cache.get(resolved)
        if cached is not None:
            return cached
        # Command construction is also used by dry-run/unit-test callers before
        # artifacts exist. Runtime overlay stages always receive a published,
        # validated SQLite, so retain the requested-range fallback only for that
        # pre-artifact case.
        if not resolved.is_file():
            return None
        try:
            with sqlite3.connect(f"file:{resolved}?mode=ro", uri=True) as connection:
                table = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='frames'
                    """
                ).fetchone()
                columns = (
                    {
                        str(row[1])
                        for row in connection.execute('PRAGMA table_info("frames")')
                    }
                    if table is not None
                    else set()
                )
                if "frame_index" not in columns:
                    return None
                row = connection.execute(
                    "SELECT MIN(frame_index), MAX(frame_index) FROM frames"
                ).fetchone()
        except sqlite3.Error as exc:
            raise OrchestrationError(
                f"could not read overlay frame bounds from {resolved}: {exc}"
            ) from exc
        if row is None or row[0] is None or row[1] is None:
            raise OrchestrationError(
                f"overlay source SQLite has no materialized frames: {resolved}"
            )
        bounds = (int(row[0]), int(row[1]))
        self._sqlite_frame_bounds_cache[resolved] = bounds
        return bounds

    def _ffmpeg_tools(self) -> tuple[Path, Path]:
        """Resolve the matching FFmpeg/FFprobe pair used for normalization."""

        configured = self.config.overlay.ffmpeg_bin
        ffmpeg = configured if configured is not None else BUNDLED_FFMPEG
        if not ffmpeg.is_file():
            system_ffmpeg = shutil.which("ffmpeg")
            if system_ffmpeg is None:
                raise OrchestrationError(
                    "interlace inspection requires FFmpeg, but no executable was found"
                )
            ffmpeg = Path(system_ffmpeg)
        ffprobe = ffmpeg.with_name("ffprobe")
        if not ffprobe.is_file():
            system_ffprobe = shutil.which("ffprobe")
            if system_ffprobe is None:
                raise OrchestrationError(
                    "interlace inspection requires FFprobe, but no executable was found"
                )
            ffprobe = Path(system_ffprobe)
        return ffmpeg.resolve(), ffprobe.resolve()

    @staticmethod
    def _probe_field_order(ffprobe: Path, video: Path) -> str:
        completed = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=field_order",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown FFprobe error"
            raise OrchestrationError(f"could not inspect input field order: {detail}")
        values = [line.strip().lower() for line in completed.stdout.splitlines()]
        return values[0] if values and values[0] else "unknown"

    def _prepare_processing_video(self) -> None:
        """Normalize a flagged interlaced source once for every later stage."""

        ffmpeg, ffprobe = self._ffmpeg_tools()
        field_order = self._probe_field_order(ffprobe, self.config.input_video)
        if field_order not in INTERLACED_FIELD_ORDERS:
            self.processing_video = self.config.input_video
            self._replace_stage_record(
                {
                    "name": "input_normalization",
                    "status": "reused",
                    "action": "use_original_progressive_input",
                    "input_field_order": field_order,
                    "artifact": str(self.config.input_video),
                    "completed_at_utc": _utc_now(),
                }
            )
            return

        output = self.preflight_dir / "input_progressive.mp4"
        if self._can_resume_stage(
            "input_normalization", {"normalized_input_video": output}
        ):
            output_field_order = self._probe_field_order(ffprobe, output)
            if output_field_order not in INTERLACED_FIELD_ORDERS:
                self.processing_video = output
                return

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.orchestrating.mp4")
        temporary.unlink(missing_ok=True)
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(self.config.input_video),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            "bwdif=mode=send_frame:parity=auto:deint=all",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-tune",
            "hq",
            "-rc",
            "constqp",
            "-qp",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            self._execute("input_normalization", command, cpu_only=False)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        output_field_order = self._probe_field_order(ffprobe, output)
        if output_field_order in INTERLACED_FIELD_ORDERS:
            raise OrchestrationError(
                "deinterlaced working input is still marked as interlaced: "
                f"{output_field_order}"
            )
        record = next(
            item
            for item in self.manifest.get("stages", [])
            if item.get("name") == "input_normalization"
        )
        record.update(
            {
                "input_field_order": field_order,
                "output_field_order": output_field_order,
                "artifact": str(output),
            }
        )
        self._replace_stage_record(record)
        self._publish_artifacts({"normalized_input_video": output})
        self.processing_video = output

    @staticmethod
    def _inference_coordinate_size(source: Path) -> tuple[int, int]:
        resolved = source.expanduser().resolve()
        with sqlite3.connect(f"file:{resolved}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT DISTINCT width, height FROM frames LIMIT 2"
            ).fetchall()
        if len(rows) != 1:
            raise OrchestrationError(
                "inference SQLite must use one frame geometry for canonical "
                f"postprocessing: {resolved}: {rows}"
            )
        return int(rows[0][0]), int(rows[0][1])

    def _prepare_postprocess_input(self, source: Path) -> Path:
        """Return inference data expressed in the analysis-video coordinates."""

        if self.analysis_video != self.proxy_video_path:
            return source
        if self.analysis_geometry is None:
            raise OrchestrationError("analysis geometry was not initialized")
        width, height = self._inference_coordinate_size(source)
        target = self.analysis_geometry
        if (width, height) == (target.width, target.height):
            return source
        if width * target.height != height * target.width:
            raise OrchestrationError(
                "inference and postprocess workspace aspect ratios differ: "
                f"inference={width}x{height}, workspace={target.width}x{target.height}"
            )
        expected = {"canonical_inference_sqlite": self.canonical_inference_path}
        if not self._can_resume_stage("postprocess_coordinate_space", expected):
            started = time.perf_counter()
            transform = rescale_inference_sqlite_for_postprocess(
                source,
                self.canonical_inference_path,
                inference=VideoGeometry(
                    width,
                    height,
                    target.fps,
                    target.frame_count,
                ),
                workspace=target,
                workspace_video=self.analysis_video,
            )
            self._replace_stage_record(
                {
                    "name": "postprocess_coordinate_space",
                    "status": "complete",
                    "source": str(source),
                    "artifact": str(self.canonical_inference_path),
                    "coordinate_transform": transform,
                    "elapsed_seconds": time.perf_counter() - started,
                    "completed_at_utc": _utc_now(),
                }
            )
        stats = validate_inference_sqlite(
            self.canonical_inference_path,
            require_segmentation=self.config.inference.uses_segmentation,
            require_faces=self.config.inference.uses_faces,
            expected_face_model=(
                self.config.inference.face_model
                if self.config.inference.uses_faces
                else None
            ),
        )
        self._publish_artifacts(
            {"canonical_inference_sqlite": self.canonical_inference_path},
            validation={"canonical_inference_sqlite": stats},
        )
        return self.canonical_inference_path

    def _restore_original_video_path(self, sqlite_path: Path) -> None:
        """Keep the public SQLite pointed at the user-selected source video."""

        if self.processing_video == self.config.input_video:
            return
        try:
            with sqlite3.connect(sqlite_path) as connection:
                table = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='videos'
                    """
                ).fetchone()
                if table is None:
                    raise OrchestrationError(
                        f"result SQLite has no videos table: {sqlite_path}"
                    )
                connection.execute(
                    "UPDATE videos SET path=?",
                    (str(self.config.input_video),),
                )
                model_metadata = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='model_metadata'
                    """
                ).fetchone()
                if model_metadata is not None:
                    connection.execute(
                        "UPDATE model_metadata SET value=?, value_type='str' "
                        "WHERE key='input'",
                        (str(self.config.input_video),),
                    )
                connection.commit()
        except sqlite3.Error as exc:
            raise OrchestrationError(
                f"could not restore source video path in {sqlite_path}: {exc}"
            ) from exc
