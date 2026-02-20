# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction

from lada.utils import os_utils

logger = logging.getLogger(__name__)


@dataclass
class SegmentFile:
    path: str
    total_frames: int
    trim_start_frames: int
    trim_end_frames: int
    fps: Fraction


def merge_segments(segment_files: list[SegmentFile], output_path: str, temp_dir: str):
    """Merge segment video files using hard-cut-at-midpoint strategy.

    Each segment's overlap regions are trimmed to the midpoint, then segments
    are concatenated with ffmpeg. This ensures cut points have full temporal
    context from both sides.
    """
    if len(segment_files) == 1:
        os.rename(segment_files[0].path, output_path)
        return

    trimmed_paths = []
    try:
        for i, seg in enumerate(segment_files):
            trimmed_path = os.path.join(temp_dir, f"trimmed_segment_{i}.mp4")
            trimmed_paths.append(trimmed_path)

            fps = seg.fps
            start_time = float(Fraction(seg.trim_start_frames, 1) / fps)
            end_time = float(Fraction(seg.total_frames - seg.trim_end_frames, 1) / fps)

            cmd = [
                "ffmpeg", "-y", "-loglevel", "warning",
                "-i", seg.path,
                "-ss", f"{start_time:.6f}",
                "-t", f"{end_time - start_time:.6f}",
                "-c", "copy",
                trimmed_path,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=os_utils.get_subprocess_startup_info(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg trim failed for segment {i}: {result.stderr.decode()}"
                )

        # Build concat demuxer file
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for path in trimmed_paths:
                f.write(f"file '{path}'\n")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=os_utils.get_subprocess_startup_info(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed: {result.stderr.decode()}"
            )
    finally:
        # Clean up trimmed files and concat list
        for path in trimmed_paths:
            if os.path.exists(path):
                os.remove(path)
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
