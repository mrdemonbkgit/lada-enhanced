# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import pathlib
import time
from dataclasses import dataclass
from fractions import Fraction

import torch
import torch.multiprocessing as mp

from lada.multigpu.merger import SegmentFile, merge_segments
from lada.utils import audio_utils, video_utils
from lada.utils.video_utils import get_video_meta_data, VideoWriter

logger = logging.getLogger(__name__)


@dataclass
class SegmentSpec:
    gpu_index: int
    device: str
    start_frame: int
    end_frame: int
    overlap_before: int
    overlap_after: int
    temp_path: str


def compute_segments(
    total_frames: int,
    num_gpus: int,
    overlap_frames: int,
    temp_dir: str,
    base_name: str,
) -> list[SegmentSpec]:
    """Split total_frames into num_gpus segments with overlap at boundaries."""
    frames_per_segment = total_frames // num_gpus
    remainder = total_frames % num_gpus
    segments = []
    current_frame = 0

    for i in range(num_gpus):
        seg_len = frames_per_segment + (1 if i < remainder else 0)
        start_frame = current_frame
        end_frame = current_frame + seg_len  # exclusive

        # Add overlap: extend start backwards and end forwards
        overlap_before = 0
        overlap_after = 0
        if i > 0:
            overlap_before = min(overlap_frames, start_frame)
        if i < num_gpus - 1:
            overlap_after = min(overlap_frames, total_frames - end_frame)

        actual_start = start_frame - overlap_before
        actual_end = end_frame + overlap_after

        temp_path = os.path.join(temp_dir, f"{base_name}_segment_{i}.mp4")

        segments.append(SegmentSpec(
            gpu_index=i,
            device=f"cuda:{i}",
            start_frame=actual_start,
            end_frame=actual_end,
            overlap_before=overlap_before,
            overlap_after=overlap_after,
            temp_path=temp_path,
        ))
        current_frame = end_frame

    return segments


def _frame_to_ns(frame_num: int, fps_exact: Fraction) -> int:
    """Convert frame number to nanoseconds offset."""
    return int(Fraction(frame_num, 1) / fps_exact * 1_000_000_000)


def _worker_process(
    gpu_index: int,
    device: str,
    input_path: str,
    temp_path: str,
    start_frame: int,
    max_frames: int,
    fps_exact: Fraction,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    detect_face_mosaics: bool,
    max_clip_length: int,
    encoder: str,
    encoder_options: str,
    mp4_fast_start: bool,
    progress_counter,  # multiprocessing.Value
    error_flag,  # multiprocessing.Value
):
    """Worker process that processes a segment on a specific GPU."""
    try:
        from lada.restorationpipeline import load_models
        from lada.restorationpipeline.frame_restorer import FrameRestorer
        from lada.utils.threading_utils import STOP_MARKER, ErrorMarker

        torch_device = torch.device(device)
        mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode = load_models(
            torch_device, mosaic_restoration_model_name, mosaic_restoration_model_path,
            mosaic_restoration_config_path, mosaic_detection_model_path, fp16, detect_face_mosaics
        )

        video_metadata = get_video_meta_data(input_path)
        start_ns = _frame_to_ns(start_frame, fps_exact)

        frame_restorer = FrameRestorer(
            torch_device, input_path, max_clip_length, mosaic_restoration_model_name,
            mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode,
            max_frames=max_frames,
        )

        frame_restorer.start(start_ns=start_ns)
        try:
            with VideoWriter(
                temp_path, video_metadata.video_width, video_metadata.video_height,
                fps_exact, encoder=encoder, encoder_options=encoder_options,
                time_base=video_metadata.time_base, mp4_fast_start=mp4_fast_start,
            ) as video_writer:
                for elem in frame_restorer:
                    if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                        error_flag.value = 1
                        logger.error(f"GPU {gpu_index}: frame restorer stopped prematurely")
                        break
                    (restored_frame, restored_frame_pts) = elem
                    video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                    with progress_counter.get_lock():
                        progress_counter.value += 1
        finally:
            frame_restorer.stop()

    except Exception as e:
        logger.error(f"GPU {gpu_index} worker error: {e}")
        error_flag.value = 1
        raise


class MultiGPUProgressbar:
    """Progress bar that polls a shared counter from worker processes."""

    def __init__(self, video_metadata, progress_counter):
        from lada.cli.utils import Progressbar
        self.progress_counter = progress_counter
        self.progressbar = Progressbar(video_metadata)

    def init(self):
        self.progressbar.init()
        self._last_count = 0

    def poll(self):
        current = self.progress_counter.value
        delta = current - self._last_count
        for _ in range(delta):
            self.progressbar.update()
        self._last_count = current

    def close(self, success=True):
        self.poll()
        self.progressbar.close(ensure_completed_bar=success)


def process_video_file_multigpu(
    input_path: str,
    output_path: str,
    temp_dir_path: str,
    num_gpus: int,
    overlap_frames: int,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    detect_face_mosaics: bool,
    max_clip_length: int,
    encoder: str,
    encoder_options: str,
    mp4_fast_start: bool,
):
    """Process a video file using multiple GPUs in parallel."""
    video_metadata = get_video_meta_data(input_path)
    total_frames = video_metadata.frames_count
    fps_exact = video_metadata.video_fps_exact
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    if total_frames < num_gpus:
        logger.warning(
            f"Video has only {total_frames} frames but {num_gpus} GPUs requested. "
            f"Using 1 GPU instead."
        )
        num_gpus = 1

    segments = compute_segments(total_frames, num_gpus, overlap_frames, temp_dir_path, base_name)

    # Use 'spawn' start method to get clean CUDA contexts
    ctx = mp.get_context("spawn")

    # Create shared values from the same spawn context
    progress_counter = ctx.Value("i", 0)
    error_flag = ctx.Value("i", 0)

    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    progressbar = MultiGPUProgressbar(video_metadata, progress_counter)

    processes = []

    print(f"Starting {num_gpus} GPU workers...")
    for seg in segments:
        max_frames = seg.end_frame - seg.start_frame
        p = ctx.Process(
            target=_worker_process,
            args=(
                seg.gpu_index,
                seg.device,
                input_path,
                seg.temp_path,
                seg.start_frame,
                max_frames,
                fps_exact,
                mosaic_restoration_model_name,
                mosaic_restoration_model_path,
                mosaic_restoration_config_path,
                mosaic_detection_model_path,
                fp16,
                detect_face_mosaics,
                max_clip_length,
                encoder,
                encoder_options,
                mp4_fast_start,
                progress_counter,
                error_flag,
            ),
        )
        p.start()
        processes.append(p)

    # Poll progress while workers are running
    progressbar.init()
    try:
        while any(p.is_alive() for p in processes):
            progressbar.poll()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Received Ctrl-C, stopping workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=10)
        progressbar.close(success=False)
        _cleanup_segment_files(segments)
        raise

    # Join all processes
    for p in processes:
        p.join()

    success = error_flag.value == 0 and all(p.exitcode == 0 for p in processes)
    progressbar.close(success=success)

    if not success:
        print("Error: one or more GPU workers failed")
        _cleanup_segment_files(segments)
        return

    # Merge segments
    print("Merging segments...")
    video_tmp_file_output_path = os.path.join(
        temp_dir_path,
        f"{base_name}.tmp{os.path.splitext(output_path)[1]}",
    )

    segment_file_specs = []
    for seg in segments:
        seg_frames = seg.end_frame - seg.start_frame
        # Trim: for overlap_before, trim half from the start; for overlap_after, trim half from the end
        trim_start = seg.overlap_before // 2
        trim_end = seg.overlap_after // 2
        segment_file_specs.append(SegmentFile(
            path=seg.temp_path,
            total_frames=seg_frames,
            trim_start_frames=trim_start,
            trim_end_frames=trim_end,
            fps=fps_exact,
        ))

    merge_segments(segment_file_specs, video_tmp_file_output_path, temp_dir_path)

    # Mux audio
    print("Processing audio")
    audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, output_path)

    # Clean up segment temp files
    _cleanup_segment_files(segments)


def _cleanup_segment_files(segments: list[SegmentSpec]):
    for seg in segments:
        if os.path.exists(seg.temp_path):
            os.remove(seg.temp_path)
