"""
Main video processing pipeline.
Fixed version: correct tensor handling, normalization, and padding.
"""

import cv2
import numpy as np
import logging
import time
import threading
import queue
from typing import Callable, Optional, List, Dict

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, model_loader, batch_size: int = 1):
        self.model_loader = model_loader
        self.batch_size = batch_size

    def process_video(
        self,
        input_path: str,
        output_path: str,
        target_fps: int = 60,
        interpolation_factor: int = 1,
        codec: str = 'mp4v',
        progress_callback: Optional[Callable] = None,
        max_frames: int = None
    ) -> Dict:

        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        source_fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        max_input_frames = None
        max_input_pairs = None
        if max_frames is not None:
            max_input_frames = max_frames
            max_input_pairs = max(0, max_frames - 1)

        output_height = frame_height * 4
        output_width = frame_width * 4

        writer = self._setup_video_writer(
            output_path, output_width, output_height, target_fps, codec
        )

        stats = {
            'frames_processed': 0,
            'frames_written': 0,
            'total_time_seconds': 0,
        }

        input_queue = queue.Queue(maxsize=16)
        output_queue = queue.Queue(maxsize=32)
        stop_event = threading.Event()

        # ======================
        # READER THREAD
        # ======================
        def reader_thread():
            prev_frame = None
            frames_read = 0

            while max_input_frames is None or frames_read < max_input_frames:
                ret, frame = video.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if prev_frame is not None:
                    input_queue.put((prev_frame, frame))

                prev_frame = frame
                frames_read += 1

            input_queue.put(None)

        # ======================
        # WRITER THREAD
        # ======================
        def writer_thread():
            while True:
                frames = output_queue.get()
                if frames is None:
                    break

                for f in frames:
                    f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)

                    if f.shape[:2] != (output_height, output_width):
                        f = cv2.resize(f, (output_width, output_height))

                    writer.write(f)
                    stats['frames_written'] += 1

        t_read = threading.Thread(target=reader_thread, daemon=True)
        t_write = threading.Thread(target=writer_thread, daemon=True)

        t_read.start()
        t_write.start()

        start_time = time.time()

        try:
            while True:
                pair = input_queue.get()
                if pair is None:
                    break

                frame1, frame2 = pair
                stats['frames_processed'] += 1

                is_last = False
                if max_input_pairs is not None and stats['frames_processed'] >= max_input_pairs:
                    is_last = True

                output_frames = self._process_frame_pair(
                    frame1, frame2, interpolation_factor, is_last
                )

                output_queue.put(output_frames)

        finally:
            output_queue.put(None)
            t_write.join()

            video.release()
            writer.release()

            stats['total_time_seconds'] = time.time() - start_time
            
            # Copy audio from input to output
            import os
            output_path_temp = output_path + '.temp.mp4'
            try:
                os.rename(output_path, output_path_temp)
                self._copy_audio_ffmpeg(input_path, output_path_temp, output_path)
                
                # Clean up temp file if audio copy succeeded
                if os.path.exists(output_path_temp):
                    try:
                        os.remove(output_path_temp)
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Audio copy setup failed: {e}")
                # If rename fails, keep original output path

        return stats

    # ======================
    # CORE PROCESSING
    # ======================
    def _process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        interpolation_factor: int,
        is_last_pair: bool
    ) -> List[np.ndarray]:

        from utils.normalization import normalize_frame, to_nchw
        from utils.padding import reflect_pad, crop_padding

        f1 = normalize_frame(frame1)
        f2 = normalize_frame(frame2)

        f1_nchw, pad_info = reflect_pad(to_nchw(f1))
        f2_nchw, _ = reflect_pad(to_nchw(f2))

        output_frames = []

        def finalize_tensor(tensor):
            """
            Robust tensor -> image conversion
            Handles:
            - [1,C,H,W]
            - [-1,1] or [0,1]
            """

            tensor = np.ascontiguousarray(tensor.astype(np.float32))
            tensor = crop_padding(tensor, pad_info, scale=4)

            img = np.squeeze(tensor)

            # CHW -> HWC
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)

            min_val = img.min()
            if min_val < -0.1:
                img = (img + 1.0) / 2.0

            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).round().astype(np.uint8)

            return np.ascontiguousarray(img)

        # Frame 1
        f1_up = self.model_loader.run_upscaler(f1_nchw)
        output_frames.append(finalize_tensor(f1_up))

        # Interpolation
        for _ in range(interpolation_factor):
            inter = self.model_loader.run_interpolator(f1_nchw, f2_nchw)
            inter_up = self.model_loader.run_upscaler(inter)
            output_frames.append(finalize_tensor(inter_up))

        # Frame 2
        if is_last_pair:
            f2_up = self.model_loader.run_upscaler(f2_nchw)
            output_frames.append(finalize_tensor(f2_up))

        return output_frames

    def _setup_video_writer(self, path, w, h, fps, codec):
        """Setup video writer without audio (audio added post-processing)."""
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        return writer if writer.isOpened() else None
    
    def _copy_audio_ffmpeg(self, input_path: str, output_path_video: str, output_path_final: str):
        """Copy audio from input to output using ffmpeg."""
        import subprocess
        import shutil
        
        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
            
            logger.info(f"Copying audio from {input_path} to {output_path_final}...")
            cmd = [
                'ffmpeg',
                '-i', output_path_video,  # video input
                '-i', input_path,          # audio input
                '-c:v', 'copy',            # copy video codec
                '-c:a', 'aac',             # encode audio as AAC
                '-map', '0:v:0',           # map video from first input
                '-map', '1:a:0',           # map audio from second input
                '-shortest',               # stop when shortest stream ends
                '-y',                      # overwrite output
                output_path_final
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            logger.info(f"✓ Audio copied successfully")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning(f"Audio copy failed: {e}. Output will be silent.")
            # Fallback: just rename video file
            try:
                shutil.move(output_path_video, output_path_final)
            except:
                pass
            return False
        except Exception as e:
            logger.warning(f"Audio copy error: {e}")
            return False