#video_optimizer.py
import cv2
import os
from dataclasses import dataclass
from typing import List
import time
from concurrent.futures import ThreadPoolExecutor

@dataclass
class VideoInfo:
    filepath: str
    filename: str
    size: int
    duration: float
    width: int
    height: int
    fps: float
    processing_time: float

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class VideoOptimizer:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def process_folder(self, folder_path: str) -> List[VideoInfo]:
        """Process all videos in a folder"""
        video_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v')):
                    video_files.append(os.path.join(root, file))

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._process_video, video_files))
        return [r for r in results if r is not None]

    def _process_video(self, filepath: str) -> VideoInfo:
        """Process a single video file"""
        try:
            start_time = time.time()
            cap = cv2.VideoCapture(filepath)
            
            if not cap.isOpened():
                return None

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file size
            size = os.path.getsize(filepath)
            
            cap.release()
            
            return VideoInfo(
                filepath=filepath,
                filename=os.path.basename(filepath),
                size=size,
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                processing_time=time.time() - start_time
            )
        except Exception:
            return None 