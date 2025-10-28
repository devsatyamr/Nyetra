#optimized_gui.py
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable, List
from video_optimizer import VideoOptimizer, VideoInfo

class ProgressWindow(tk.Toplevel):
    def __init__(self, parent, title="Processing"):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x150")
        
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(pady=20, padx=20, fill='x')
        
        self.status_label = ttk.Label(self, text="Initializing...")
        self.status_label.pack(pady=10)
        
    def update_progress(self, value: float, status: str):
        self.progress['value'] = value
        self.status_label['text'] = status
        self.update()

class OptimizedUploadWidget(ttk.Frame):
    def __init__(self, parent, on_complete: Callable[[List[VideoInfo]], None]):
        super().__init__(parent)
        self.on_complete = on_complete
        self.video_optimizer = VideoOptimizer()
        
        # Create upload button
        self.upload_btn = ttk.Button(self, text="Upload Videos", command=self.start_upload)
        self.upload_btn.pack(pady=10)
        
    def start_upload(self):
        folder_path = filedialog.askdirectory(title="Select Video Folder")
        if folder_path:
            self.progress_window = ProgressWindow(self, "Processing Videos")
            self.upload_btn.config(state='disabled')
            
            # Process videos in background
            self.after(100, lambda: self.process_videos(folder_path))
            
    def process_videos(self, folder_path: str):
        try:
            results = self.video_optimizer.process_folder(folder_path)
            self.progress_window.destroy()
            self.upload_btn.config(state='normal')
            self.on_complete(results)
        except Exception as e:
            self.progress_window.destroy()
            self.upload_btn.config(state='normal')
            tk.messagebox.showerror("Error", f"Failed to process videos: {str(e)}") 