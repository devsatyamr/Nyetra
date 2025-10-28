import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️ face_recognition not available")

try:
    from deepface import DeepFace
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace library loaded successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available, using fallback methods")

class FaceRecognitionTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Test Interface - DeepFace Enhanced")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Color scheme
        self.colors = {
            'bg_primary': '#1a1a1a',
            'bg_secondary': '#2d2d2d',
            'card_bg': '#333333',
            'accent': '#238636',
            'accent_hover': '#2ea043',
            'text_primary': '#ffffff',
            'text_secondary': '#b3b3b3',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'border': '#555555'
        }
        
        # Variables
        self.reference_image = None
        self.reference_encoding = None
        self.reference_features = None  # For OpenCV fallback
        self.reference_image_path = None  # For DeepFace
        self.test_image = None
        self.test_image_path = None  # For static test images
        self.video_capture = None
        self.is_camera_running = False
        self.camera_thread = None
        
        # Status tracking
        self.reference_loaded = False
        self.test_loaded = False
        
        # Animation tracking
        self.scanning_active = False
        self.scan_line_position = 0
        self.scan_direction = 1
        self.animation_after_id = None
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize ORB detector for feature matching (fallback method)
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        title_frame.pack(fill='x', padx=20, pady=(20, 0))
        
        title_text = "🤖 Face Recognition Test Laboratory"
        if DEEPFACE_AVAILABLE:
            title_text += " - DeepFace Enhanced"
        
        title_label = tk.Label(
            title_frame,
            text=title_text,
            font=('Segoe UI', 20, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        )
        title_label.pack(anchor='w')
        
        subtitle_text = "Upload reference faces and test recognition accuracy"
        if DEEPFACE_AVAILABLE:
            subtitle_text += " with state-of-the-art deep learning models"
        
        subtitle_label = tk.Label(
            title_frame,
            text=subtitle_text,
            font=('Segoe UI', 12),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack(anchor='w', pady=(5, 0))
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Image viewers
        left_panel = tk.Frame(main_container, bg=self.colors['bg_primary'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right panel - Controls
        right_panel = tk.Frame(main_container, bg=self.colors['bg_primary'], width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        self.setup_image_viewers(left_panel)
        self.setup_controls(right_panel)
    
    def setup_image_viewers(self, parent):
        """Setup image viewer panels"""
        # Reference image viewer
        ref_frame = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=2)
        ref_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Reference header
        ref_header = tk.Frame(ref_frame, bg=self.colors['accent'], height=40)
        ref_header.pack(fill='x')
        ref_header.pack_propagate(False)
        
        tk.Label(
            ref_header,
            text="📷 Reference Face",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15
        ).pack(side='left', pady=8)
        
        # Reference image area
        self.ref_image_frame = tk.Frame(ref_frame, bg=self.colors['card_bg'])
        self.ref_image_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.ref_image_label = tk.Label(
            self.ref_image_frame,
            text="🖼️\n\nClick to upload reference image\n(JPG, PNG supported)",
            font=('Segoe UI', 12),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            relief='ridge',
            bd=2,
            cursor='hand2'
        )
        self.ref_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.ref_image_label.bind("<Button-1>", lambda e: self.upload_reference_image())
        
        # Test image viewer
        test_frame = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=2)
        test_frame.pack(fill=tk.BOTH, expand=True)
        
        # Test header
        test_header = tk.Frame(test_frame, bg=self.colors['accent'], height=40)
        test_header.pack(fill='x')
        test_header.pack_propagate(False)
        
        tk.Label(
            test_header,
            text="🔍 Test Image / Live Camera",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15
        ).pack(side='left', pady=8)
        
        # Test image area
        self.test_image_frame = tk.Frame(test_frame, bg=self.colors['card_bg'])
        self.test_image_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.test_image_label = tk.Label(
            self.test_image_frame,
            text="🎥\n\nUpload test image or start camera\nfor live recognition",
            font=('Segoe UI', 12),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            relief='ridge',
            bd=2
        )
        self.test_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_controls(self, parent):
        """Setup control panel"""
        # Upload Controls Card
        upload_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=2)
        upload_card.pack(fill='x', pady=(0, 15))
        
        # Card header
        upload_header = tk.Frame(upload_card, bg=self.colors['accent'], height=35)
        upload_header.pack(fill='x')
        upload_header.pack_propagate(False)
        
        tk.Label(
            upload_header,
            text="📁 Upload Controls",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        upload_content = tk.Frame(upload_card, bg=self.colors['card_bg'])
        upload_content.pack(fill='x', padx=12, pady=12)
        
        # Upload reference button
        self.upload_ref_btn = tk.Button(
            upload_content,
            text="📷 Upload Reference Face",
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.upload_reference_image
        )
        self.upload_ref_btn.pack(fill='x', pady=(0, 8))
        
        # Upload test button
        self.upload_test_btn = tk.Button(
            upload_content,
            text="🔍 Upload Test Image",
            font=('Segoe UI', 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['border'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.upload_test_image
        )
        self.upload_test_btn.pack(fill='x')
        
        # Camera Controls Card
        camera_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=2)
        camera_card.pack(fill='x', pady=(0, 15))
        
        # Card header
        camera_header = tk.Frame(camera_card, bg=self.colors['accent'], height=35)
        camera_header.pack(fill='x')
        camera_header.pack_propagate(False)
        
        tk.Label(
            camera_header,
            text="🎥 Camera Controls",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        camera_content = tk.Frame(camera_card, bg=self.colors['card_bg'])
        camera_content.pack(fill='x', padx=12, pady=12)
        
        # Start camera button
        self.start_camera_btn = tk.Button(
            camera_content,
            text="📹 Start Live Camera",
            font=('Segoe UI', 11),
            bg=self.colors['success'],
            fg='white',
            activebackground='#1e7e34',
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.start_camera
        )
        self.start_camera_btn.pack(fill='x', pady=(0, 8))
        
        # Stop camera button
        self.stop_camera_btn = tk.Button(
            camera_content,
            text="⏹️ Stop Camera",
            font=('Segoe UI', 11),
            bg=self.colors['error'],
            fg='white',
            activebackground='#c82333',
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            state='disabled',
            command=self.stop_camera
        )
        self.stop_camera_btn.pack(fill='x')
        
        # Recognition Control Card
        recognition_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=2)
        recognition_card.pack(fill='x', pady=(15, 15))
        
        # Card header
        recognition_header = tk.Frame(recognition_card, bg=self.colors['accent'], height=35)
        recognition_header.pack(fill='x')
        recognition_header.pack_propagate(False)
        
        tk.Label(
            recognition_header,
            text="🧠 Face Recognition",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        recognition_content = tk.Frame(recognition_card, bg=self.colors['card_bg'])
        recognition_content.pack(fill='x', padx=12, pady=12)
        
        # Recognition status
        self.recognition_status = tk.Label(
            recognition_content,
            text="📝 Upload reference and test images to begin",
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary'],
            wraplength=300
        )
        self.recognition_status.pack(fill='x', pady=(0, 10))
        
        # Run recognition button
        self.run_recognition_btn = tk.Button(
            recognition_content,
            text="🔍 Run Face Recognition",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['warning'],
            fg='white',
            activebackground='#e0a800',
            activeforeground='white',
            bd=0,
            padx=20,
            pady=12,
            cursor="hand2",
            state='disabled',
            command=self.run_recognition
        )
        self.run_recognition_btn.pack(fill='x')
        
        # Recognition Settings Card
        settings_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=2)
        settings_card.pack(fill='x', pady=(0, 15))
        
        # Card header
        settings_header = tk.Frame(settings_card, bg=self.colors['accent'], height=35)
        settings_header.pack(fill='x')
        settings_header.pack_propagate(False)
        
        tk.Label(
            settings_header,
            text="⚙️ Recognition Settings",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        settings_content = tk.Frame(settings_card, bg=self.colors['card_bg'])
        settings_content.pack(fill='x', padx=12, pady=12)
        
        # Tolerance setting (for face_recognition method only)
        self.tolerance_frame = tk.Frame(settings_content, bg=self.colors['card_bg'])
        
        tk.Label(
            self.tolerance_frame,
            text="Recognition Tolerance:",
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        tolerance_frame = tk.Frame(self.tolerance_frame, bg=self.colors['card_bg'])
        tolerance_frame.pack(fill='x', pady=(5, 10))
        
        self.tolerance_var = tk.DoubleVar(value=0.6)
        tolerance_scale = tk.Scale(
            tolerance_frame,
            from_=0.3,
            to=0.9,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.tolerance_var,
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_secondary']
        )
        tolerance_scale.pack(side='left', fill='x', expand=True)
        
        tolerance_label = tk.Label(
            tolerance_frame,
            textvariable=self.tolerance_var,
            font=('Segoe UI', 9),
            bg=self.colors['card_bg'],
            fg=self.colors['accent'],
            width=5
        )
        tolerance_label.pack(side='right', padx=(5, 0))
        
        # Model selection
        tk.Label(
            settings_content,
            text="Recognition Method:",
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 5))
        
        # Determine available methods
        methods = []
        if DEEPFACE_AVAILABLE:
            methods.extend(["DeepFace-VGG", "DeepFace-Facenet", "DeepFace-OpenFace", "DeepFace-ArcFace"])
        if FACE_RECOGNITION_AVAILABLE:
            methods.extend(["face_recognition-hog", "face_recognition-cnn"])
        methods.append("OpenCV-ORB")
        
        self.model_var = tk.StringVar(value=methods[0] if methods else "OpenCV-ORB")
        model_combo = ttk.Combobox(
            settings_content,
            textvariable=self.model_var,
            values=methods,
            state="readonly",
            font=('Segoe UI', 10)
        )
        model_combo.pack(fill='x', pady=(0, 10))
        
        # Bind model change to show/hide tolerance setting
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Show tolerance frame initially if needed
        self.on_model_change()
        
        # Results Card
        results_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=2)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        results_header = tk.Frame(results_card, bg=self.colors['accent'], height=35)
        results_header.pack(fill='x')
        results_header.pack_propagate(False)
        
        tk.Label(
            results_header,
            text="📊 Recognition Results",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Clear results button
        clear_btn = tk.Button(
            results_header,
            text="🗑️ Clear",
            font=('Segoe UI', 9),
            bg=self.colors['error'],
            fg='white',
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self.clear_results
        )
        clear_btn.pack(side='right', padx=12, pady=6)
        
        # Card content
        results_content = tk.Frame(results_card, bg=self.colors['card_bg'])
        results_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        # Results summary section
        self.results_summary_frame = tk.Frame(results_content, bg=self.colors['bg_secondary'], relief='flat', bd=1)
        self.results_summary_frame.pack(fill='x', pady=(0, 10))
        
        # Summary content
        summary_content = tk.Frame(self.results_summary_frame, bg=self.colors['bg_secondary'])
        summary_content.pack(fill='x', padx=12, pady=8)
        
        # Latest result display
        self.latest_result_label = tk.Label(
            summary_content,
            text="🎯 Latest Recognition Result",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        )
        self.latest_result_label.pack(anchor='w')
        
        self.latest_result_text = tk.Label(
            summary_content,
            text="No recognition performed yet",
            font=('Segoe UI', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            wraplength=300,
            justify='left'
        )
        self.latest_result_text.pack(anchor='w', pady=(5, 0))
        
        # Detailed log section
        log_label = tk.Label(
            results_content,
            text="📝 Detailed Activity Log",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        )
        log_label.pack(anchor='w', pady=(0, 5))
        
        # Results text area with scrollbar
        log_frame = tk.Frame(results_content, bg=self.colors['card_bg'])
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(
            log_frame,
            height=8,
            font=('Consolas', 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            relief='flat',
            wrap=tk.WORD,
            padx=10,
            pady=8
        )
        
        results_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial message
        if DEEPFACE_AVAILABLE:
            self.results_text.insert(tk.END, "🔬 Face Recognition Test Laboratory initialized\n")
            self.results_text.insert(tk.END, "📝 Upload a reference face to begin testing\n")
            self.results_text.insert(tk.END, "🧠 Using DeepFace with state-of-the-art models\n")
            self.results_text.insert(tk.END, "💡 Try different models: VGG-Face, Facenet, OpenFace, ArcFace\n")
        elif FACE_RECOGNITION_AVAILABLE:
            self.results_text.insert(tk.END, "🔬 Face Recognition Test Laboratory initialized\n")
            self.results_text.insert(tk.END, "📝 Upload a reference face to begin testing\n")
            self.results_text.insert(tk.END, "💡 Lower tolerance = stricter matching\n")
            self.results_text.insert(tk.END, "🚀 Using face_recognition library\n")
        else:
            self.results_text.insert(tk.END, "🔬 Face Recognition Test Laboratory initialized\n")
            self.results_text.insert(tk.END, "📝 Upload a reference face to begin testing\n")
            self.results_text.insert(tk.END, "⚙️ Using OpenCV + ORB feature matching\n")
            self.results_text.insert(tk.END, "💡 Install DeepFace for better accuracy\n")
        self.results_text.config(state='disabled')
    
    def update_recognition_status(self):
        """Update recognition button status and text"""
        if self.reference_loaded and self.test_loaded:
            self.run_recognition_btn.config(state='normal', bg=self.colors['success'])
            self.recognition_status.config(
                text="✅ Ready for recognition - Click button to analyze",
                fg=self.colors['success']
            )
        elif self.reference_loaded:
            self.run_recognition_btn.config(state='disabled', bg=self.colors['warning'])
            self.recognition_status.config(
                text="📷 Reference loaded - Upload test image to continue",
                fg=self.colors['warning']
            )
        elif self.test_loaded:
            self.run_recognition_btn.config(state='disabled', bg=self.colors['warning'])
            self.recognition_status.config(
                text="🔍 Test image loaded - Upload reference face to continue",
                fg=self.colors['warning']
            )
        else:
            self.run_recognition_btn.config(state='disabled', bg=self.colors['warning'])
            self.recognition_status.config(
                text="📝 Upload reference and test images to begin",
                fg=self.colors['text_secondary']
            )
    
    def clear_results(self):
        """Clear all results"""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')
        
        self.latest_result_text.config(
            text="No recognition performed yet",
            fg=self.colors['text_secondary']
        )
        
        self.log_result("🗑️ Results cleared")
    
    def run_recognition(self):
        """Run recognition on uploaded images"""
        if not (self.reference_loaded and self.test_loaded):
            messagebox.showwarning("Warning", "Please upload both reference and test images first")
            return
        
        # Start scanning animation
        self.start_scanning_animation()
        
        # Disable button during processing
        self.run_recognition_btn.config(state='disabled', text="🔄 Processing...")
        
        # Clear previous results
        self.latest_result_text.config(
            text="🔄 Analyzing faces... Please wait",
            fg=self.colors['warning']
        )
        
        # Run recognition in a separate thread to avoid blocking UI
        recognition_thread = threading.Thread(target=self._perform_recognition_thread, daemon=True)
        recognition_thread.start()
    
    def _perform_recognition_thread(self):
        """Perform recognition in a separate thread"""
        try:
            method = self.model_var.get()
            
            if method.startswith("DeepFace") and DEEPFACE_AVAILABLE and self.test_image_path:
                # Use DeepFace
                self.perform_deepface_recognition(self.test_image_path, os.path.basename(self.test_image_path))
                
            elif method.startswith("face_recognition") and FACE_RECOGNITION_AVAILABLE and self.test_image:
                # Use face_recognition library
                self.perform_recognition(self.test_image, os.path.basename(self.test_image_path or "Test Image"))
                
            else:
                # Use OpenCV fallback
                if self.test_image is not None:
                    test_image_bgr = self.test_image
                    if len(self.test_image.shape) == 3 and self.test_image.shape[2] == 3:
                        # Assume RGB, convert to BGR for OpenCV
                        test_image_bgr = cv2.cvtColor(self.test_image, cv2.COLOR_RGB2BGR)
                    
                    self.perform_opencv_recognition(test_image_bgr, os.path.basename(self.test_image_path or "Test Image"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_result(f"❌ Recognition error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recognition failed: {str(e)}"))
        finally:
            # Stop animation and re-enable button
            self.root.after(0, self.stop_scanning_animation)
            self.root.after(0, lambda: self.run_recognition_btn.config(
                state='normal', 
                text="🔍 Run Face Recognition"
            ))
    
    def update_latest_result(self, is_match, confidence, method, details=""):
        """Update the latest result display"""
        if is_match:
            status = "✅ MATCH FOUND"
            color = self.colors['success']
            icon = "🎉"
        else:
            status = "❌ NO MATCH"
            color = self.colors['error']
            icon = "🚫"
        
        result_text = f"{icon} {status}\n"
        result_text += f"Confidence: {confidence:.1f}% | Method: {method}"
        if details:
            result_text += f"\n{details}"
        
        self.latest_result_text.config(
            text=result_text,
            fg=color
        )
    
    def on_model_change(self, event=None):
        """Handle model selection change"""
        method = self.model_var.get()
        if method.startswith("face_recognition"):
            self.tolerance_frame.pack(fill='x', pady=(0, 10))
        else:
            self.tolerance_frame.pack_forget()
        
        # Reset status when model changes
        self.reference_loaded = False
        self.test_loaded = False
        self.reference_image = None
        self.reference_encoding = None
        self.reference_features = None
        self.reference_image_path = None
        self.test_image = None
        self.test_image_path = None
        self.update_recognition_status()
    
    def start_scanning_animation(self):
        """Start the scanning animation on both images"""
        self.scanning_active = True
        self.scan_line_position = 0
        self.scan_direction = 1
        
        # Create scanning overlays if they don't exist
        self.create_scanning_overlay(self.ref_image_label, "ref_overlay")
        self.create_scanning_overlay(self.test_image_label, "test_overlay")
        
        # Start animation loop
        self.animate_scan_line()
    
    def stop_scanning_animation(self):
        """Stop the scanning animation"""
        self.scanning_active = False
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        
        # Remove overlays
        self.remove_scanning_overlay(self.ref_image_label)
        self.remove_scanning_overlay(self.test_image_label)
    
    def create_scanning_overlay(self, parent_label, overlay_name):
        """Create a scanning overlay on an image label"""
        # Create canvas overlay
        canvas = tk.Canvas(
            parent_label,
            highlightthickness=0,
            bg=self.colors['card_bg']
        )
        canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Store canvas reference
        setattr(self, overlay_name, canvas)
        
        # Create scanning line
        line_id = canvas.create_line(0, 0, 0, 0, fill=self.colors['accent'], width=3)
        setattr(self, f"{overlay_name}_line", line_id)
        
        # Create scanning effect text
        text_id = canvas.create_text(
            0, 0, 
            text="🔍 SCANNING...", 
            fill=self.colors['accent'], 
            font=('Segoe UI', 12, 'bold')
        )
        setattr(self, f"{overlay_name}_text", text_id)
    
    def remove_scanning_overlay(self, parent_label):
        """Remove scanning overlay from image label"""
        for overlay_name in ["ref_overlay", "test_overlay"]:
            if hasattr(self, overlay_name):
                overlay = getattr(self, overlay_name)
                overlay.destroy()
                delattr(self, overlay_name)
                
                # Clean up line and text references
                for attr in [f"{overlay_name}_line", f"{overlay_name}_text"]:
                    if hasattr(self, attr):
                        delattr(self, attr)
    
    def animate_scan_line(self):
        """Animate the scanning line"""
        if not self.scanning_active:
            return
        
        for overlay_name in ["ref_overlay", "test_overlay"]:
            if hasattr(self, overlay_name):
                canvas = getattr(self, overlay_name)
                line_id = getattr(self, f"{overlay_name}_line", None)
                text_id = getattr(self, f"{overlay_name}_text", None)
                
                if line_id and text_id:
                    # Get canvas dimensions
                    canvas.update_idletasks()
                    width = canvas.winfo_width()
                    height = canvas.winfo_height()
                    
                    if width > 1 and height > 1:
                        # Update scan line position
                        y_pos = self.scan_line_position
                        
                        # Update line
                        canvas.coords(line_id, 0, y_pos, width, y_pos)
                        
                        # Update text position
                        text_x = width // 2
                        text_y = min(y_pos + 20, height - 20)
                        canvas.coords(text_id, text_x, text_y)
        
        # Update position
        self.scan_line_position += self.scan_direction * 8
        
        # Bounce at edges
        if self.scan_line_position <= 0:
            self.scan_direction = 1
            self.scan_line_position = 0
        elif hasattr(self, 'ref_overlay'):
            canvas = getattr(self, 'ref_overlay')
            canvas.update_idletasks()
            height = canvas.winfo_height()
            if height > 1 and self.scan_line_position >= height:
                self.scan_direction = -1
                self.scan_line_position = height
        
        # Continue animation
        if self.scanning_active:
            self.animation_after_id = self.root.after(50, self.animate_scan_line)
        """Handle model selection change"""
        method = self.model_var.get()
        if method.startswith("face_recognition"):
            self.tolerance_frame.pack(fill='x', pady=(0, 10))
        else:
            self.tolerance_frame.pack_forget()
    
    def log_result(self, message):
        """Log a result message"""
        self.results_text.config(state='normal')
        timestamp = time.strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.results_text.config(state='disabled')
    
    def upload_reference_image(self):
        """Upload reference image for comparison"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                method = self.model_var.get()
                filename = os.path.basename(file_path)
                
                if method.startswith("DeepFace") and DEEPFACE_AVAILABLE:
                    # Use DeepFace
                    try:
                        # Test if face can be detected
                        model_name = method.split("-")[1] + "-Face" if method.split("-")[1] != "VGG" else "VGG-Face"
                        test_result = DeepFace.represent(file_path, model_name=model_name, enforce_detection=True)
                        
                        # Store the file path for DeepFace comparisons
                        self.reference_image_path = file_path
                        
                        # Load image for display
                        self.reference_image = cv2.imread(file_path)
                        rgb_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
                        self.display_image(rgb_image, self.ref_image_label)
                        
                        # Update status
                        self.reference_loaded = True
                        self.update_recognition_status()
                        
                        self.log_result(f"✅ Reference face loaded: {filename}")
                        self.log_result(f"🧠 Using DeepFace with {model_name} model")
                        self.log_result(f"🔍 Face representation extracted successfully")
                        
                    except Exception as e:
                        self.log_result(f"❌ DeepFace error: {str(e)}")
                        messagebox.showerror("Error", f"DeepFace could not process the image: {str(e)}")
                        return
                
                elif method.startswith("face_recognition") and FACE_RECOGNITION_AVAILABLE:
                    # Use face_recognition library
                    self.reference_image = face_recognition.load_image_file(file_path)
                    model_type = method.split("-")[1]
                    face_encodings = face_recognition.face_encodings(self.reference_image, model=model_type)
                    
                    if face_encodings:
                        self.reference_encoding = face_encodings[0]
                        
                        # Display image
                        self.display_image(self.reference_image, self.ref_image_label)
                        
                        # Update status
                        self.reference_loaded = True
                        self.update_recognition_status()
                        
                        self.log_result(f"✅ Reference face loaded: {filename}")
                        self.log_result(f"🔍 Face encoding generated successfully")
                        self.log_result(f"📐 Using {model_type.upper()} detection model")
                        
                    else:
                        self.log_result(f"❌ No face detected in reference image")
                        messagebox.showerror("Error", "No face detected in the selected image")
                        return
                
                else:
                    # Use OpenCV fallback
                    self.reference_image = cv2.imread(file_path)
                    gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        # Take the first (largest) face
                        x, y, w, h = faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Extract ORB features from face region
                        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
                        
                        if descriptors is not None:
                            self.reference_features = descriptors
                            
                            # Convert BGR to RGB for display
                            rgb_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
                            self.display_image(rgb_image, self.ref_image_label)
                            
                            # Update status
                            self.reference_loaded = True
                            self.update_recognition_status()
                            
                            self.log_result(f"✅ Reference face loaded: {filename}")
                            self.log_result(f"🔍 Face detected using OpenCV Haar Cascades")
                            self.log_result(f"📐 Extracted {len(descriptors)} ORB features")
                        else:
                            self.log_result(f"❌ Could not extract features from detected face")
                            messagebox.showerror("Error", "Could not extract features from face")
                            return
                    else:
                        self.log_result(f"❌ No face detected in reference image")
                        messagebox.showerror("Error", "No face detected in the selected image")
                        return
                        
            except Exception as e:
                self.log_result(f"❌ Error loading reference image: {str(e)}")
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def upload_test_image(self):
        """Upload test image for recognition"""
        method = self.model_var.get()
        
        # Check if reference is loaded based on method
        if method.startswith("DeepFace") and not self.reference_image_path:
            messagebox.showwarning("Warning", "Please upload a reference image first")
            return
        elif method.startswith("face_recognition") and not self.reference_encoding:
            messagebox.showwarning("Warning", "Please upload a reference image first")
            return
        elif method.startswith("OpenCV") and not self.reference_features:
            messagebox.showwarning("Warning", "Please upload a reference image first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Store test image path
                self.test_image_path = file_path
                filename = os.path.basename(file_path)
                
                # Load and display image based on method
                method = self.model_var.get()
                
                if method.startswith("DeepFace") and DEEPFACE_AVAILABLE:
                    # Display image
                    test_image = cv2.imread(file_path)
                    rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                    self.display_image(rgb_image, self.test_image_label)
                    
                    # Update status
                    self.test_loaded = True
                    self.update_recognition_status()
                    
                    self.log_result(f"📷 Test image loaded: {filename}")
                    self.log_result(f"🎯 Ready for DeepFace recognition")
                    
                elif method.startswith("face_recognition") and FACE_RECOGNITION_AVAILABLE:
                    # Load image for face_recognition
                    self.test_image = face_recognition.load_image_file(file_path)
                    
                    # Display image
                    self.display_image(self.test_image, self.test_image_label)
                    
                    # Update status
                    self.test_loaded = True
                    self.update_recognition_status()
                    
                    self.log_result(f"📷 Test image loaded: {filename}")
                    self.log_result(f"🎯 Ready for face_recognition analysis")
                else:
                    # OpenCV fallback
                    test_image = cv2.imread(file_path)
                    self.test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Store as RGB
                    
                    # Display image
                    self.display_image(self.test_image, self.test_image_label)
                    
                    # Update status
                    self.test_loaded = True
                    self.update_recognition_status()
                    
                    self.log_result(f"📷 Test image loaded: {filename}")
                    self.log_result(f"🎯 Ready for OpenCV ORB analysis")
                
            except Exception as e:
                self.log_result(f"❌ Error processing test image: {str(e)}")
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def perform_deepface_recognition(self, test_image_path, source_name="Test Image"):
        """Perform face recognition using DeepFace"""
        try:
            method = self.model_var.get()
            model_name = method.split("-")[1]
            
            # Adjust model name for DeepFace format
            if model_name == "VGG":
                model_name = "VGG-Face"
            elif model_name == "Facenet":
                model_name = "Facenet"
            elif model_name == "OpenFace":
                model_name = "OpenFace"
            elif model_name == "ArcFace":
                model_name = "ArcFace"
            
            self.log_result(f"🔍 Analyzing: {source_name}")
            self.log_result(f"🧠 Using DeepFace with {model_name} model")
            
            # Perform face verification
            result = DeepFace.verify(
                img1_path=self.reference_image_path,
                img2_path=test_image_path,
                model_name=model_name,
                enforce_detection=True,
                distance_metric='cosine'
            )
            
            is_match = result['verified']
            distance = result['distance']
            threshold = result['threshold']
            
            # Calculate confidence (inverse of normalized distance)
            confidence = max(0, (1 - (distance / threshold)) * 100)
            
            # Update latest result display
            method_name = f"DeepFace ({model_name})"
            details = f"Distance: {distance:.4f} | Threshold: {threshold:.4f}"
            self.update_latest_result(is_match, confidence, method_name, details)
            
            # Log results
            match_status = "✅ MATCH" if is_match else "❌ NO MATCH"
            self.log_result(f"🔍 Analyzing: {source_name}")
            self.log_result(f"👤 Face Verification: {match_status}")
            self.log_result(f"📊 Distance: {distance:.4f} (threshold: {threshold:.4f})")
            self.log_result(f"🎯 Confidence: {confidence:.2f}%")
            self.log_result(f"📐 Model: {model_name}")
            
            if is_match:
                self.log_result(f"🎉 Recognition successful!")
            else:
                self.log_result(f"🚫 Face does not match reference")
            
            self.log_result("─" * 40)
            
        except Exception as e:
            self.log_result(f"❌ DeepFace recognition error: {str(e)}")
            # If DeepFace fails, try to provide more specific error information
            if "Face could not be detected" in str(e):
                self.log_result("💡 Tip: Ensure the image contains a clear, front-facing face")
            elif "enforce_detection" in str(e):
                self.log_result("💡 Tip: Try with a higher quality image or different angle")
    
    def perform_recognition(self, test_image, source_name="Test Image", model_type=None):
        """Perform face recognition on test image using face_recognition library"""
        try:
            if model_type is None:
                method = self.model_var.get()
                model_type = method.split("-")[1] if "-" in method else "hog"
            
            # Find faces in test image
            face_locations = face_recognition.face_locations(test_image, model=model_type)
            face_encodings = face_recognition.face_encodings(test_image, face_locations, model=model_type)
            
            self.log_result(f"🔍 Analyzing: {source_name}")
            self.log_result(f"👥 Found {len(face_locations)} face(s) in image")
            
            if not face_encodings:
                self.log_result("❌ No faces detected for comparison")
                return
            
            # Compare each face
            for i, face_encoding in enumerate(face_encodings):
                # Calculate distance
                face_distance = face_recognition.face_distance([self.reference_encoding], face_encoding)[0]
                
                # Determine if it's a match
                is_match = face_distance <= self.tolerance_var.get()
                
                # Calculate confidence percentage
                confidence = max(0, (1.0 - face_distance) * 100)
                
                # Update latest result display (for first face)
                if i == 0:
                    method_name = f"face_recognition ({model_type})"
                    details = f"Distance: {face_distance:.4f} | Tolerance: {self.tolerance_var.get()}"
                    self.update_latest_result(is_match, confidence, method_name, details)
                
                # Log results
                match_status = "✅ MATCH" if is_match else "❌ NO MATCH"
                self.log_result(f"👤 Face {i+1}: {match_status}")
                self.log_result(f"📊 Distance: {face_distance:.4f} (threshold: {self.tolerance_var.get()})")
                self.log_result(f"🎯 Confidence: {confidence:.2f}%")
                
                if is_match:
                    self.log_result(f"🎉 Recognition successful!")
                else:
                    self.log_result(f"🚫 Face does not match reference")
                
                self.log_result("─" * 40)
                
        except Exception as e:
            self.log_result(f"❌ Recognition error: {str(e)}")
    
    def perform_opencv_recognition(self, test_image, source_name="Test Image"):
        """Perform face recognition using OpenCV and ORB features"""
        try:
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            self.log_result(f"🔍 Analyzing: {source_name}")
            self.log_result(f"👥 Found {len(faces)} face(s) in image")
            
            if len(faces) == 0:
                self.log_result("❌ No faces detected for comparison")
                return
            
            # Compare each detected face
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = gray[y:y+h, x:x+w]
                
                # Extract ORB features
                keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
                
                if descriptors is not None:
                    # Match features with reference
                    matches = self.matcher.match(self.reference_features, descriptors)
                    
                    # Sort matches by distance
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Calculate match score
                    good_matches = [m for m in matches if m.distance < 50]  # Threshold for good matches
                    match_score = len(good_matches) / max(len(matches), 1) * 100
                    
                    # Determine if it's a match (more than 30% good matches)
                    is_match = match_score > 30
                    
                    # Update latest result display (for first face)
                    if i == 0:
                        method_name = "OpenCV ORB"
                        details = f"Features: {len(good_matches)}/{len(matches)} matches"
                        self.update_latest_result(is_match, match_score, method_name, details)
                    
                    # Log results
                    match_status = "✅ MATCH" if is_match else "❌ NO MATCH"
                    self.log_result(f"👤 Face {i+1}: {match_status}")
                    self.log_result(f"📊 Feature matches: {len(good_matches)}/{len(matches)}")
                    self.log_result(f"🎯 Match score: {match_score:.2f}%")
                    
                    if is_match:
                        self.log_result(f"🎉 Recognition successful!")
                    else:
                        self.log_result(f"🚫 Face does not match reference")
                else:
                    self.log_result(f"👤 Face {i+1}: ❌ Could not extract features")
                
                self.log_result("─" * 40)
                
        except Exception as e:
            self.log_result(f"❌ Recognition error: {str(e)}")
    
    def display_image(self, image_array, label_widget):
        """Display image in label widget"""
        try:
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Check if it's already RGB or BGR
                pil_image = Image.fromarray(image_array)
            else:
                pil_image = Image.fromarray(image_array)
            
            # Resize to fit label while maintaining aspect ratio
            label_width = label_widget.winfo_width() or 400
            label_height = label_widget.winfo_height() or 300
            
            pil_image.thumbnail((label_width-20, label_height-20), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            label_widget.configure(image=photo, text="")
            label_widget.image = photo  # Keep a reference
            
        except Exception as e:
            self.log_result(f"❌ Display error: {str(e)}")
    
    def start_camera(self):
        """Start live camera recognition"""
        method = self.model_var.get()
        
        # Check if reference is loaded based on method
        if method.startswith("DeepFace") and not self.reference_image_path:
            messagebox.showwarning("Warning", "Please upload a reference image first")
            return
        elif method.startswith("face_recognition") and not self.reference_encoding:
            messagebox.showwarning("Warning", "Please upload a reference image first")
            return
        elif method.startswith("OpenCV") and not self.reference_features:
            messagebox.showwarning("Warning", "Please upload a reference image first")
            return
        
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.is_camera_running = True
            self.start_camera_btn.config(state='disabled')
            self.stop_camera_btn.config(state='normal')
            
            if method.startswith("DeepFace"):
                method_name = f"DeepFace ({method.split('-')[1]})"
            elif method.startswith("face_recognition"):
                method_name = f"face_recognition ({method.split('-')[1]})"
            else:
                method_name = "OpenCV ORB"
                
            self.log_result("📹 Live camera started")
            self.log_result(f"🔴 Real-time face recognition active ({method_name})")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            self.log_result(f"❌ Camera error: {str(e)}")
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop live camera"""
        self.is_camera_running = False
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.start_camera_btn.config(state='normal')
        self.stop_camera_btn.config(state='disabled')
        
        self.log_result("⏹️ Camera stopped")
    
    def camera_loop(self):
        """Camera capture loop"""
        frame_count = 0
        temp_frame_path = "temp_frame.jpg"
        
        while self.is_camera_running:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                # Process every 30th frame for performance (especially for DeepFace)
                if frame_count % 30 == 0:
                    method = self.model_var.get()
                    
                    if method.startswith("DeepFace") and DEEPFACE_AVAILABLE:
                        # Save frame temporarily for DeepFace
                        cv2.imwrite(temp_frame_path, frame)
                        
                        # Perform recognition (in a separate thread to avoid blocking)
                        threading.Thread(
                            target=lambda: self.perform_deepface_recognition(temp_frame_path, "Live Camera"),
                            daemon=True
                        ).start()
                        
                        # Display frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.root.after(0, lambda f=rgb_frame: self.display_image(f, self.test_image_label))
                        
                    elif method.startswith("face_recognition") and FACE_RECOGNITION_AVAILABLE:
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Perform recognition
                        model_type = method.split("-")[1]
                        self.root.after(0, lambda f=rgb_frame: self.perform_recognition(f, "Live Camera", model_type))
                        
                        # Display frame
                        self.root.after(0, lambda f=rgb_frame: self.display_image(f, self.test_image_label))
                    else:
                        # Use OpenCV approach
                        self.root.after(0, lambda f=frame: self.perform_opencv_recognition(f, "Live Camera"))
                        
                        # Display frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.root.after(0, lambda f=rgb_frame: self.display_image(f, self.test_image_label))
                else:
                    # Just display the frame without processing
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.root.after(0, lambda f=rgb_frame: self.display_image(f, self.test_image_label))
                
                frame_count += 1
                time.sleep(0.1)  # Limit FPS
                
            except Exception as e:
                self.log_result(f"❌ Camera loop error: {str(e)}")
                break
        
        # Cleanup temp file
        try:
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        except:
            pass
    
    def __del__(self):
        """Cleanup on exit"""
        if self.video_capture:
            self.video_capture.release()
        
        # Stop any running animation
        self.scanning_active = False
        if hasattr(self, 'animation_after_id') and self.animation_after_id:
            try:
                self.root.after_cancel(self.animation_after_id)
            except:
                pass

def main():
    """Main function"""
    if DEEPFACE_AVAILABLE:
        print("✅ DeepFace library loaded successfully")
        print("🧠 Available models: VGG-Face, Facenet, OpenFace, ArcFace")
    elif FACE_RECOGNITION_AVAILABLE:
        print("✅ face_recognition library loaded successfully")
        print("💡 Install DeepFace for state-of-the-art models")
    else:
        print("⚠️ Advanced face recognition libraries not available")
        print("🔄 Using OpenCV-based face detection and ORB feature matching")
        print("💡 For better accuracy, install:")
        print("   pip install deepface tensorflow")
        print("   pip install dlib face_recognition")
    
    root = tk.Tk()
    app = FaceRecognitionTest(root)
    
    # Handle window close
    def on_closing():
        if app.is_camera_running:
            app.stop_camera()
        if hasattr(app, 'scanning_active'):
            app.stop_scanning_animation()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()