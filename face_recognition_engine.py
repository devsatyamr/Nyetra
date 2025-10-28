"""
Enhanced Face Recognition Engine with Advanced Features
Features:
- Multi-model support (VGG-Face, Facenet, OpenFace, ArcFace)
- Real-time face extraction and comparison
- Queue-based architecture with threading
- Popup management (max 3 popups)
- Automatic scan stopping when matches found
- PDF report generation
- Advanced matching with cosine similarity
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available, PDF generation disabled")
import tempfile
import base64
import io

# DeepFace imports with fallback
try:
    from deepface import DeepFace
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available, using OpenCV fallback")

# Face recognition library (not needed - using DeepFace instead)
FACE_RECOGNITION_AVAILABLE = False


class MatchResult:
    """Class to store face match results"""
    def __init__(self, frame_number: int, timestamp: float, confidence: float, 
                 frame_image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                 video_path: str):
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.confidence = confidence
        self.frame_image = frame_image.copy()
        self.face_bbox = face_bbox
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        
        # Extract face image from frame using bbox
        x, y, w, h = face_bbox
        self.face_image = frame_image[y:y+h, x:x+w].copy()


class FaceRecognitionEngine:
    """Advanced Face Recognition Engine with popup management and PDF generation"""
    
    def __init__(self):
        # Core attributes
        self.model = "VGG-Face"  # Default model
        self.reference_embedding = None
        self.reference_image_path = None
        self.reference_image = None  # Store actual image data for PDF generation
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Threading and queue management
        self.extraction_queue = queue.Queue(maxsize=100)
        self.comparison_queue = queue.Queue(maxsize=50)
        self.extraction_thread = None
        self.comparison_thread = None
        self.stop_scanning_flag = False
        
        # Match storage and popup management
        self.matches_found: List[MatchResult] = []
        self.popups_shown = 0
        self.max_popups = 3
        self.popup_windows = []
        
        # Detection parameters - Using sensitive preset for better accuracy
        self.confidence_threshold = 0.3  # Sensitive preset
        self.similarity_threshold = 0.25  # Sensitive preset for cosine similarity
        
        # Video processing
        self.current_video_path = None
        self.total_frames_processed = 0
        
        # PDF generation
        self.pdf_temp_dir = tempfile.mkdtemp()
        
        # Initialize face detection fallback methods
        self._init_fallback_methods()
        
        logging.info("Face Recognition Engine initialized")
    
    def _init_fallback_methods(self):
        """Initialize fallback face detection methods"""
        # ORB feature detector for pattern matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # SIFT detector if available
        try:
            self.sift = cv2.SIFT_create()
            self.sift_available = True
        except AttributeError:
            self.sift_available = False
    
    def set_model(self, model: str) -> bool:
        """Set the face recognition model (same logic as test file)"""
        supported_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace"]
        if model in supported_models:
            self.model = model
            logging.info(f"Model set to: {model}")
            return True
        else:
            logging.error(f"Unsupported model: {model}")
            return False
    
    def set_thresholds(self, confidence_threshold: float = None, similarity_threshold: float = None):
        """Set detection thresholds for better accuracy control"""
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.1, min(1.0, confidence_threshold))  # Clamp between 0.1 and 1.0
            logging.info(f"Confidence threshold set to: {self.confidence_threshold}")
        
        if similarity_threshold is not None:
            self.similarity_threshold = max(0.1, min(1.0, similarity_threshold))  # Clamp between 0.1 and 1.0
            logging.info(f"Similarity threshold set to: {self.similarity_threshold}")
    
    def get_thresholds(self) -> dict:
        """Get current threshold values"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'similarity_threshold': self.similarity_threshold
        }
    
    def set_accuracy_preset(self, preset: str):
        """Set predefined accuracy presets for different use cases"""
        presets = {
            'strict': {
                'confidence_threshold': 0.7,
                'similarity_threshold': 0.6,
                'description': 'High precision, fewer false positives'
            },
            'balanced': {
                'confidence_threshold': 0.5,
                'similarity_threshold': 0.4,
                'description': 'Good balance of precision and recall'
            },
            'sensitive': {
                'confidence_threshold': 0.3,
                'similarity_threshold': 0.25,
                'description': 'High recall, more matches but some false positives'
            },
            'very_sensitive': {
                'confidence_threshold': 0.2,
                'similarity_threshold': 0.15,
                'description': 'Maximum sensitivity, catches most matches'
            }
        }
        
        if preset in presets:
            config = presets[preset]
            self.set_thresholds(config['confidence_threshold'], config['similarity_threshold'])
            logging.info(f"Applied {preset} preset: {config['description']}")
            return True
        else:
            logging.error(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
            return False
    
    def get_available_presets(self) -> dict:
        """Get available accuracy presets"""
        return {
            'strict': 'High precision, fewer false positives',
            'balanced': 'Good balance of precision and recall',
            'sensitive': 'High recall, more matches but some false positives',
            'very_sensitive': 'Maximum sensitivity, catches most matches'
        }
    
    def _get_deepface_model_name(self, model: str) -> str:
        """Convert GUI model name to DeepFace format (same as test file)"""
        if model == "VGG-Face":
            return "VGG-Face"
        elif model == "Facenet":
            return "Facenet"
        elif model == "OpenFace":
            return "OpenFace"
        elif model == "ArcFace":
            return "ArcFace"
        else:
            return "VGG-Face"  # Default fallback
    
    def load_reference_image(self, image_path: str) -> bool:
        """Load and process reference image for comparison using DeepFace"""
        try:
            if not os.path.exists(image_path):
                logging.error(f"Reference image not found: {image_path}")
                return False
            
            self.reference_image_path = image_path
            
            # Load the actual image data for PDF generation
            # Normalize path for cross-platform compatibility
            normalized_path = os.path.normpath(image_path)
            self.reference_image = cv2.imread(normalized_path)
            if self.reference_image is None:
                logging.error(f"Failed to load reference image: {image_path} (normalized: {normalized_path})")
                return False
            
            # Debug: Confirm image loaded
            logging.info(f"Reference image loaded successfully - Shape: {self.reference_image.shape}, Path: {normalized_path}")
            
            # Use DeepFace verification approach (like in test file)
            if DEEPFACE_AVAILABLE:
                try:
                    # Test if face can be detected in reference image
                    model_name = self._get_deepface_model_name(self.model)
                    test_result = DeepFace.represent(
                        img_path=image_path,
                        model_name=model_name,
                        enforce_detection=True
                    )
                    
                    # Store the file path for DeepFace comparisons (verification method)
                    self.reference_image_path = image_path
                    logging.info(f"Reference image loaded with {self.model} model using DeepFace verification")
                    return True
                    
                except Exception as e:
                    logging.warning(f"DeepFace failed: {e}, trying OpenCV fallback")
            
            # OpenCV fallback with feature extraction
            return self._load_reference_opencv(image_path)
            
        except Exception as e:
            logging.error(f"Failed to load reference image: {e}")
            return False
    
    def _load_reference_opencv(self, image_path: str) -> bool:
        """Load reference image using OpenCV methods"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                logging.warning("No faces detected in reference image")
                return False
            
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_img = gray[y:y+h, x:x+w]
            
            # Extract ORB features
            keypoints, descriptors = self.orb.detectAndCompute(face_img, None)
            if descriptors is not None:
                self.reference_embedding = descriptors
                logging.info("Reference image loaded with OpenCV ORB features")
                return True
            
            # If ORB fails, use simple histogram
            hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
            self.reference_embedding = hist.flatten()
            logging.info("Reference image loaded with histogram features")
            return True
            
        except Exception as e:
            logging.error(f"OpenCV reference loading failed: {e}")
            return False
    
    def start_real_time_recognition(self, video_path: str) -> bool:
        """Start real-time face recognition on video"""
        try:
            logging.info(f"Attempting to start recognition for: {video_path}")
            
            if not os.path.exists(video_path):
                logging.error(f"Video file not found: {video_path}")
                return False
            
            if self.reference_image_path is None:
                logging.error("No reference image loaded")
                return False
            
            # Try to open the video file to make sure it's valid
            import cv2
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                logging.error(f"Cannot open video file: {video_path}")
                test_cap.release()
                return False
            test_cap.release()
            logging.info(f"Video file opened successfully: {video_path}")
            
            self.current_video_path = video_path
            self.stop_scanning_flag = False
            self.matches_found.clear()
            self.popups_shown = 0
            self._close_all_popups()
            
            # Start processing threads
            self.extraction_thread = threading.Thread(
                target=self._extract_faces_from_video,
                args=(video_path,),
                daemon=True
            )
            self.comparison_thread = threading.Thread(
                target=self._compare_faces,
                daemon=True
            )
            
            self.extraction_thread.start()
            self.comparison_thread.start()
            
            logging.info(f"Started real-time recognition threads for: {video_path}")
            return True
            
        except Exception as e:
            import traceback
            logging.error(f"Failed to start real-time recognition: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _extract_faces_from_video(self, video_path: str):
        """Extract faces from video frames (runs in separate thread)"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Could not open video: {video_path}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_number = 0
            
            while cap.isOpened() and not self.stop_scanning_flag:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame for performance
                if frame_number % 10 == 0:
                    timestamp = frame_number / fps
                    
                    # Detect faces in frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Add faces to queue for comparison
                    for face_bbox in faces:
                        if not self.extraction_queue.full():
                            face_data = {
                                'frame': frame.copy(),
                                'face_bbox': face_bbox,
                                'frame_number': frame_number,
                                'timestamp': timestamp,
                                'video_path': video_path
                            }
                            self.extraction_queue.put(face_data)
                
                frame_number += 1
                self.total_frames_processed = frame_number
                
                # Check if we should stop (max popups reached)
                if self.popups_shown >= self.max_popups:
                    logging.info("Max popups reached, stopping extraction")
                    self.stop_scanning_flag = True
                    break
            
            cap.release()
            logging.info(f"Face extraction completed for {video_path}")
            
        except Exception as e:
            logging.error(f"Face extraction error: {e}")
    
    def _compare_faces(self):
        """Compare extracted faces with reference (runs in separate thread)"""
        try:
            while not self.stop_scanning_flag or not self.extraction_queue.empty():
                try:
                    # Get face data from queue
                    face_data = self.extraction_queue.get(timeout=1.0)
                    
                    # Extract face region
                    frame = face_data['frame']
                    x, y, w, h = face_data['face_bbox']
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Compare with reference
                    similarity = self._compare_face_with_reference(face_img)
                    
                    # Debug: Show comparison results
                    logging.info(f"Face comparison - Similarity: {similarity:.4f}, Threshold: {self.similarity_threshold:.4f}, Match: {similarity >= self.similarity_threshold}")
                    
                    if similarity >= self.similarity_threshold:
                        # Create match result
                        match = MatchResult(
                            frame_number=face_data['frame_number'],
                            timestamp=face_data['timestamp'],
                            confidence=similarity,
                            frame_image=frame,
                            face_bbox=face_data['face_bbox'],
                            video_path=face_data['video_path']
                        )
                        
                        self.matches_found.append(match)
                        
                        # Show popup if within limit
                        if self.popups_shown < self.max_popups:
                            self._show_match_popup(match)
                            self.popups_shown += 1
                        
                        # Stop scanning if max popups reached
                        if self.popups_shown >= self.max_popups:
                            logging.info("Max popups reached, stopping scan")
                            self.stop_scanning_flag = True
                            break
                    
                    self.extraction_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Face comparison error: {e}")
                    continue
            
            logging.info("Face comparison completed")
            
        except Exception as e:
            logging.error(f"Face comparison thread error: {e}")
    
    def _compare_face_with_reference(self, face_img: np.ndarray) -> float:
        """Compare face image with reference using DeepFace verification (like test file)"""
        try:
            if face_img is None or face_img.size == 0:
                return 0.0
            
            # Use DeepFace verification approach (same as test file)
            if DEEPFACE_AVAILABLE and self.reference_image_path:
                try:
                    # Save face to temporary file
                    temp_path = os.path.join(self.pdf_temp_dir, f"temp_face_{time.time()}.jpg")
                    cv2.imwrite(temp_path, face_img)
                    
                    # Perform face verification using DeepFace (same logic as test file)
                    model_name = self._get_deepface_model_name(self.model)
                    result = DeepFace.verify(
                        img1_path=self.reference_image_path,
                        img2_path=temp_path,
                        model_name=model_name,
                        enforce_detection=False,
                        distance_metric='cosine'  # Same as test file
                    )
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    # Calculate confidence (same logic as test file)
                    is_match = result['verified']
                    distance = result['distance']
                    threshold = result['threshold']
                    
                    # Calculate confidence (inverse of normalized distance) - same as test file
                    confidence = max(0, (1 - (distance / threshold)) * 100) / 100.0  # Normalize to 0-1
                    
                    # Debug: Print comparison results for tuning
                    logging.info(f"DeepFace comparison - Distance: {distance:.4f}, Threshold: {threshold:.4f}, Confidence: {confidence:.4f}, Is_Match: {is_match}")
                    
                    return confidence
                        
                except Exception as e:
                    logging.warning(f"DeepFace comparison failed: {e}")
            
            # OpenCV fallback comparison
            return self._compare_opencv_features(face_img)
            
        except Exception as e:
            logging.error(f"Face comparison error: {e}")
            return 0.0
    
    def _compare_opencv_features(self, face_img: np.ndarray) -> float:
        """Compare faces using OpenCV features (fallback when DeepFace fails)"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Try ORB feature matching (if we have reference features from OpenCV loading)
            if hasattr(self, 'reference_embedding') and isinstance(self.reference_embedding, np.ndarray) and len(self.reference_embedding.shape) == 2:
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
                if descriptors is not None:
                    matches = self.matcher.match(self.reference_embedding, descriptors)
                    if len(matches) > 10:  # Need sufficient matches
                        good_matches = sorted(matches, key=lambda x: x.distance)[:20]
                        avg_distance = np.mean([m.distance for m in good_matches])
                        return max(0, 1.0 - (avg_distance / 100.0))  # Normalize distance
            
            # Histogram comparison fallback
            elif hasattr(self, 'reference_embedding') and isinstance(self.reference_embedding, np.ndarray) and len(self.reference_embedding.shape) == 1:
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                correlation = cv2.compareHist(hist.flatten(), self.reference_embedding, cv2.HISTCMP_CORREL)
                return max(0, correlation)
            
            # If we only have reference_image_path but DeepFace failed, try basic template matching
            elif self.reference_image_path:
                try:
                    ref_img = cv2.imread(self.reference_image_path, cv2.IMREAD_GRAYSCALE)
                    ref_faces = self.face_cascade.detectMultiScale(ref_img, 1.1, 4)
                    if len(ref_faces) > 0:
                        x, y, w, h = ref_faces[0]  # Take first face
                        ref_face = ref_img[y:y+h, x:x+w]
                        
                        # Resize both faces to same size for comparison
                        face_resized = cv2.resize(gray, (100, 100))
                        ref_resized = cv2.resize(ref_face, (100, 100))
                        
                        # Template matching
                        result = cv2.matchTemplate(face_resized, ref_resized, cv2.TM_CCOEFF_NORMED)
                        return float(np.max(result))
                except:
                    pass
            
            return 0.0
            
        except Exception as e:
            logging.error(f"OpenCV feature comparison error: {e}")
            return 0.0
    
    def _show_match_popup(self, match: MatchResult):
        """Show popup window with match details"""
        try:
            popup = tk.Toplevel()
            popup.title(f"Face Match Found! ({self.popups_shown + 1}/{self.max_popups})")
            popup.geometry("400x500")
            popup.configure(bg='#1a1a1a')
            popup.resizable(False, False)
            
            # Make popup stay on top
            popup.attributes('-topmost', True)
            
            # Header
            header_frame = tk.Frame(popup, bg='#28a745', height=60)
            header_frame.pack(fill='x', padx=0, pady=0)
            header_frame.pack_propagate(False)
            
            title_label = tk.Label(
                header_frame,
                text="🎯 FACE MATCH DETECTED",
                font=('Segoe UI', 14, 'bold'),
                bg='#28a745',
                fg='white'
            )
            title_label.pack(expand=True)
            
            # Content frame
            content_frame = tk.Frame(popup, bg='#1a1a1a')
            content_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            # Match info
            info_text = f"""📹 Video: {match.video_name}
⏰ Time: {match.timestamp:.1f}s (Frame {match.frame_number})
🎯 Confidence: {match.confidence*100:.1f}%
🧠 Model: {self.model}
📊 Match #{len(self.matches_found)} of {self.max_popups}"""
            
            info_label = tk.Label(
                content_frame,
                text=info_text,
                font=('Segoe UI', 10),
                bg='#1a1a1a',
                fg='white',
                justify='left'
            )
            info_label.pack(pady=(0, 15))
            
            # Image display
            try:
                # Extract and display the face region
                x, y, w, h = match.face_bbox
                face_region = match.frame_image[y:y+h, x:x+w]
                
                # Convert to RGB and resize
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                pil_image = pil_image.resize((200, 200), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(pil_image)
                
                image_label = tk.Label(content_frame, image=photo, bg='#1a1a1a')
                image_label.image = photo  # Keep reference
                image_label.pack(pady=(0, 15))
                
            except Exception as e:
                logging.error(f"Error displaying match image: {e}")
                error_label = tk.Label(
                    content_frame,
                    text="❌ Could not display image",
                    font=('Segoe UI', 10),
                    bg='#1a1a1a',
                    fg='#dc3545'
                )
                error_label.pack()
            
            # Buttons frame
            button_frame = tk.Frame(content_frame, bg='#1a1a1a')
            button_frame.pack(fill='x', pady=(15, 0))
            
            close_btn = tk.Button(
                button_frame,
                text="Close",
                font=('Segoe UI', 10, 'bold'),
                bg='#dc3545',
                fg='white',
                relief='flat',
                padx=20,
                pady=8,
                command=popup.destroy
            )
            close_btn.pack(side='right', padx=(10, 0))
            
            # Store popup reference
            self.popup_windows.append(popup)
            
            # Auto-close after 30 seconds
            popup.after(30000, lambda: self._safe_destroy_popup(popup))
            
            logging.info(f"Match popup shown: {match.video_name} at {match.timestamp:.1f}s")
            
        except Exception as e:
            logging.error(f"Error showing match popup: {e}")
    
    def _safe_destroy_popup(self, popup):
        """Safely destroy popup window"""
        try:
            if popup.winfo_exists():
                popup.destroy()
            if popup in self.popup_windows:
                self.popup_windows.remove(popup)
        except:
            pass
    
    def _close_all_popups(self):
        """Close all open popup windows"""
        for popup in self.popup_windows[:]:  # Copy list to avoid modification during iteration
            self._safe_destroy_popup(popup)
        self.popup_windows.clear()
    
    def stop_recognition(self):
        """Stop the face recognition process"""
        self.stop_scanning_flag = True
        
        # Wait for threads to finish
        if self.extraction_thread and self.extraction_thread.is_alive():
            self.extraction_thread.join(timeout=2.0)
        if self.comparison_thread and self.comparison_thread.is_alive():
            self.comparison_thread.join(timeout=2.0)
        
        logging.info("Face recognition stopped")
    
    def generate_pdf_report(self, output_path: str = None) -> str:
        """Generate PDF report of face recognition results using matplotlib"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logging.error("Matplotlib not available for PDF generation")
                return None
                
            if not self.matches_found:
                logging.warning("No matches found to generate report")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"face_recognition_report_{timestamp}.pdf"
            
            # Create PDF with matplotlib
            with PdfPages(output_path) as pdf:
                # Page 1: Summary and images
                fig = plt.figure(figsize=(8.5, 11))  # Letter size
                gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 2, 2, 1])
                
                # Title and summary
                ax_title = fig.add_subplot(gs[0, :])
                ax_title.axis('off')
                ax_title.text(0.5, 0.8, 'Face Recognition Analysis Report', 
                             ha='center', va='top', fontsize=20, fontweight='bold')
                
                # Summary statistics
                total_matches = len(self.matches_found)
                videos_processed = len(set(match.video_path for match in self.matches_found))
                avg_confidence = np.mean([match.confidence for match in self.matches_found]) * 100
                analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                summary_text = f"""Analysis Summary:
• Total Face Matches Found: {total_matches}
• Videos Processed: {videos_processed}
• Average Confidence: {avg_confidence:.1f}%
• Analysis Date: {analysis_date}
• Model Used: {self.model}"""
                
                ax_title.text(0.1, 0.4, summary_text, ha='left', va='top', fontsize=12)
                
                # Reference image
                ax_ref = fig.add_subplot(gs[1, 0])
                ax_ref.set_title('Reference Image', fontweight='bold')
                
                # Debug: Check reference image status
                logging.info(f"PDF Generation - Reference image available: {hasattr(self, 'reference_image') and self.reference_image is not None}")
                if hasattr(self, 'reference_image'):
                    logging.info(f"PDF Generation - Reference image shape: {self.reference_image.shape if self.reference_image is not None else 'None'}")
                
                if hasattr(self, 'reference_image') and self.reference_image is not None:
                    try:
                        # Convert BGR to RGB for matplotlib
                        reference_rgb = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
                        ax_ref.imshow(reference_rgb)
                        logging.info("PDF Generation - Reference image displayed successfully")
                    except Exception as e:
                        logging.error(f"PDF Generation - Error displaying reference image: {e}")
                        ax_ref.text(0.5, 0.5, f'Reference Image\nError: {str(e)}', 
                                   ha='center', va='center', fontsize=10)
                else:
                    logging.warning("PDF Generation - Reference image not available")
                    ax_ref.text(0.5, 0.5, 'Reference Image\nNot Available', 
                               ha='center', va='center', fontsize=12)
                ax_ref.axis('off')
                
                # Top 3 matched images
                matches_to_show = self.matches_found[:3]  # Show first 3 matches
                
                for i, match in enumerate(matches_to_show):
                    if i >= 3:
                        break
                        
                    # Position for matched images
                    if i == 0:
                        ax_match = fig.add_subplot(gs[1, 1])
                    elif i == 1:
                        ax_match = fig.add_subplot(gs[2, 0])
                    else:
                        ax_match = fig.add_subplot(gs[2, 1])
                    
                    ax_match.set_title(f'Match {i+1}: {match.confidence*100:.1f}% confidence', 
                                      fontweight='bold')
                    
                    # Try to load the matched face image
                    if hasattr(match, 'face_image') and match.face_image is not None:
                        ax_match.imshow(cv2.cvtColor(match.face_image, cv2.COLOR_BGR2RGB))
                    else:
                        ax_match.text(0.5, 0.5, f'Match {i+1}\n{match.video_name}\nFrame: {match.frame_number}\nTime: {match.timestamp:.1f}s', 
                                     ha='center', va='center', fontsize=10)
                    ax_match.axis('off')
                
                # Details table
                ax_table = fig.add_subplot(gs[3, :])
                ax_table.axis('off')
                
                # Create table data
                table_data = [['#', 'Video File', 'Time (s)', 'Frame', 'Confidence']]
                for i, match in enumerate(self.matches_found[:5], 1):  # Show first 5 matches
                    video_name = match.video_name[:25] + '...' if len(match.video_name) > 25 else match.video_name
                    table_data.append([
                        str(i),
                        video_name,
                        f"{match.timestamp:.1f}",
                        str(match.frame_number),
                        f"{match.confidence*100:.1f}%"
                    ])
                
                # Create matplotlib table
                table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                                     cellLoc='center', loc='center',
                                     colWidths=[0.1, 0.4, 0.15, 0.15, 0.2])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                # Style the table
                for i in range(len(table_data[0])):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                if total_matches > 5:
                    ax_table.text(0.5, 0.1, f'Note: Showing first 5 matches. Total matches: {total_matches}', 
                                 ha='center', va='bottom', fontsize=10, style='italic')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            logging.info(f"PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"PDF generation error: {e}")
            # Try fallback text-based report
            try:
                return self.generate_text_report(output_path)
            except Exception as fallback_error:
                logging.error(f"Fallback text report also failed: {fallback_error}")
                return None
    
    def generate_text_report(self, output_path: str = None) -> str:
        """Generate a simple text-based report as fallback"""
        try:
            if not self.matches_found:
                logging.warning("No matches found to generate report")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"face_recognition_report_{timestamp}.txt"
            
            # Change extension to .txt for text report
            if output_path.endswith('.pdf'):
                output_path = output_path[:-4] + '.txt'
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("FACE RECOGNITION ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Summary statistics
                total_matches = len(self.matches_found)
                videos_processed = len(set(match.video_path for match in self.matches_found))
                avg_confidence = np.mean([match.confidence for match in self.matches_found]) * 100
                analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                f.write("ANALYSIS SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"• Total Face Matches Found: {total_matches}\n")
                f.write(f"• Videos Processed: {videos_processed}\n")
                f.write(f"• Average Confidence: {avg_confidence:.1f}%\n")
                f.write(f"• Analysis Date: {analysis_date}\n")
                f.write(f"• Model Used: {self.model}\n\n")
                
                f.write("DETAILED MATCH RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'#':<3} {'Video File':<35} {'Time (s)':<10} {'Frame':<8} {'Confidence':<12}\n")
                f.write("-" * 75 + "\n")
                
                for i, match in enumerate(self.matches_found[:20], 1):  # Show first 20 matches
                    video_name = match.video_name[:32] + '...' if len(match.video_name) > 32 else match.video_name
                    f.write(f"{i:<3} {video_name:<35} {match.timestamp:<10.1f} {match.frame_number:<8} {match.confidence*100:<12.1f}%\n")
                
                if total_matches > 20:
                    f.write(f"\n... and {total_matches - 20} more matches\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Analysis completed successfully\n")
                f.write("Note: This is a text-based fallback report.\n")
                f.write("For visual reports with images, ensure matplotlib is properly installed.\n")
            
            logging.info(f"Text report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Text report generation error: {e}")
            return None
    
    def get_match_summary(self) -> Dict[str, Any]:
        """Get summary of current matches"""
        if not self.matches_found:
            return {
                'total_matches': 0,
                'videos_processed': 0,
                'avg_confidence': 0,
                'popups_shown': self.popups_shown,
                'max_popups_reached': False
            }
        
        return {
            'total_matches': len(self.matches_found),
            'videos_processed': len(set(match.video_path for match in self.matches_found)),
            'avg_confidence': np.mean([match.confidence for match in self.matches_found]) * 100,
            'popups_shown': self.popups_shown,
            'max_popups_reached': self.popups_shown >= self.max_popups,
            'matches': [
                {
                    'video_name': match.video_name,
                    'timestamp': match.timestamp,
                    'confidence': match.confidence * 100,
                    'frame_number': match.frame_number
                }
                for match in self.matches_found
            ]
        }
    
    def reset(self):
        """Reset the engine state"""
        self.stop_recognition()
        self.matches_found.clear()
        self.popups_shown = 0
        self._close_all_popups()
        self.total_frames_processed = 0
        logging.info("Face recognition engine reset")
    
    def __del__(self):
        """Cleanup when engine is destroyed"""
        try:
            self.stop_recognition()
            self._close_all_popups()
            # Clean up temp directory
            import shutil
            if os.path.exists(self.pdf_temp_dir):
                shutil.rmtree(self.pdf_temp_dir, ignore_errors=True)
        except:
            pass


# Test functionality
if __name__ == "__main__":
    # Basic test
    engine = FaceRecognitionEngine()
    print("✅ Face Recognition Engine initialized successfully")
    
    # Test model setting
    print(f"Setting model to VGG-Face: {engine.set_model('VGG-Face')}")
    print(f"Setting model to invalid model: {engine.set_model('InvalidModel')}")
    
    print("🎯 Face Recognition Engine ready for integration")