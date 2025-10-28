import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os
import logging

# Try to import YOLO for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Using classification-only detection.")

class PersonDetector:
    """Person detection model for CCTV integration with bounding box tracking"""
    
    def __init__(self, model_path=None, use_yolo=True):
        self.model = None
        self.yolo_model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        
        # Initialize YOLO for bounding box detection
        if self.use_yolo:
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # Use custom trained model
                logging.info("Custom YOLO model loaded for person detection and tracking")
                self.model_loaded = True
                return
            except Exception as e:
                logging.warning(f"Failed to load YOLO model: {e}. Falling back to classification.")
                self.use_yolo = False
        
        # Fallback to classification model
        if model_path is None:
            possible_paths = [
                'obj/person_detector_for_cctv.pth',
                'obj/best_person_classifier.pth',
                'best_person_classifier.pth',
                'person_detector_for_cctv.pth'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                logging.warning("No person detection model found in expected locations")
                return
        
        # Try to load the classification model
        self.load_classification_model(model_path)
    
    def load_classification_model(self, model_path):
        """Load the trained person classification model"""
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                logging.warning(f"Person detection model not found at {model_path}")
                return False
            
            logging.info(f"Loading person detection model from: {model_path}")
            
            # Load model architecture
            self.model = models.resnet18()
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info("Loaded model from structured checkpoint")
                elif 'model_architecture' in checkpoint:
                    # Handle the export format from the notebook
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info("Loaded model from exported checkpoint")
                else:
                    # Assume it's a state dict directly
                    self.model.load_state_dict(checkpoint)
                    logging.info("Loaded model from state dict")
            else:
                # Assume it's a state dict directly
                self.model.load_state_dict(checkpoint)
                logging.info("Loaded model from direct state dict")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.model_loaded = True
            logging.info(f"Person detection model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load person detection model: {str(e)}")
            return False
    
    def detect_person(self, frame):
        """
        Detect persons in the frame with bounding boxes
        Args:
            frame: OpenCV frame (BGR format)
        Returns:
            tuple: (has_person: bool, confidence: float, processed_frame: np.array, detections: list)
        """
        if not self.model_loaded:
            return False, 0.0, frame, []
        
        try:
            # Use YOLO for bounding box detection if available
            if self.use_yolo and self.yolo_model:
                return self.detect_with_yolo(frame)
            else:
                # Fallback to classification-based detection
                return self.detect_with_classification(frame)
                
        except Exception as e:
            logging.error(f"Person detection error: {e}")
            return False, 0.0, frame, []
    
    def detect_with_yolo(self, frame):
        """Detect persons using YOLO with bounding boxes"""
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        has_person = False
        max_confidence = 0.0
        processed_frame = frame.copy()
        
        # Process detections
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Class 0 is 'person' in COCO dataset
                    if cls == 0 and conf > 0.3:  # Confidence threshold
                        has_person = True
                        max_confidence = max(max_confidence, conf)
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class': 'person'
                        })
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green for person
                        thickness = 2
                        
                        # Draw rectangle
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Add label
                        label = f"Person {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(processed_frame, 
                                    (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(processed_frame, label, 
                                  (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                  (0, 0, 0), 2)
        
        # Add detection status overlay
        processed_frame = self.add_detection_overlay(processed_frame, has_person, max_confidence, len(detections))
        
        return has_person, max_confidence, processed_frame, detections
    
    def detect_with_classification(self, frame):
        """Detect persons using classification model (fallback)"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        has_person = prediction.item() == 1
        confidence_score = confidence.item()
        
        # Add visual indicators to frame (no bounding boxes for classification)
        processed_frame = self.add_detection_overlay(frame, has_person, confidence_score, 0)
        
        return has_person, confidence_score, processed_frame, []
    
    def add_detection_overlay(self, frame, has_person, confidence, detection_count=0):
        """Add visual overlay to indicate person detection"""
        # Create a copy of the frame to avoid modifying the original
        overlay_frame = frame.copy()
        
        # Get frame dimensions
        height, width = overlay_frame.shape[:2]
        
        # Detection indicator
        if has_person:
            # Green indicator for person detected
            color = (0, 255, 0)  # BGR format
            if detection_count > 0:
                status_text = f"PERSONS DETECTED: {detection_count}"
            else:
                status_text = "PERSON DETECTED"
            confidence_text = f"Confidence: {confidence:.2f}"
        else:
            # Red indicator for no person
            color = (0, 0, 255)  # BGR format
            status_text = "NO PERSON"
            confidence_text = f"Confidence: {confidence:.2f}"
        
        # Add status rectangle at top-left
        rect_width = max(350, len(status_text) * 12)
        cv2.rectangle(overlay_frame, (10, 10), (rect_width, 80), color, -1)
        cv2.rectangle(overlay_frame, (10, 10), (rect_width, 80), (0, 0, 0), 2)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay_frame, status_text, (20, 35), font, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay_frame, confidence_text, (20, 60), font, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        time_text = f"Time: {timestamp:.1f}s"
        cv2.putText(overlay_frame, time_text, (width - 150, 30), font, 0.5, (255, 255, 255), 1)
        
        # Add detection mode indicator
        mode_text = "YOLO Detection" if self.use_yolo else "Classification"
        cv2.putText(overlay_frame, mode_text, (width - 150, height - 30), font, 0.4, (255, 255, 255), 1)
        
        # Add detection indicator in corner
        indicator_size = 20
        if has_person:
            cv2.circle(overlay_frame, (width - 30, height - 60), indicator_size, (0, 255, 0), -1)
            cv2.putText(overlay_frame, "P", (width - 38, height - 52), font, 0.7, (0, 0, 0), 2)
        
        return overlay_frame
    
    def is_available(self):
        """Check if the person detector is available and loaded"""
        return self.model_loaded
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model_loaded:
            return "Person detection model not loaded"
        
        if self.use_yolo:
            return f"YOLO Person Detection - Device: {self.device}, Status: Ready (Bounding Boxes)"
        else:
            return f"Classification Person Detection - Device: {self.device}, Status: Ready (No Bounding Boxes)"