import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from datetime import datetime
import sqlite3
import logging
import hashlib
import zlib
import numpy as np
import queue
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import ssl
import base64
from typing import Dict, List, Optional
import socket
import random
import math
import pyotp
from local_db import init_db, add_user, get_user, verify_user, get_user_role, add_video, get_user_videos, get_all_users, update_user_totp, delete_videos, get_video_id_by_filepath, verify_backup_code, get_user_backup_codes
from video_optimizer import VideoOptimizer,VideoInfo
from optimized_gui import OptimizedUploadWidget
import qrcode
import secrets
import psutil
from cert_manager import CertificateManager
from face_recognition_engine import FaceRecognitionEngine

# Try to import person detector (optional dependency)
try:
    from person_detector import PersonDetector
    PERSON_DETECTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Person detection not available: {e}")
    PERSON_DETECTION_AVAILABLE = False
    PersonDetector = None

# Configure logging
logging.basicConfig(
    filename='cctv_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_vpn_connected():
    """Check if 10.0.0.2 is present in any network interface (WireGuard VPN)."""
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == "10.0.0.2":
                return True
    return False
# Security enhancements
ENCRYPTION_SALT = b'secure_salt_value_'

def generate_encryption_key():
    """Generate and save a new encryption key"""
    key = Fernet.generate_key()
    with open('encryption.key', 'wb') as f:
        f.write(key)
    return key

def load_encryption_key():
    """Load encryption key or generate if not exists"""
    try:
        with open('encryption.key', 'rb') as f:
            return f.read()
    except FileNotFoundError:
        return generate_encryption_key()

def derive_encryption_key(password: str) -> bytes:
    """Derive encryption key from password using PBKDF2"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=ENCRYPTION_SALT,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_data(data: str, password: str) -> str:
    """Encrypt data using derived key"""
    key = derive_encryption_key(password)
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(token: str, password: str) -> str:
    """Decrypt data using derived key"""
    key = derive_encryption_key(password)
    f = Fernet(key)
    return f.decrypt(token.encode()).decode()

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with SHA-256 and salt using HMAC"""
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return h.hex(), salt

def generate_totp_secret():
    """Generate a new TOTP secret"""
    return pyotp.random_base32()

def get_totp_uri(username, secret):
    """Get provisioning URI for authenticator apps"""
    return pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name="CCTV System")

def verify_totp_token(token, secret):
    """Verify a TOTP token"""
    totp = pyotp.TOTP(secret)
    return totp.verify(token)

def generate_backup_codes(count=10):
    """Generate backup codes"""
    return [secrets.token_urlsafe(6) for _ in range(count)]

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def format_duration(seconds):
    """Format duration in HH:MM:SS format"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

class VideoPlayer(threading.Thread):
    """Threaded video player for smooth playback with person detection"""
    def __init__(self, video_path, frame_queue, enable_detection=True):
        super().__init__()
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.running = True
        self.paused = False
        self.lock = threading.Lock()
        self.daemon = True
        
        # Person detection setup
        self.enable_detection = enable_detection and PERSON_DETECTION_AVAILABLE
        self.person_detector = None
        
        if self.enable_detection:
            try:
                self.person_detector = PersonDetector()
                if not self.person_detector.is_available():
                    logging.warning("Person detector model not available, running without detection")
                    self.enable_detection = False
                else:
                    logging.info("Person detection enabled for video playback")
            except Exception as e:
                logging.error(f"Failed to initialize person detector: {e}")
                self.enable_detection = False
        

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {self.video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1 / fps if fps > 0 else 1/30
        
        # Detection statistics
        detection_stats = {
            'total_frames': 0,
            'person_detected': 0,
            'detection_events': []
        }
        
        while self.running and cap.isOpened():
            with self.lock:
                if self.paused:
                    time.sleep(0.01)
                    continue
                    
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            detection_stats['total_frames'] += 1
            processed_frame = frame
            
            # Apply person detection if enabled
            if self.enable_detection and self.person_detector:
                try:
                    has_person, confidence, processed_frame, detections = self.person_detector.detect_person(frame)
                    
                    if has_person:
                        detection_stats['person_detected'] += 1
                        # Log detection event
                        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        detection_stats['detection_events'].append({
                            'timestamp': current_time,
                            'confidence': confidence,
                            'num_persons': len(detections) if detections else 1
                        })
                        
                        # Log high-confidence detections
                        if confidence > 0.8:
                            if detections:
                                logging.info(f"High confidence person detection at {current_time:.2f}s: {len(detections)} person(s) (confidence: {confidence:.3f})")
                            else:
                                logging.info(f"High confidence person detection at {current_time:.2f}s (confidence: {confidence:.3f})")
                    
                except Exception as e:
                    logging.error(f"Person detection error: {e}")
                    # Continue with original frame if detection fails
                    processed_frame = frame
            
            # Compress frame to reduce queue size
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            compressed_frame = zlib.compress(buffer)
            
            # Put frame in queue (non-blocking)
            try:
                self.frame_queue.put(compressed_frame, block=False)
            except queue.Full:
                pass
                
            time.sleep(frame_delay)
        
        # Log detection summary when video ends
        if self.enable_detection and detection_stats['total_frames'] > 0:
            detection_rate = (detection_stats['person_detected'] / detection_stats['total_frames']) * 100
            logging.info(f"Video analysis complete: {detection_stats['person_detected']}/{detection_stats['total_frames']} frames with person detection ({detection_rate:.1f}%)")
            
        cap.release()

    def pause(self):
        with self.lock:
            self.paused = True

    def resume(self):
        with self.lock:
            self.paused = False

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
    
    def is_detection_enabled(self):
        """Check if person detection is enabled and available"""
        return self.enable_detection and self.person_detector and self.person_detector.is_available()
    
    def get_detection_info(self):
        """Get information about the detection system"""
        if not self.enable_detection:
            return "Person detection disabled"
        elif not self.person_detector:
            return "Person detector not initialized"
        else:
            return self.person_detector.get_model_info()

class CCTVApp:
    """Main CCTV Management System Application"""
    def __init__(self, root):
        self.root = root
        self.root.title("Nyetra - Secured CCTV System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set custom window icon
        self.set_window_icon()
        self.vpn_last_status = None  # Track last VPN status for logging
        self.vpn_warned = False      # Track if warning was already shown
        self.cert_manager = CertificateManager()
        
        # Initialize database
        init_db()
        
        # Professional Defense Theme Color Scheme
        self.colors = {
            'bg_primary': '#0d1117',      # Dark charcoal background
            'bg_secondary': '#161b22',    # Slightly lighter charcoal
            'bg_accent': '#010409',       # Deep black accent
            'bg': '#161b22',              # Alias for secondary background (for compatibility)
            'card_bg': '#21262d',         # Dark card background
            'text_primary': '#f0f6fc',    # Bright white text
            'text_secondary': '#8b949e',  # Muted gray text
            'accent': '#238636',          # Military green primary
            'accent_hover': '#2ea043',    # Brighter green on hover
            'accent_dark': '#1a7f37',     # Darker green
            'success': '#3fb950',         # Success green
            'warning': '#d29922',         # Amber warning
            'error': '#da3633',           # Critical red
            'danger': '#da3633',          # Danger red (alias for error)
            'info': '#1f6feb',            # Information blue
            'border': '#30363d',          # Subtle border
            'menu_hover': '#262c36',      # Menu hover state
            'shadow': '#010409',          # Shadow color
            'player_bg': '#0d1117',       # Video player background
            'active_border': '#f85149',   # Active player border (red)
            'inactive_border': '#30363d'  # Inactive player border
        }
        
        # Configure window
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Initialize variables
        self.current_user = None
        self.current_role = None
        self.video_files = []
        self.failed_login_attempts = {}
        self.backup_codes = []
        
        # Initialize face recognition engine
        self.face_engine = FaceRecognitionEngine()
        self.target_image_path = None
        self.scanning_active = False
        self.scan_line_position = 0
        self.scan_direction = 1
        self.animation_after_id = None
        
        # Create background
        self.create_background()
        
        # Create login screen
        self.create_login_screen()
    
    def set_window_icon(self):
        """Set custom window icon for the CCTV application"""
        try:
            # Option 1: Try to use an existing .ico file
            # Uncomment and modify the path below if you have an icon file:
            # self.root.iconbitmap('path/to/your/icon.ico')
            
            # Option 2: Create a simple programmatic icon using PIL
            from PIL import Image, ImageDraw, ImageTk
            
            # Create a 32x32 icon with defense theme
            size = 32
            icon_image = Image.new('RGBA', (size, size), (13, 17, 23, 255))  # Dark background
            draw = ImageDraw.Draw(icon_image)
            
            # Draw a shield-like icon
            # Shield outline
            shield_points = [
                (16, 4),   # Top center
                (26, 8),   # Top right
                (26, 20),  # Bottom right
                (16, 28),  # Bottom center
                (6, 20),   # Bottom left
                (6, 8),    # Top left
                (16, 4)    # Back to top
            ]
            draw.polygon(shield_points, fill=(35, 134, 54, 255), outline=(46, 160, 67, 255))  # Green shield
            
            # Inner elements
            # Lightning bolt or security symbol
            draw.polygon([(12, 10), (16, 14), (14, 14), (20, 22), (16, 18), (18, 18), (12, 10)], 
                        fill=(240, 246, 252, 255))  # White lightning
            
            # Convert to PhotoImage for Tkinter
            self.window_icon = ImageTk.PhotoImage(icon_image)
            self.root.iconphoto(True, self.window_icon)
            
        except ImportError:
            # If PIL is not available, try a simple text-based approach
            try:
                # Create a simple colored square as fallback
                # This requires tkinter's built-in capabilities
                pass  # Tkinter default icon will be used
            except:
                pass  # Use system default
        
    def create_background(self):
        """Create background with dots"""
        self.bg_canvas = tk.Canvas(self.root, bg=self.colors['bg_primary'], highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Create dot pattern (more subtle for dark theme)
        for _ in range(80):
            x = random.randint(0, self.root.winfo_width())
            y = random.randint(0, self.root.winfo_height())
            size = random.randint(1, 3)
            color = '#30363d' if random.random() > 0.8 else '#21262d'
            self.bg_canvas.create_oval(x, y, x+size, y+size, fill=color, outline="")
    
    def create_login_screen(self):
        """Create login screen with enhanced security"""
        self.login_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        self.login_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Login container
        login_box = tk.Frame(
            self.login_frame,
            width=400,
            height=500,
            bg=self.colors['card_bg'],
            highlightbackground=self.colors['border'],
            highlightthickness=1
        )
        login_box.pack(padx=40, pady=40)
        login_box.pack_propagate(False)
        
        # Title with defense-style branding
        title_label = tk.Label(
            login_box,
            text="⚡ Nyetra Access",
            font=('Consolas', 18, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        )
        title_label.pack(pady=(30, 15))
        
        # Subtitle with military styling
        subtitle_label = tk.Label(
            login_box,
            text="SECURE ACCESS TERMINAL",
            font=('Consolas', 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Username field
        username_frame = tk.Frame(login_box, bg=self.colors['card_bg'])
        username_frame.pack(fill='x', padx=40, pady=5)
        
        tk.Label(username_frame, text="USERNAME", bg=self.colors['card_bg'], 
                fg=self.colors['accent'], font=('Consolas', 9, 'bold')).pack(anchor='w')
        
        self.username_var = tk.StringVar(value="admin")
        self.username_var.trace_add('write', lambda *args: self.check_show_setup_qr())
        
        
        self.setup_qr_btn = tk.Button(
            login_box, 
            text="Setup Authenticator",
            font=('Segoe UI', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            command=self.show_qr_setup
        )
        self.setup_qr_btn.pack(pady=(0, 15))
        self.setup_qr_btn.pack_forget()  # Hide by default

        self.check_show_setup_qr()  # <-- Only call here, after setup_qr_btn is created
        username_entry = ttk.Entry(username_frame, textvariable=self.username_var, width=30, 
                                  font=('Segoe UI', 11))
        username_entry.pack(fill='x', pady=(5, 0))
        
        # Password field
        password_frame = tk.Frame(login_box, bg=self.colors['card_bg'])
        password_frame.pack(fill='x', padx=40, pady=(15, 5))
        
        tk.Label(password_frame, text="PASSWORD", bg=self.colors['card_bg'], 
                fg=self.colors['accent'], font=('Consolas', 9, 'bold')).pack(anchor='w')
        
        self.password_var = tk.StringVar(value="Admin@123")
        password_entry = ttk.Entry(password_frame, textvariable=self.password_var, show="•", width=30, 
                                  font=('Segoe UI', 11))
        password_entry.pack(fill='x', pady=(5, 0))
        
        # Security token
        self.token_frame = tk.Frame(login_box, bg=self.colors['card_bg'])
        self.token_frame.pack(fill='x', padx=40, pady=(10, 5))
        
        tk.Label(self.token_frame, text="SECURITY TOKEN (6-DIGIT)", bg=self.colors['card_bg'], 
                fg=self.colors['accent'], font=('Consolas', 9, 'bold')).pack(anchor='w')
        
        self.token_var = tk.StringVar()
        token_entry = ttk.Entry(self.token_frame, textvariable=self.token_var, width=30, 
                               font=('Segoe UI', 11))
        token_entry.pack(fill='x', pady=(5, 0))
        
        # Token help
        self.token_help = tk.Label(
            self.token_frame,
            text="Enter code from authenticator app",
            font=('Consolas', 8),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        self.token_help.pack(anchor='e')
        
        # Backup code option with tactical styling
        backup_frame = tk.Frame(self.token_frame, bg=self.colors['card_bg'])
        backup_frame.pack(fill='x', pady=(8, 0))
        
        self.use_backup_var = tk.BooleanVar(value=False)
        backup_check = tk.Checkbutton(
            backup_frame,
            text="🔑 USE BACKUP CODE",
            variable=self.use_backup_var,
            font=('Consolas', 9, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['warning'],
            selectcolor=self.colors['warning'],
            activebackground=self.colors['card_bg'],
            cursor="hand2",
            command=self.toggle_backup_mode
        )
        backup_check.pack(anchor='w')
        
        # Backup code instructions
        backup_help = tk.Label(
            backup_frame,
            text="Use when OTP unavailable • Single use only",
            font=('Consolas', 7),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        backup_help.pack(anchor='w', padx=(20, 0))
        
        # Login button with defense styling
        login_btn = tk.Button(
            login_box, 
            text="▶ AUTHORIZE ACCESS", 
            font=('Consolas', 11, 'bold'),
            bg=self.colors['accent'],
            fg='black',
            activebackground=self.colors['accent_hover'],
            activeforeground='black',
            relief="flat",
            borderwidth=2,
            highlightbackground=self.colors['accent_dark'],
            cursor="hand2",
            command=self.authenticate
        )
        login_btn.pack(fill='x', padx=40, pady=20, ipady=8)
        
        # Setup QR code button (for admin only)
        self.setup_qr_btn = tk.Button(
            login_box, 
            text="Setup Authenticator",
            font=('Segoe UI', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            command=self.show_qr_setup
        )
        self.setup_qr_btn.pack(pady=(0, 15))
        self.setup_qr_btn.pack_forget()  # Hide by default
        
        # link_auth_btn = tk.Button(
        #     login_box,
        #     text="Link/Relink Authenticator App",
        #     font=('Segoe UI', 10, 'underline'),
        #     bg=self.colors['card_bg'],
        #     fg=self.colors['accent'],
        #     activebackground=self.colors['bg_secondary'],
        #     activeforeground=self.colors['accent_hover'],
        #     relief="flat",
        #     borderwidth=0,
        #     cursor="hand2",
        #     command=self.show_qr_setup
        # )
        # link_auth_btn.pack(fill='x', padx=40, pady=(0, 10))
        # Version info
        login_box.pack(padx=40, pady=40)
        login_box.pack_propagate(False)

        version_label = tk.Label(
            self.login_frame,  # <-- place on the parent frame, not inside login_box
            text="Nyetra CCTV MANAGEMENT SYSTEM v3.0 • CLASSIFIED",
            font=('Consolas', 8),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_primary'],
            anchor='center'
        )
        version_label.pack(side='bottom', fill='x', pady=(0, 10))

        
        # Bind Enter key to login
        username_entry.bind('<Return>', lambda e: self.authenticate())
        password_entry.bind('<Return>', lambda e: self.authenticate())
        token_entry.bind('<Return>', lambda e: self.authenticate())
        
        # Focus username field
        username_entry.focus_set()
        
    def show_qr_setup(self):
        """Show QR code setup for authenticator app"""
        user = self.username_var.get()
        user_record = get_user(user)
        
        if not user_record:
            messagebox.showerror("Error", "User not found")
            return
            
        # Generate new secret if needed
        if not user_record.get('totp_secret'):
            secret = generate_totp_secret()
            update_user_totp(user, secret)
            user_record['totp_secret'] = secret
            
        # Generate QR code
        secret = user_record['totp_secret']
        uri = get_totp_uri(user, secret)
        
        # Create QR code window
        qr_window = tk.Toplevel(self.root)
        qr_window.title("Authenticator Setup")
        qr_window.geometry("400x500")
        qr_window.resizable(False, False)
        qr_window.grab_set()
        
        # Title
        tk.Label(
            qr_window,
            text="Setup Authenticator App",
            font=('Segoe UI', 14, 'bold'),
            pady=10
        ).pack()
        
        # Instructions
        instructions = tk.Label(
            qr_window,
            text="Scan this QR code with your authenticator app\n(Google Authenticator, Authy, Microsoft Authenticator, etc.)",
            font=('Segoe UI', 10),
            pady=10,
            wraplength=350
        )
        instructions.pack()
        
        # Generate QR code
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=8,
            border=2,
        )
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to PhotoImage
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(img)
        
        # Display QR code
        qr_label = tk.Label(qr_window, image=tk_img)
        qr_label.image = tk_img  # Keep reference
        qr_label.pack(pady=10)
        
        # Show secret key
        secret_frame = tk.Frame(qr_window)
        secret_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(secret_frame, text="Secret Key:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        secret_text = tk.Text(secret_frame, height=2, width=30, font=('Consolas', 10), bg='#f0f0f0')
        secret_text.insert(tk.END, secret)
        secret_text.config(state='disabled')
        secret_text.pack(fill='x', pady=5)
        
        # Backup codes
        backup_frame = tk.Frame(qr_window)
        backup_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(backup_frame, text="Backup Codes:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        backup_text = tk.Text(backup_frame, height=6, width=30, font=('Consolas', 9), bg='#f0f0f0')
        
        # Generate backup codes
        self.backup_codes = generate_backup_codes()
        for code in self.backup_codes:
            backup_text.insert(tk.END, f"• {code}\n")
            
        backup_text.config(state='disabled')
        backup_text.pack(fill='x', pady=5)
        
        # Save button
        save_btn = tk.Button(
            qr_window,
            text="Save Backup Codes",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg='white',
            command=lambda: self.save_backup_codes(self.backup_codes)
        )
        save_btn.pack(pady=10)
        
        # Close button
        close_btn = tk.Button(
            qr_window,
            text="Close",
            font=('Segoe UI', 10),
            command=qr_window.destroy
        )
        close_btn.pack(pady=5)
    def toggle_backup_mode(self):
        """Toggle between token and backup code mode"""
        if self.use_backup_var.get():
            self.token_var.set("")
            token_label = self.login_frame.nametowidget(self.token_frame.winfo_children()[0])
            token_label.config(text="BACKUP CODE")
            self.token_help.config(text="Enter one-time backup code")
        else:
            self.token_var.set("")
            token_label = self.login_frame.nametowidget(self.token_frame.winfo_children()[0])
            token_label.config(text="SECURITY TOKEN (6-DIGIT)")
            self.token_help.config(text="Enter code from authenticator app")
    def check_show_setup_qr(self, *args):
        """Hide setup button since we use first-time login flow"""
        # Always hide the setup button - we handle setup through first-time login flow
        self.setup_qr_btn.pack_forget()
    def show_qr_setup(self):
        """Show QR code setup for authenticator app"""
        user = self.username_var.get()
        user_record = get_user(user)
        
        if not user_record:
            messagebox.showerror("Error", "User not found")
            return
            
        # Generate new secret if needed
        if not user_record.get('totp_secret'):
            secret = generate_totp_secret()
            update_user_totp(user, secret)
            user_record['totp_secret'] = secret
            
        # Generate QR code
        secret = user_record['totp_secret']
        uri = get_totp_uri(user, secret)
        
        # Create QR code window
        qr_window = tk.Toplevel(self.root)
        qr_window.title("Authenticator Setup")
        qr_window.geometry("400x500")
        qr_window.resizable(False, False)
        qr_window.grab_set()
        
        # Title
        tk.Label(
            qr_window,
            text="Setup Authenticator App",
            font=('Segoe UI', 14, 'bold'),
            pady=10
        ).pack()
        
        # Instructions
        instructions = tk.Label(
            qr_window,
            text="Scan this QR code with your authenticator app\n(Google Authenticator, Authy, Microsoft Authenticator, etc.)",
            font=('Segoe UI', 10),
            pady=10,
            wraplength=350
        )
        instructions.pack()
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=8,
            border=2,
        )
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to PhotoImage
        tk_img = ImageTk.PhotoImage(img)
        
        # Display QR code
        qr_label = tk.Label(qr_window, image=tk_img)
        qr_label.image = tk_img  # Keep reference
        qr_label.pack(pady=10)
        
        # Show secret key
        secret_frame = tk.Frame(qr_window)
        secret_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(secret_frame, text="Secret Key:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        secret_text = tk.Text(secret_frame, height=2, width=30, font=('Consolas', 10), bg='#f0f0f0')
        secret_text.insert(tk.END, secret)
        secret_text.config(state='disabled')
        secret_text.pack(fill='x', pady=5)
        
        # Backup codes
        backup_frame = tk.Frame(qr_window)
        backup_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(backup_frame, text="Backup Codes:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        backup_text = tk.Text(backup_frame, height=6, width=30, font=('Consolas', 9), bg='#f0f0f0')
        
        # Generate backup codes
        self.backup_codes = generate_backup_codes()
        for code in self.backup_codes:
            backup_text.insert(tk.END, f"• {code}\n")
            
        backup_text.config(state='disabled')
        backup_text.pack(fill='x', pady=5)
        
        # Save button
        save_btn = tk.Button(
            qr_window,
            text="Save Backup Codes",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg='white',
            command=lambda: self.save_backup_codes(self.backup_codes)
        )
        save_btn.pack(pady=10)
        
        # Close button
        close_btn = tk.Button(
            qr_window,
            text="Close",
            font=('Segoe UI', 10),
            command=qr_window.destroy
        )
        close_btn.pack(pady=5)

    def show_mandatory_authenticator_setup(self):
        """Show mandatory authenticator setup for first-time login"""
        # Create a blocking setup window
        setup_window = tk.Toplevel(self.root)
        setup_window.title("🔐 Mandatory Authenticator Setup")
        setup_window.geometry("500x700")
        setup_window.resizable(False, False)
        setup_window.grab_set()
        setup_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        # Configure dark theme for setup window
        setup_window.configure(bg=self.colors['bg_primary'])
        
        # Header
        header_frame = tk.Frame(setup_window, bg=self.colors['accent'], height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="🛡️ MANDATORY SECURITY SETUP",
            font=('Consolas', 14, 'bold'),
            bg=self.colors['accent'],
            fg='black',
            pady=15
        ).pack()
        
        # Main content
        content_frame = tk.Frame(setup_window, bg=self.colors['bg_primary'])
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Welcome message
        welcome_label = tk.Label(
            content_frame,
            text=f"Welcome, {self.current_user.upper()}!\nRole: {self.current_role}",
            font=('Consolas', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            justify=tk.CENTER
        )
        welcome_label.pack(pady=(0, 20))
        
        # Instructions with better formatting
        instructions = tk.Label(
            content_frame,
            text="🔐 SECURITY PROTOCOL MANDATORY\n\nFor defense-grade security, you must set up\ntwo-factor authentication before proceeding.\n\n📱 STEP 1: Scan QR code with your authenticator app\n📝 STEP 2: Enter the 6-digit code to verify setup",
            font=('Consolas', 10),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER
        )
        instructions.pack(pady=(0, 20))
        
        # Step indicator
        self.step_indicator = tk.Label(
            content_frame,
            text="📱 STEP 1: SCAN QR CODE",
            font=('Consolas', 11, 'bold'),
            bg=self.colors['warning'],
            fg='black',
            padx=15,
            pady=5
        )
        self.step_indicator.pack(pady=(0, 15))
        
        # Generate new secret for user
        secret = generate_totp_secret()
        update_user_totp(self.current_user, secret)
        
        # Generate QR code
        uri = get_totp_uri(self.current_user, secret)
        
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=8,
            border=2,
        )
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to PhotoImage
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(img)
        
        # Display QR code
        qr_label = tk.Label(content_frame, image=tk_img, bg=self.colors['bg_primary'])
        qr_label.image = tk_img  # Keep reference
        qr_label.pack(pady=10)
        
        # Show secret key
        secret_frame = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='solid', bd=1)
        secret_frame.pack(fill='x', pady=10)
        
        tk.Label(secret_frame, text="🔑 SECRET KEY:", font=('Consolas', 9, 'bold'), 
                bg=self.colors['card_bg'], fg=self.colors['accent']).pack(anchor='w', padx=10, pady=(10, 5))
        
        secret_text = tk.Text(secret_frame, height=2, width=30, font=('Consolas', 9), 
                             bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                             relief='flat', bd=0)
        secret_text.insert(tk.END, secret)
        secret_text.config(state='disabled')
        secret_text.pack(fill='x', padx=10, pady=(0, 10))
        
        # Verification section with better UX
        verify_frame = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='solid', bd=1)
        verify_frame.pack(fill='x', pady=20)
        
        tk.Label(verify_frame, text="� STEP 2: VERIFY SETUP", font=('Consolas', 9, 'bold'),
                bg=self.colors['card_bg'], fg=self.colors['accent']).pack(anchor='w', padx=10, pady=(10, 5))
        
        tk.Label(verify_frame, text="After scanning, enter the 6-digit code from your authenticator app:",
                font=('Consolas', 8), bg=self.colors['card_bg'], fg=self.colors['text_secondary']).pack(anchor='w', padx=10)
        
        # Entry with better styling
        entry_frame = tk.Frame(verify_frame, bg=self.colors['card_bg'])
        entry_frame.pack(fill='x', padx=10, pady=10)
        
        verify_var = tk.StringVar()
        verify_entry = ttk.Entry(entry_frame, textvariable=verify_var, width=15, font=('Consolas', 14))
        verify_entry.pack(side='left')
        
        # Status indicator
        self.verify_status = tk.Label(
            entry_frame,
            text="⏳ Waiting for code...",
            font=('Consolas', 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        self.verify_status.pack(side='left', padx=(10, 0))
        
        # Real-time verification as user types
        def on_code_change(*args):
            code = verify_var.get().strip()
            if len(code) == 6 and code.isdigit():
                self.verify_status.config(text="🔍 Verifying...", fg=self.colors['warning'])
                # Delay verification slightly to show the checking state
                self.root.after(500, lambda: verify_code_realtime(code))
            elif len(code) > 0:
                self.verify_status.config(text="✍️ Keep typing...", fg=self.colors['text_secondary'])
            else:
                self.verify_status.config(text="⏳ Waiting for code...", fg=self.colors['text_secondary'])
        
        def verify_code_realtime(code):
            if verify_totp_token(code, secret):
                self.verify_status.config(text="✅ Valid! Click Complete", fg=self.colors['success'])
                verify_btn.config(state='normal', bg=self.colors['success'])
                self.step_indicator.config(text="✅ SETUP COMPLETE!", bg=self.colors['success'])
            else:
                self.verify_status.config(text="❌ Invalid code", fg=self.colors['error'])
                verify_btn.config(state='disabled', bg=self.colors['bg_secondary'])
        
        verify_var.trace('w', on_code_change)
        
        # Buttons with improved UX
        button_frame = tk.Frame(content_frame, bg=self.colors['bg_primary'])
        button_frame.pack(fill='x', pady=20)
        
        def verify_and_complete():
            code = verify_var.get().strip()
            if not code:
                messagebox.showerror("Verification Failed", "Please enter a verification code")
                return
                
            if verify_totp_token(code, secret):
                # Show success animation/feedback
                self.verify_status.config(text="✅ SUCCESS!", fg=self.colors['success'])
                self.step_indicator.config(text="🎉 AUTHENTICATION ACTIVE!", bg=self.colors['success'])
                
                # Brief delay to show success state
                self.root.after(1000, lambda: self.complete_setup(setup_window))
            else:
                self.verify_status.config(text="❌ Invalid code - try again", fg=self.colors['error'])
                verify_var.set("")
                verify_entry.focus_set()
        
        verify_btn = tk.Button(
            button_frame,
            text="✓ COMPLETE SETUP",
            font=('Consolas', 10, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            activebackground=self.colors['accent_hover'],
            activeforeground='black',
            relief="flat",
            borderwidth=2,
            padx=20,
            pady=8,
            cursor="hand2",
            state='disabled',  # Start disabled until valid code
            command=verify_and_complete
        )
        verify_btn.pack(side='right')
        
        # Help text
        help_text = tk.Label(
            content_frame,
            text="💡 TIP: Popular authenticator apps include Google Authenticator,\nMicrosoft Authenticator, Authy, and 1Password",
            font=('Consolas', 8),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER
        )
        help_text.pack(pady=(20, 0))
        
        # Bind Enter key to verify
        verify_entry.bind('<Return>', lambda e: verify_and_complete())
        verify_entry.focus_set()
        
        # Store references for later use
        setup_window.verify_btn = verify_btn
    
    def complete_setup(self, setup_window):
        """Complete the authenticator setup and proceed to main interface"""
        messagebox.showinfo("Setup Complete", 
                          f"🎉 Authenticator setup successful!\n\n✅ User: {self.current_user}\n✅ Role: {self.current_role}\n\n🛡️ Your account is now secured with two-factor authentication!")
        logging.info(f"Authenticator setup completed for {self.current_user}")
        setup_window.destroy()
        self.create_main_interface()

    def save_backup_codes(self, codes):
        """Save backup codes to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write("CCTV System Backup Codes\n\n")
                f.write("Use these codes if you lose access to your authenticator app:\n\n")
                for code in codes:
                    f.write(f"{code}\n")
                    
            messagebox.showinfo("Backup Codes Saved", "Backup codes saved successfully")
            logging.info(f"Backup codes saved for {self.username_var.get()}")

    def authenticate(self):
        """Authenticate user with enhanced security"""
        username = self.username_var.get()
        password = self.password_var.get()
        token = self.token_var.get()
        use_backup = self.use_backup_var.get()
        
        if not username or not password:
            messagebox.showerror("Authentication Failed", "Username and password are required")
            return
            
        user_record = verify_user(username, password)
        if user_record:
            # Check if TOTP is set up (First-time login flow)
            if not user_record.get('totp_secret'):
                # First-time login: Allow bypass and force setup
                result = messagebox.askyesno(
                    "First-Time Login", 
                    f"🔐 Welcome {username}!\n\nThis is your first login. You need to set up two-factor authentication for security.\n\n▶ Continue to setup authenticator now?",
                    icon='question'
                )
                
                if result:
                    # Proceed with temporary login and force setup
                    self.current_user = username
                    self.current_role = get_user_role(username)
                    logging.info(f"First-time login for {username} - proceeding to authenticator setup")
                    
                    # Clear login frame
                    self.login_frame.destroy()
                    
                    # Show mandatory authenticator setup
                    self.show_mandatory_authenticator_setup()
                    return
                else:
                    messagebox.showinfo("Setup Required", "Authenticator setup is mandatory for system access.")
                    return
                
            # Regular login flow: Verify token or backup code
            if use_backup:
                # Backup code verification
                if not token:
                    messagebox.showerror("Authentication Failed", "Backup code is required")
                    return
                
                # Verify the backup code
                if verify_backup_code(username, token):
                    # Show remaining backup codes
                    remaining_codes = get_user_backup_codes(username)
                    messagebox.showinfo("Backup Code Accepted", 
                                      f"✅ Backup code accepted!\n\nRemaining codes: {len(remaining_codes)}\n\n⚠️ Warning: Each backup code can only be used once.")
                else:
                    messagebox.showerror("Authentication Failed", "Invalid backup code")
                    return
            else:
                if not token:
                    messagebox.showerror("Authentication Failed", "Security token is required")
                    return
                    
                if not verify_totp_token(token, user_record['totp_secret']):
                    messagebox.showerror("Authentication Failed", "Invalid security token")
                    return
                    
            # Successful authentication
            self.current_user = username
            self.current_role = get_user_role(username)
            logging.info(f"User {username} logged in")
            
            # Reset failed login attempts
            self.failed_login_attempts[username] = 0
            
            # Clear login frame
            self.login_frame.destroy()
            
            # Create main interface
            self.create_main_interface()
        else:
            # Increment failed login attempts
            self.failed_login_attempts[username] = self.failed_login_attempts.get(username, 0) + 1
            ip_address = socket.gethostbyname(socket.gethostname())
            logging.warning(f"Failed login attempt for {username} from IP {ip_address}")
            
            # Show error message
            messagebox.showerror("Authentication Failed", "Invalid username or password")
            
            # Lock account after 3 attempts
            if self.failed_login_attempts[username] >= 3:
                logging.critical(f"Account locked for {username} after 3 failed attempts")
                messagebox.showwarning("Account Locked", "Too many failed attempts. Account temporarily locked.")
                self.failed_login_attempts[username] = 0
    
    def create_main_interface(self):
        """Create main interface after authentication"""
        # Create header
        self.create_header()
        
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        
        # Configure tab styling with larger SF font
        style = ttk.Style()
        style.configure('TNotebook.Tab', 
                       font=('Segoe UI', 12, 'bold'),
                       padding=[20, 10],  # [horizontal, vertical] padding
                       foreground='black')  # Default text color
        style.map('TNotebook.Tab',
                 background=[('selected', self.colors['accent']),
                           ('active', self.colors['card_bg']),
                           ('!selected', 'SystemButtonFace')],
                 foreground=[('selected', 'black'),
                           ('active', 'black'),
                           ('!selected', 'black')])
        
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create tabs
        self.create_video_tab()
        
        print(f"DEBUG: Creating tabs for user: {self.current_user}, role: {self.current_role}")
        
        if self.current_role in ("Admin", "Operator"):
            print(f"DEBUG: Creating upload and face recognition tabs for {self.current_role}")
            self.create_upload_tab()
            self.create_face_recognition_tab()
            print(f"DEBUG: Face recognition tab created. results_text exists: {hasattr(self, 'results_text')}")
        else:
            print(f"DEBUG: Skipping upload/face recognition tabs for role: {self.current_role}")
            
        if self.current_role == "Admin":
            print(f"DEBUG: Creating admin-specific tabs")
            self.create_security_tab()
            self.create_threat_tab()
        self.create_logs_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value=f"Welcome, {self.current_user}! Role: {self.current_role}")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            font=("Segoe UI", 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            padx=10
        )
        status_bar.pack(side="bottom", fill="x")
        
        # Start monitoring thread
        self.start_monitoring()
        
        # Bind window resize event for responsive design
        self.root.bind('<Configure>', self.on_window_resize)
    
    def create_header(self):
        """Create professional defense-style app header"""
        header_frame = tk.Frame(self.root, bg=self.colors['bg_accent'])
        header_frame.pack(fill='x', pady=(0, 5))
        
        # Logo and title with defense styling
        logo_frame = tk.Frame(header_frame, bg=self.colors['bg_accent'])
        logo_frame.pack(side='left', padx=20)
        
        logo_label = tk.Label(
            logo_frame, 
            text="🛡️", 
            font=('Segoe UI', 28, 'bold'),
            bg=self.colors['bg_accent'],
            fg=self.colors['accent']
        )
        logo_label.pack(side='left', padx=(0, 15))
        
        title_label = tk.Label(
            logo_frame,
            text="Nyetra - Secured CCTV SYSTEM",
            font=('Segoe UI', 20, 'bold'),
            bg=self.colors['bg_accent'],
            fg=self.colors['text_primary']
        )
        title_label.pack(side='left')
        
        # Classification label
        class_label = tk.Label(
            logo_frame,
            text="CLASSIFIED",
            font=('Consolas', 8, 'bold'),
            bg=self.colors['error'],
            fg='white',
            padx=6,
            pady=2
        )
        class_label.pack(side='left', padx=(15, 0))
        
        # User info with professional styling
        user_frame = tk.Frame(header_frame, bg=self.colors['bg_accent'])
        user_frame.pack(side='right', padx=20)
        
        # User role badge with professional colors
        role_colors = {
            "Admin": "#da3633",      # Critical red for admin
            "Operator": "#d29922",   # Amber for operator  
            "Viewer": "#1f6feb"      # Blue for viewer
        }
        role_color = role_colors.get(self.current_role, "#8b949e")
        
        role_badge = tk.Label(
            user_frame,
            text=f"◈ {self.current_role.upper()}",
            font=('Consolas', 9, 'bold'),
            bg=role_color,
            fg='black' if self.current_role == 'Operator' else 'white',
            padx=10,
            pady=4
        )
        role_badge.pack(side='right', padx=(15, 0))
        
        # User info with professional font
        user_info = tk.Label(
            user_frame,
            text=f"USER: {self.current_user.upper()}",
            font=('Consolas', 11, 'bold'),
            bg=self.colors['bg_accent'],
            fg=self.colors['text_primary']
        )
        user_info.pack(side='right')
        
        # Logout button with defense styling
        logout_btn = tk.Button(
            user_frame,
            text="⏻ LOGOUT",
            font=('Consolas', 9, 'bold'),
            bg=self.colors['error'],
            fg='white',
            activebackground='#a40e26',
            activeforeground='white',
            borderwidth=0,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.logout
        )
        logout_btn.pack(side='right', padx=25)
    def create_video_tab(self):
        """Create video playback tab with 4 simultaneous video players"""
        video_tab = ttk.Frame(self.notebook)
        self.notebook.add(video_tab, text="📹 Video Playback")
        
        # Main container - split into video area and control panel
        main_container = ttk.Frame(video_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video players (2x2 grid)
        video_container = tk.Frame(main_container, bg=self.colors['bg_primary'])
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create 2x2 grid for video players
        self.video_players = []
        self.video_labels = []
        self.video_frames = []
        self.active_player_index = 0  # Track which player is currently active
        
        # Initialize target video dimensions (will be updated dynamically)
        self.target_video_width = 380
        self.target_video_height = 280
        
        # Title for multi-view
        title_frame = tk.Frame(video_container, bg=self.colors['card_bg'],
                              highlightthickness=1, highlightbackground=self.colors['border'])
        title_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            title_frame,
            text="⬢ TACTICAL SURVEILLANCE MATRIX",
            font=('Consolas', 13, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['accent'],
            pady=12
        ).pack()
        
        # Video grid container
        grid_container = tk.Frame(video_container, bg=self.colors['bg_primary'])
        grid_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid weights for responsive resizing
        grid_container.grid_rowconfigure(0, weight=1)
        grid_container.grid_rowconfigure(1, weight=1)
        grid_container.grid_columnconfigure(0, weight=1)
        grid_container.grid_columnconfigure(1, weight=1)
        
        # Store grid container reference for resizing
        self.grid_container = grid_container
        
        # Create 4 video player frames in 2x2 grid with responsive dimensions
        for i in range(4):
            row = i // 2
            col = i % 2
            
            # Video frame with professional defense styling
            video_frame = tk.Frame(
                grid_container,
                bg=self.colors['player_bg'],
                highlightthickness=3,
                highlightbackground=self.colors['inactive_border'],
                cursor="hand2"
            )
            video_frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            
            # Video display label with tactical styling
            video_label = tk.Label(
                video_frame,
                text=f"FEED {i+1:02d}\n◦ STANDBY ◦",
                font=('Consolas', 11, 'bold'),
                bg=self.colors['player_bg'],
                fg=self.colors['text_secondary'],
                justify=tk.CENTER
            )
            video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            # Player info label with tactical design
            info_label = tk.Label(
                video_frame,
                text=f"◈ CHANNEL {i+1:02d}",
                font=('Consolas', 8, 'bold'),
                bg=self.colors['accent'],
                fg='black',
                pady=4
            )
            info_label.pack(fill='x', side='bottom')
            
            # Store references
            self.video_frames.append(video_frame)
            self.video_labels.append(video_label)
            
            # Bind click events
            video_frame.bind("<Button-1>", lambda e, idx=i: self.select_active_player(idx))
            video_label.bind("<Button-1>", lambda e, idx=i: self.select_active_player(idx))
            info_label.bind("<Button-1>", lambda e, idx=i: self.select_active_player(idx))
        
        # Highlight the first player as active
        self.update_active_player_highlight()
        
        # Right side - Control panel and video library
        control_panel = ttk.Frame(main_container)
        control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Active player info
        active_info_frame = tk.Frame(control_panel, bg=self.colors['card_bg'],
                                   highlightthickness=1, highlightbackground=self.colors['border'])
        active_info_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            active_info_frame,
            text="◢ ACTIVE CHANNEL",
            font=('Consolas', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(pady=6)
        
        self.active_player_info = tk.Label(
            active_info_frame,
            text="FEED 01 • STANDBY",
            font=('Consolas', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        )
        self.active_player_info.pack()
        
        # Video controls
        control_frame = tk.Frame(control_panel, bg=self.colors['card_bg'],
                               highlightthickness=1, highlightbackground=self.colors['border'])
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Control buttons
        btn_container = tk.Frame(control_frame, bg=self.colors['card_bg'])
        btn_container.pack(expand=True, fill='both', padx=15, pady=15)
        
        # Row 1 - Transport controls
        transport_frame = tk.Frame(btn_container, bg=self.colors['card_bg'])
        transport_frame.pack(fill='x', pady=(0, 10))
        
        # Play button with tactical styling
        self.play_btn = tk.Button(
            transport_frame,
            text="▶ ENGAGE",
            font=('Consolas', 9, 'bold'),
            bg=self.colors['accent'],
            fg='black',
            activebackground=self.colors['accent_hover'],
            activeforeground='black',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.play_active_video
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        # Pause button
        self.pause_btn = tk.Button(
            transport_frame,
            text="⏸ HOLD",
            font=('Consolas', 9, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.pause_active_video
        )
        self.pause_btn.pack(side=tk.LEFT, padx=4)
        
        # Stop button
        self.stop_btn = tk.Button(
            transport_frame,
            text="⏹ CEASE",
            font=('Consolas', 9, 'bold'),
            bg=self.colors['error'],
            fg='white',
            activebackground='#a40e26',
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.stop_active_video
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Row 2 - Player controls
        player_frame = tk.Frame(btn_container, bg=self.colors['card_bg'])
        player_frame.pack(fill='x')
        
        # Previous video
        prev_btn = tk.Button(
            player_frame,
            text="◀ PREV",
            font=('Consolas', 9, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.previous_video
        )
        prev_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        # Next video
        next_btn = tk.Button(
            player_frame,
            text="NEXT ▶",
            font=('Consolas', 9, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.next_video
        )
        next_btn.pack(side=tk.LEFT, padx=5)
        
        # Person Detection Controls
        detection_frame = tk.Frame(control_panel, bg=self.colors['card_bg'],
                                 highlightthickness=1, highlightbackground=self.colors['border'])
        detection_frame.pack(fill='x', pady=(0, 10))
        
        detection_container = tk.Frame(detection_frame, bg=self.colors['card_bg'])
        detection_container.pack(expand=True, fill='both', padx=15, pady=15)
        
        # Detection toggle with tactical styling
        self.detection_enabled = tk.BooleanVar(value=True)
        detection_check = tk.Checkbutton(
            detection_container,
            text="◉ THREAT DETECTION",
            variable=self.detection_enabled,
            font=('Consolas', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['accent'],
            selectcolor=self.colors['accent'],
            activebackground=self.colors['card_bg'],
            cursor="hand2",
            command=self.toggle_person_detection
        )
        detection_check.pack(anchor='w')
        
        # Detection status indicator
        self.detection_status_var = tk.StringVar()
        self.detection_status_label = tk.Label(
            detection_container,
            textvariable=self.detection_status_var,
            font=('Segoe UI', 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        self.detection_status_label.pack(anchor='w', pady=(5, 0))
        
        # Update detection status
        self.update_detection_status()
        
        # Video library
        library_frame = tk.Frame(control_panel, bg=self.colors['card_bg'],
                                highlightthickness=1, highlightbackground=self.colors['border'])
        library_frame.pack(fill=tk.BOTH, expand=True)
        
        # Library title
        library_title = tk.Frame(library_frame, bg=self.colors['accent'])
        library_title.pack(fill='x')
        
        tk.Label(
            library_title,
            text="◢ TACTICAL ARCHIVE",
            font=('Consolas', 11, 'bold'),
            bg=self.colors['accent'],
            fg='black',
            padx=20,
            pady=10
        ).pack(side='left')
        
        # Search bar
        search_frame = tk.Frame(library_frame, bg=self.colors['card_bg'], padx=10, pady=10)
        search_frame.pack(fill='x')
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=('Segoe UI', 10))
        search_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        search_btn = tk.Button(
            search_frame,
            text="🔍",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=lambda: self.filter_videos(self.search_var.get())
        )
        search_btn.pack(side='right')
        
        # Selection and Delete Controls (only for admins and operators)
        if self.current_role in ("Admin", "Operator"):
            controls_frame = tk.Frame(library_frame, bg=self.colors['card_bg'], padx=10, pady=5)
            controls_frame.pack(fill='x')
            
            # Selection controls
            select_frame = tk.Frame(controls_frame, bg=self.colors['card_bg'])
            select_frame.pack(side='left')
            
            select_all_btn = tk.Button(
                select_frame,
                text="Select All",
                font=('Segoe UI', 9),
                bg=self.colors['accent'],
                fg='white',
                activebackground=self.colors['accent_hover'],
                activeforeground='white',
                bd=0,
                padx=8,
                pady=2,
                cursor="hand2",
                command=self.select_all_videos
            )
            select_all_btn.pack(side='left', padx=(0, 5))
            
            select_none_btn = tk.Button(
                select_frame,
                text="Select None",
                font=('Segoe UI', 9),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                activebackground=self.colors['menu_hover'],
                activeforeground=self.colors['text_primary'],
                bd=0,
                padx=8,
                pady=2,
                cursor="hand2",
                command=self.select_none_videos
            )
            select_none_btn.pack(side='left', padx=(0, 10))
            
            # Delete controls
            delete_frame = tk.Frame(controls_frame, bg=self.colors['card_bg'])
            delete_frame.pack(side='right')
            
            # Selected count label
            self.selected_count_label = tk.Label(
                delete_frame,
                text="0 selected",
                font=('Segoe UI', 9),
                bg=self.colors['card_bg'],
                fg=self.colors['text_primary']
            )
            self.selected_count_label.pack(side='left', padx=(0, 10))
            
            delete_selected_btn = tk.Button(
                delete_frame,
                text="🗑️ Delete Selected",
                font=('Segoe UI', 9),
                bg='#dc3545',  # Red color for delete
                fg='white',
                activebackground='#c82333',
                activeforeground='white',
                bd=0,
                padx=8,
                pady=2,
                cursor="hand2",
                command=self.delete_selected_videos
            )
            delete_selected_btn.pack(side='right')
        
        # Video list - configure columns based on user role
        if self.current_role in ("Admin", "Operator"):
            columns = ('Select', 'Name', 'Duration', 'Size', 'Resolution')
        else:
            columns = ('Name', 'Duration', 'Size', 'Resolution')
            
        self.video_tree = ttk.Treeview(library_frame, columns=columns, show='headings', height=12)
        
        # Configure columns
        if self.current_role in ("Admin", "Operator"):
            self.video_tree.heading('Select', text='☐')
            self.video_tree.column('Select', width=40, anchor='center')
            
        self.video_tree.heading('Name', text='Name')
        self.video_tree.column('Name', width=100)
        self.video_tree.heading('Duration', text='Duration')
        self.video_tree.column('Duration', width=60)
        self.video_tree.heading('Size', text='Size')
        self.video_tree.column('Size', width=60)
        self.video_tree.heading('Resolution', text='Resolution')
        self.video_tree.column('Resolution', width=80)
        
        video_scrollbar = ttk.Scrollbar(library_frame, orient=tk.VERTICAL, command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=video_scrollbar.set)
        
        self.video_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10, padx=(0, 10))
        
        # Initialize video system
        self.video_players = [None] * 4  # Store video player threads for each slot
        self.video_paths = [None] * 4    # Store video paths for each slot
        self.frame_queues = [queue.Queue(maxsize=30) for _ in range(4)]  # Frame queues for each player
        self.selected_videos = set()
        self.video_data = []  # Store video data with IDs
        
        # Load videos
        self.load_videos_for_playback()
        self.video_tree.bind('<<TreeviewSelect>>', self.on_video_select)
        self.video_tree.bind('<Button-1>', self.on_video_click)
        self.video_tree.bind('<Double-1>', self.on_video_double_click)  # Double-click to load video
        
        # Add keyboard shortcuts for video deletion (Admin/Operator only)
        if self.current_role in ['Admin', 'Operator']:
            self.video_tree.bind('<Delete>', self.delete_selected_videos_keyboard)
            self.video_tree.bind('<Button-3>', self.show_context_menu)  # Right-click
            
            # Create context menu
            self.context_menu = tk.Menu(self.video_tree, tearoff=0)
            self.context_menu.add_command(label="Load to Active Player", command=self.load_selected_to_active_player)
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Delete Selected", command=self.delete_selected_videos)
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Select All", command=self.select_all_videos)
            self.context_menu.add_command(label="Select None", command=self.select_none_videos)
        else:
            # For viewers, add context menu without delete options
            self.video_tree.bind('<Button-3>', self.show_context_menu_viewer)  # Right-click
            
            # Create viewer context menu
            self.context_menu = tk.Menu(self.video_tree, tearoff=0)
            self.context_menu.add_command(label="Load to Active Player", command=self.load_selected_to_active_player)
        
        # Start video frame update loop
        self.start_video_frame_updates()
        
        # Bind window resize event for responsive sizing
        self.root.bind('<Configure>', self.on_window_configure)
        
        # Initial size update after a short delay to ensure everything is rendered
        self.root.after(500, self.update_player_sizes)
    
    def load_videos_for_playback(self):
        """Load videos for playback"""
        self.video_files = []
        self.video_data = []
        self.selected_videos = set()
        self.video_tree.delete(*self.video_tree.get_children())
        
        # For viewers, show videos uploaded by operator or admin
        if self.current_role == "Viewer":
            videos = []
            for user in get_all_users():
                role = get_user_role(user)
                if role in ("Admin", "Operator"):
                    videos.extend(get_user_videos(user, self.current_user))
        else:
            videos = get_user_videos(self.current_user, self.current_user)
        
        index = 1
        for v in videos:
            self.video_files.append(v['filepath'])
            self.video_data.append(v)  # Store complete video data including ID
            duration_str = format_duration(v['duration'])
            size_str = format_file_size(v['size'])
            resolution_str = f"{v['width']}x{v['height']}"
            
            # Show checkbox only for admins and operators
            if self.current_role in ("Admin", "Operator"):
                checkbox = "☐"
                self.video_tree.insert('', 'end', text=str(index),
                                   values=(checkbox, v['filename'], duration_str, size_str, resolution_str))
            else:
                # For viewers, don't show the Select column
                self.video_tree.insert('', 'end', text=str(index),
                                   values=(v['filename'], duration_str, size_str, resolution_str))
            index += 1
        
        # Update selection count
        if hasattr(self, 'selected_count_label'):
            self.update_selection_count()
        
        # Select first video if available
        if self.video_files:
            items = self.video_tree.get_children()
            if items:
                self.video_tree.selection_set(items[0])
                self.video_tree.see(items[0])
    
    def filter_videos(self, search_term):
        """Filter videos based on search term"""
        self.video_tree.delete(*self.video_tree.get_children())
        self.selected_videos = set()
        
        if not search_term:
            self.load_videos_for_playback()
            return
        
        search_term = search_term.lower()
        filtered_videos = []
        filtered_paths = []
        
        if self.current_role == "Viewer":
            videos = []
            for user in get_all_users():
                role = get_user_role(user)
                if role in ("Admin", "Operator"):
                    videos.extend(get_user_videos(user, self.current_user))
        else:
            videos = get_user_videos(self.current_user, self.current_user)
            
        index = 1
        for v in videos:
            if (search_term in v['filename'].lower() or
                search_term in str(v['width']) or
                search_term in str(v['height'])):
                
                filtered_videos.append(v)
                filtered_paths.append(v['filepath'])
                
                duration_str = format_duration(v['duration'])
                size_str = format_file_size(v['size'])
                resolution_str = f"{v['width']}x{v['height']}"
                
                # Show checkbox only for admins and operators
                if self.current_role in ("Admin", "Operator"):
                    checkbox = "☐"
                    self.video_tree.insert('', 'end', text=str(index),
                                       values=(checkbox, v['filename'], duration_str, size_str, resolution_str))
                else:
                    self.video_tree.insert('', 'end', text=str(index),
                                       values=(v['filename'], duration_str, size_str, resolution_str))
                index += 1
                
        self.video_files = filtered_paths
        self.video_data = filtered_videos
        
        # Update selection count
        if hasattr(self, 'selected_count_label'):
            self.update_selection_count()
            
        if filtered_videos:
            items = self.video_tree.get_children()
            if items:
                self.video_tree.selection_set(items[0])
                self.video_tree.see(items[0])
        else:
            self.video_info_var.set("No matching videos")
    
    def on_video_select(self, event):
        """Handle video selection in the tree view"""
        # This method can be used to update UI when a video is selected
        # For now, we'll keep it simple since the main functionality
        # is handled by double-click and context menu
        pass
    
    def on_video_click(self, event):
        """Handle clicks on video list items for selection"""
        if self.current_role not in ("Admin", "Operator"):
            return
            
        # Get the clicked item and region
        region = self.video_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
            
        # Get the clicked item
        item = self.video_tree.identify_row(event.y)
        if not item:
            return
            
        # Get the column
        column = self.video_tree.identify_column(event.x)
        if column == "#1":  # Select column
            self.toggle_video_selection(item)
    
    def toggle_video_selection(self, item):
        """Toggle selection state of a video item"""
        item_index = self.video_tree.index(item)
        if item_index >= len(self.video_data):
            return
            
        video_id = self.video_data[item_index]['id']
        
        if video_id in self.selected_videos:
            self.selected_videos.remove(video_id)
            checkbox = "☐"
        else:
            self.selected_videos.add(video_id)
            checkbox = "☑"
        
        # Update the checkbox display
        values = list(self.video_tree.item(item)['values'])
        values[0] = checkbox
        self.video_tree.item(item, values=values)
        
        self.update_selection_count()
    
    def select_all_videos(self):
        """Select all videos"""
        self.selected_videos.clear()
        
        for i, video in enumerate(self.video_data):
            self.selected_videos.add(video['id'])
            
            # Update checkbox display
            item = self.video_tree.get_children()[i]
            values = list(self.video_tree.item(item)['values'])
            values[0] = "☑"
            self.video_tree.item(item, values=values)
        
        self.update_selection_count()
    
    def select_none_videos(self):
        """Deselect all videos"""
        self.selected_videos.clear()
        
        for i, video in enumerate(self.video_data):
            item = self.video_tree.get_children()[i]
            values = list(self.video_tree.item(item)['values'])
            values[0] = "☐"
            self.video_tree.item(item, values=values)
        
        self.update_selection_count()
    
    def update_selection_count(self):
        """Update the selection count label"""
        if hasattr(self, 'selected_count_label'):
            count = len(self.selected_videos)
            text = f"{count} selected" if count != 1 else "1 selected"
            self.selected_count_label.config(text=text)
    
    def delete_selected_videos(self):
        """Delete selected videos after confirmation"""
        if not self.selected_videos:
            messagebox.showwarning("Warning", "No videos selected for deletion.")
            return
        
        # Confirmation dialog
        count = len(self.selected_videos)
        video_names = []
        for video_id in self.selected_videos:
            for video in self.video_data:
                if video['id'] == video_id:
                    video_names.append(video['filename'])
                    break
        
        if count == 1:
            message = f"Are you sure you want to delete the video:\n\n{video_names[0]}\n\nThis action cannot be undone."
        else:
            message = f"Are you sure you want to delete {count} videos?\n\nThis action cannot be undone.\n\nVideos to be deleted:\n"
            message += "\n".join(f"• {name}" for name in video_names[:10])
            if count > 10:
                message += f"\n... and {count - 10} more"
        
        response = messagebox.askyesno(
            "Confirm Deletion",
            message,
            icon='warning'
        )
        
        if not response:
            return
        
        # Perform deletion
        try:
            result = delete_videos(list(self.selected_videos), self.current_user)
            
            if result['success_count'] > 0:
                success_msg = f"Successfully deleted {result['success_count']} video(s)."
                if result['error_count'] > 0:
                    success_msg += f"\n{result['error_count']} deletion(s) failed."
                    
                messagebox.showinfo("Deletion Complete", success_msg)
                
                # Reload the video list
                self.load_videos_for_playback()
                
            elif result['error_count'] > 0:
                error_msg = "Failed to delete videos:\n\n"
                error_msg += "\n".join(f"• {error}" for error in result['errors'][:5])
                if len(result['errors']) > 5:
                    error_msg += f"\n... and {len(result['errors']) - 5} more errors"
                    
                messagebox.showerror("Deletion Failed", error_msg)
            
        except PermissionError as e:
            messagebox.showerror("Permission Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during deletion:\n{str(e)}")
            logging.error(f"Error during video deletion: {str(e)}")

    def delete_selected_videos_keyboard(self, event):
        """Handle Delete key press to delete selected videos"""
        if self.selected_videos:
            self.delete_selected_videos()

    def show_context_menu(self, event):
        """Show context menu on right-click"""
        try:
            # Select the item under cursor
            item = self.video_tree.identify_row(event.y)
            if item:
                self.video_tree.selection_set(item)
                self.context_menu.post(event.x_root, event.y_root)
        except Exception as e:
            logging.error(f"Error showing context menu: {str(e)}")

    def select_active_player(self, player_index):
        """Select which player is currently active"""
        if 0 <= player_index < 4:
            self.active_player_index = player_index
            self.update_active_player_highlight()
            self.update_active_player_info()
    
    def update_active_player_highlight(self):
        """Update visual highlighting of the active player with tactical styling"""
        for i, frame in enumerate(self.video_frames):
            if i == self.active_player_index:
                # Highlight active player with tactical red border
                frame.configure(highlightbackground=self.colors['active_border'], highlightthickness=4)
            else:
                # Normal border for inactive players
                frame.configure(highlightbackground=self.colors['inactive_border'], highlightthickness=3)
    
    def on_window_resize(self, event):
        """Handle window resize events for responsive design"""
        # Only respond to window resize events (not child widget resizes)
        if event.widget == self.root:
            # Update video player dimensions if grid container exists
            if hasattr(self, 'grid_container'):
                self.root.after_idle(self.update_video_dimensions)
    
    def update_video_dimensions(self):
        """Update video player dimensions based on current window size"""
        try:
            if hasattr(self, 'grid_container'):
                # Get current grid container size
                self.grid_container.update_idletasks()
                container_width = self.grid_container.winfo_width()
                container_height = self.grid_container.winfo_height()
                
                # Calculate optimal video dimensions (2x2 grid with padding)
                if container_width > 50 and container_height > 50:  # Ensure valid dimensions
                    video_width = max(200, (container_width - 40) // 2)  # Minimum 200px width
                    video_height = max(150, (container_height - 40) // 2)  # Minimum 150px height
                    
                    # Update target dimensions
                    self.target_video_width = video_width
                    self.target_video_height = video_height
        except:
            pass  # Ignore any errors during resize
    
    def update_active_player_info(self):
        """Update the active player information display with tactical terminology"""
        feed_num = self.active_player_index + 1
        if self.video_paths[self.active_player_index]:
            video_name = os.path.basename(self.video_paths[self.active_player_index])
            self.active_player_info.config(text=f"FEED {feed_num:02d} • {video_name.upper()}")
        else:
            self.active_player_info.config(text=f"FEED {feed_num:02d} • STANDBY")
    
    def on_video_double_click(self, event):
        """Handle double-click on video list to load to active player"""
        self.load_selected_to_active_player()
    
    def load_selected_to_active_player(self):
        """Load selected video to the currently active player"""
        selection = self.video_tree.selection()
        if not selection:
            return
            
        items = self.video_tree.get_children()
        if selection and items:
            idx = items.index(selection[0])
            if 0 <= idx < len(self.video_files):
                video_path = self.video_files[idx]
                self.load_video_to_player(video_path, self.active_player_index)
    
    def load_video_to_player(self, video_path, player_index):
        """Load a video to a specific player"""
        if not (0 <= player_index < 4):
            return
        
        # Stop existing player if any
        if self.video_players[player_index]:
            self.video_players[player_index].stop()
            self.video_players[player_index] = None
        
        # Clear the frame queue
        while not self.frame_queues[player_index].empty():
            try:
                self.frame_queues[player_index].get_nowait()
            except queue.Empty:
                break
        
        try:
            # Create new video player
            detection_enabled = getattr(self, 'detection_enabled', None)
            enable_detection = detection_enabled.get() if detection_enabled else True
            
            self.video_players[player_index] = VideoPlayer(video_path, self.frame_queues[player_index], enable_detection)
            self.video_paths[player_index] = video_path
            self.video_players[player_index].start()
            
            # Update the label with tactical styling
            video_name = os.path.basename(video_path)
            self.video_labels[player_index].config(text=f"FEED {player_index + 1:02d}\n◦ {video_name.upper()} ◦")
            
            # Update active player info if this is the active player
            if player_index == self.active_player_index:
                self.update_active_player_info()
            
            logging.info(f"Loaded video {video_name} to Feed {player_index + 1:02d}")
            
        except Exception as e:
            logging.error(f"Error loading video to player {player_index + 1}: {str(e)}")
            messagebox.showerror("Error", f"Failed to load video to Player {player_index + 1}")
    
    def play_active_video(self):
        """Play video in the active player"""
        if self.video_players[self.active_player_index]:
            self.video_players[self.active_player_index].resume()
    
    def pause_active_video(self):
        """Pause video in the active player"""
        if self.video_players[self.active_player_index]:
            self.video_players[self.active_player_index].pause()
    
    def stop_active_video(self):
        """Stop video in the active player"""
        if self.video_players[self.active_player_index]:
            self.video_players[self.active_player_index].stop()
            self.video_players[self.active_player_index] = None
            self.video_paths[self.active_player_index] = None
            
            # Reset the label with tactical styling
            feed_num = self.active_player_index + 1
            self.video_labels[self.active_player_index].config(text=f"FEED {feed_num:02d}\n◦ STANDBY ◦")
            self.update_active_player_info()
    
    def start_video_frame_updates(self):
        """Start the video frame update loop for all players"""
        self.update_all_video_frames()
    
    def update_all_video_frames(self):
        """Update video frames for all 4 players"""
        for i in range(4):
            if self.video_players[i]:
                try:
                    # Get frame from queue (non-blocking)
                    compressed_frame = self.frame_queues[i].get_nowait()
                    
                    # Decompress frame
                    frame_data = zlib.decompress(compressed_frame)
                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Use dynamic dimensions based on current player size
                    target_width = getattr(self, 'target_video_width', 380)
                    target_height = getattr(self, 'target_video_height', 280)
                    
                    # Resize frame while maintaining aspect ratio
                    frame_height, frame_width = frame_rgb.shape[:2]
                    scale_w = target_width / frame_width
                    scale_h = target_height / frame_height
                    scale = min(scale_w, scale_h)  # Maintain aspect ratio
                    
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    
                    if new_width > 0 and new_height > 0:
                        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                        
                        # Convert to PIL Image and then to PhotoImage
                        pil_image = Image.fromarray(frame_resized)
                        photo = ImageTk.PhotoImage(pil_image)
                        
                        # Update the label
                        self.video_labels[i].configure(image=photo, text="")
                        self.video_labels[i].image = photo  # Keep a reference
                
                except queue.Empty:
                    pass  # No new frame available
                except Exception as e:
                    logging.error(f"Error updating frame for player {i + 1}: {str(e)}")
        
        # Schedule next update (30 FPS)
        self.root.after(33, self.update_all_video_frames)
    
    def show_context_menu_viewer(self, event):
        """Show context menu for viewers (without delete options)"""
        try:
            # Select the item under cursor
            item = self.video_tree.identify_row(event.y)
            if item:
                self.video_tree.selection_set(item)
                self.context_menu.post(event.x_root, event.y_root)
        except Exception as e:
            logging.error(f"Error showing viewer context menu: {str(e)}")
    
    def previous_video(self):
        """Load previous video to active player"""
        if not self.video_files:
            return
            
        current_selection = self.video_tree.selection()
        if current_selection:
            items = self.video_tree.get_children()
            current_idx = items.index(current_selection[0])
            new_idx = (current_idx - 1) % len(items)
            
            self.video_tree.selection_set(items[new_idx])
            self.video_tree.see(items[new_idx])
            self.load_selected_to_active_player()
    
    def next_video(self):
        """Load next video to active player"""
        if not self.video_files:
            return
            
        current_selection = self.video_tree.selection()
        if current_selection:
            items = self.video_tree.get_children()
            current_idx = items.index(current_selection[0])
            new_idx = (current_idx + 1) % len(items)
            
            self.video_tree.selection_set(items[new_idx])
            self.video_tree.see(items[new_idx])
            self.load_selected_to_active_player()
    
    def toggle_person_detection(self):
        """Toggle person detection for all players"""
        # This would restart all video players with new detection setting
        current_paths = self.video_paths[:]
        for i in range(4):
            if current_paths[i]:
                self.load_video_to_player(current_paths[i], i)
        
        self.update_detection_status()
    
    def update_detection_status(self):
        """Update detection status display with tactical terminology"""
        if PERSON_DETECTION_AVAILABLE:
            if self.detection_enabled.get():
                self.detection_status_var.set("◉ THREAT SCANNER ONLINE")
            else:
                self.detection_status_var.set("◯ THREAT SCANNER OFFLINE")
        else:
            self.detection_status_var.set("⚠ THREAT SCANNER UNAVAILABLE")
    
    def update_player_sizes(self):
        """Update video player sizes based on current window dimensions"""
        try:
            # Get available space for video grid
            self.root.update_idletasks()  # Ensure geometry is updated
            
            # Get the grid container dimensions
            container_width = self.grid_container.winfo_width()
            container_height = self.grid_container.winfo_height()
            
            # Skip if container hasn't been rendered yet
            if container_width <= 1 or container_height <= 1:
                self.root.after(100, self.update_player_sizes)
                return
            
            # Calculate optimal size for each video frame (2x2 grid)
            # Account for padding (4px each side = 8px total per frame, plus 8px between frames)
            available_width = (container_width - 20) // 2  # 20px total padding
            available_height = (container_height - 20) // 2  # 20px total padding
            
            # Set minimum and maximum sizes to prevent too small/large players
            min_size = 200
            max_width = 600
            max_height = 450
            
            player_width = max(min_size, min(available_width, max_width))
            player_height = max(min_size, min(available_height, max_height))
            
            # Maintain 4:3 aspect ratio
            aspect_ratio = 4.0 / 3.0
            if player_width / player_height > aspect_ratio:
                player_width = int(player_height * aspect_ratio)
            else:
                player_height = int(player_width / aspect_ratio)
            
            # Update each video frame with calculated dimensions
            for i, frame in enumerate(self.video_frames):
                frame.configure(width=player_width, height=player_height)
                frame.pack_propagate(False)
                frame.grid_propagate(False)
            
            # Update video processing target dimensions
            self.target_video_width = player_width - 20  # Account for padding
            self.target_video_height = player_height - 20
            
        except Exception as e:
            logging.error(f"Error updating player sizes: {e}")
    
    def on_window_configure(self, event=None):
        """Handle window resize events"""
        # Only update sizes when the main window is resized, not child widgets
        if event and event.widget == self.root:
            # Delay the size update to avoid excessive calls during resize
            if hasattr(self, '_resize_after_id'):
                self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = self.root.after(150, self.update_player_sizes)
    
    def create_upload_tab(self):
        """Create video upload tab with optimized widget"""
        upload_tab = ttk.Frame(self.notebook)
        self.notebook.add(upload_tab, text="📁 Video Upload")
        
        # Split into two columns
        left_frame = ttk.Frame(upload_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_frame = ttk.Frame(upload_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Upload area
        upload_card = tk.Frame(left_frame, bg=self.colors['card_bg'],
                              highlightthickness=1, highlightbackground=self.colors['border'])
        upload_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Card title
        upload_title = tk.Frame(upload_card, bg=self.colors['accent'])
        upload_title.pack(fill='x')
        
        tk.Label(
            upload_title,
            text="Upload Videos",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Upload widget
        self.upload_widget = OptimizedUploadWidget(
            upload_card, 
            self.on_upload_complete
        )
        self.upload_widget.pack(fill='x', padx=15, pady=15)
        
        # Single file upload button
        single_upload_btn = tk.Button(
            upload_card,
            text="Upload Single Video",
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.upload_single_file
        )
        single_upload_btn.pack(fill='x', padx=15, pady=(0, 10))
        
        # In create_upload_tab, after the single_upload_btn code, update/add the folder upload button:

        # Folder upload button (styled to match single video upload)
        folder_upload_btn = tk.Button(
            upload_card,
            text="Upload Video Folder",
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.upload_folder_basic
        )
        folder_upload_btn.pack(fill='x', padx=15, pady=(0, 10))
        # Statistics area
        stats_card = tk.Frame(right_frame, bg=self.colors['card_bg'],
                             highlightthickness=1, highlightbackground=self.colors['border'])
        stats_card.pack(fill='x', expand=False, padx=5, pady=5)
        
        # Stats title
        stats_title = tk.Frame(stats_card, bg=self.colors['accent'], height=40)
        stats_title.pack(fill='x')
        
        tk.Label(
            stats_title,
            text="Upload Statistics",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Stats content
        stats_content = tk.Frame(stats_card, bg=self.colors['card_bg'], padx=10, pady=10)
        stats_content.pack(fill='x')
        
        self.stats_text = tk.Text(stats_content, height=10, bg='#f8fafc')
        stats_scrollbar = ttk.Scrollbar(stats_content, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Previously uploaded videos
        prev_card = tk.Frame(right_frame, bg=self.colors['card_bg'],
                            highlightthickness=1, highlightbackground=self.colors['border'])
        prev_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Previous uploads title
        prev_title = tk.Frame(prev_card, bg=self.colors['accent'], height=40)
        prev_title.pack(fill='x')
        
        tk.Label(
            prev_title,
            text="Previously Uploaded Videos",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Previous uploads content
        prev_content = tk.Frame(prev_card, bg=self.colors['card_bg'], padx=10, pady=10)
        prev_content.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for uploaded videos
        columns = ('Name', 'Upload Time', 'Size', 'Duration', 'Resolution')
        self.prev_video_tree = ttk.Treeview(prev_content, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.prev_video_tree.heading(col, text=col)
            if col == 'Name':
                self.prev_video_tree.column(col, width=150)
            elif col == 'Upload Time':
                self.prev_video_tree.column(col, width=150)
            else:
                self.prev_video_tree.column(col, width=100)
        
        prev_scrollbar = ttk.Scrollbar(prev_content, orient=tk.VERTICAL, command=self.prev_video_tree.yview)
        self.prev_video_tree.configure(yscrollcommand=prev_scrollbar.set)
        
        self.prev_video_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        prev_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load previous videos
        self.load_previous_videos()
    
    def on_upload_complete(self, results: list):
        """Callback when video upload completes"""
        if not results:
            messagebox.showinfo("Info", "No videos were processed")
            return

        success_count = 0
        for video_info in results:
            try:
                # Add to database
                add_video(self.current_user, {
                    'filename': video_info.filename,
                    'filepath': video_info.filepath,
                    'size': video_info.size,
                    'duration': video_info.duration,
                    'width': video_info.width,
                    'height': video_info.height,
                    'fps': video_info.fps
                })
                success_count += 1
            except Exception as e:
                logging.error(f"Error adding video {video_info.filename}: {str(e)}")

        messagebox.showinfo("Upload Complete", 
                           f"Successfully processed {success_count} out of {len(results)} videos")
        self.load_previous_videos()
        self.load_videos_for_playback()
        
    def upload_folder_basic(self):
        """Upload folder using VideoOptimizer for accurate statistics and info"""
        folder_path = filedialog.askdirectory(title="Select Video Folder")
        if folder_path:
            try:
                optimizer = VideoOptimizer()
                results = optimizer.process_folder(folder_path)
                if results:
                    self.add_log(f"Found {len(results)} video files in folder")
                    for video_info in results:
                        add_video(self.current_user, video_info)  # Store in DB
                        self.add_video_to_library(video_info)
                    self.update_upload_statistics(results)
                    self.load_previous_videos()  # Refresh the section
                    messagebox.showinfo("Success", f"Added {len(results)} videos to library and database")
                else:
                    messagebox.showwarning("No Videos", "No supported video files found in the selected folder")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to upload folder: {str(e)}")
                self.add_log(f"Error uploading folder: {str(e)}")
                
    def upload_single_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                optimizer = VideoOptimizer(max_workers=1)
                video_info = optimizer._process_video(file_path)
                if video_info:
                    # Store in DB as VideoInfo object (not dict)
                    add_video(self.current_user, video_info)
                    # Add to library as VideoInfo object
                    self.add_video_to_library(video_info)
                    self.update_upload_statistics([video_info])
                    self.load_previous_videos()
                    messagebox.showinfo("Success", "Video uploaded and stored successfully.")
                    logging.info(f"User {self.current_user} uploaded single video: {video_info.filename}")
                else:
                    messagebox.showerror("Error", "Failed to process the selected video file")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to upload video: {str(e)}")
                logging.error(f"Error uploading single file: {str(e)}")
                
    def update_upload_statistics(self, results: List[VideoInfo]):
        """Update upload statistics display"""
        if not results:
            return
            
        # Calculate statistics
        total_size = sum(v.size for v in results)
        total_duration = sum(v.duration for v in results)
        total_processing_time = sum(v.processing_time for v in results)
        avg_processing_time = total_processing_time / len(results)
        
        # Clear previous stats
        self.stats_text.delete(1.0, tk.END)
        
        # Add new statistics
        stats = f"""UPLOAD STATISTICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
Total Videos Uploaded: {len(results)}
Total File Size: {format_file_size(total_size)}
Total Video Duration: {format_duration(total_duration)}
Total Processing Time: {total_processing_time:.2f} seconds
Average Processing Time: {avg_processing_time:.2f} seconds per video
Processing Speed: {len(results)/total_processing_time:.2f} videos/second

VIDEO DETAILS:
{'='*60}
"""
        
        self.stats_text.insert(tk.END, stats)
        
        for i, video in enumerate(results, 1):
            detail = f"{i:3d}. {video.filename}\n"
            detail += f"     Size: {format_file_size(video.size)}, Duration: {format_duration(video.duration)}\n"
            detail += f"     Resolution: {video.width}x{video.height}, FPS: {video.fps}\n"
            detail += f"     Processing Time: {video.processing_time:.2f}s\n\n"
            self.stats_text.insert(tk.END, detail)
    
    
    def add_log(self, message: str):
        """Add a log message to the logs tab and log file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        if hasattr(self, 'logs_text') and self.logs_text:
            self.logs_text.insert('end', log_entry)
            self.logs_text.see('end')
        logging.info(message)
    def add_video_to_library(self, video_info: VideoInfo):
        """Add video to the library display"""
        self.video_files.append(video_info.filepath)
        index = len(self.video_files)
        duration_str = format_duration(video_info.duration)
        size_str = format_file_size(video_info.size)
        resolution_str = f"{video_info.width}x{video_info.height}"
        self.video_tree.insert('', 'end', 
                          text=str(index),
                          values=(video_info.filename, duration_str, size_str, resolution_str))
        # If this is the first video, select it
        if len(self.video_files) == 1:
            items = self.video_tree.get_children()
            if items:
                self.video_tree.selection_set(items[0])
                self.video_tree.see(items[0])
    
    def add_alert(self, message: str, severity: str = "Info"):
        """Add an alert to the alert text widget and log file, and update counters."""
        if hasattr(self, 'alert_text') and self.alert_text:
            self.alert_text.insert('end', f"[{severity}] {message}\n")
            self.alert_text.see('end')
        logging.info(f"THREAT ALERT: {message}")

        # Update alert statistics
        if severity.lower() == "error" or severity.lower() == "high":
            self.error_count.set(str(int(self.error_count.get()) + 1))
        elif severity.lower() == "warning" or severity.lower() == "medium":
            self.warning_count.set(str(int(self.warning_count.get()) + 1))
        else:
            self.info_count.set(str(int(self.info_count.get()) + 1))
        
    def load_previous_videos(self):
        """Load previously uploaded videos"""
        self.prev_video_tree.delete(*self.prev_video_tree.get_children())
        
        if self.current_role == "Viewer":
            videos = []
            for user in get_all_users():
                role = get_user_role(user)
                if role in ("Admin", "Operator"):
                    videos.extend(get_user_videos(user, self.current_user))
        else:
            videos = get_user_videos(self.current_user, self.current_user)
            
        videos.sort(key=lambda v: v.get('upload_time', ''), reverse=True)
        
        for v in videos:
            duration_str = format_duration(v['duration'])
            size_str = format_file_size(v['size'])
            resolution_str = f"{v['width']}x{v['height']}"
            self.prev_video_tree.insert('', 'end', values=(
                v['filename'],
                v['upload_time'],
                size_str,
                duration_str,
                resolution_str
            ))
    
    def create_face_recognition_tab(self):
        """Create face recognition tab with DeepFace integration"""
        print(f"DEBUG: Starting face recognition tab creation for user: {self.current_user}, role: {getattr(self, 'current_role', 'Unknown')}")
        
        face_tab = ttk.Frame(self.notebook)
        self.notebook.add(face_tab, text="🔍 Face Recognition (BETA)")
        
        print(f"DEBUG: Face recognition tab frame created")
        
        # Main container with padding
        main_frame = tk.Frame(face_tab, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Advanced Face Recognition System",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        )
        title_label.pack(pady=(0, 20))
        
        # Top controls frame
        controls_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Model selection
        model_frame = tk.Frame(controls_frame, bg=self.colors['bg'])
        model_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(
            model_frame,
            text="DeepFace Model:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        ).pack(anchor=tk.W)
        
        self.model_var = tk.StringVar(value="VGG-Face")
        model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=["VGG-Face", "Facenet", "OpenFace", "ArcFace"],
            state="readonly",
            font=("Segoe UI", 10),
            width=15
        )
        model_dropdown.pack(pady=(5, 0))
        
        # Accuracy preset selection
        accuracy_frame = tk.Frame(controls_frame, bg=self.colors['bg'])
        accuracy_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(
            accuracy_frame,
            text="Accuracy Mode:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        ).pack(anchor=tk.W)
        
        self.accuracy_var = tk.StringVar(value="sensitive")
        accuracy_dropdown = ttk.Combobox(
            accuracy_frame,
            textvariable=self.accuracy_var,
            values=["strict", "balanced", "sensitive", "very_sensitive"],
            state="readonly",
            font=("Segoe UI", 10),
            width=15
        )
        accuracy_dropdown.pack(pady=(5, 0))
        accuracy_dropdown.bind('<<ComboboxSelected>>', self.on_accuracy_changed)
        
        # Target image upload
        upload_frame = tk.Frame(controls_frame, bg=self.colors['bg'])
        upload_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(
            upload_frame,
            text="Target Image:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        ).pack(anchor=tk.W)
        
        upload_btn = tk.Button(
            upload_frame,
            text="📁 Upload Image",
            font=("Segoe UI", 10),
            bg=self.colors['accent'],
            fg='black',
            activebackground=self.colors['accent_hover'],
            activeforeground='black',
            relief='flat',
            padx=15,
            pady=5,
            command=self.upload_target_image
        )
        upload_btn.pack(pady=(5, 0))
        
        # Scan controls
        scan_frame = tk.Frame(controls_frame, bg=self.colors['bg'])
        scan_frame.pack(side=tk.LEFT)
        
        tk.Label(
            scan_frame,
            text="Scanning:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text_primary']
        ).pack(anchor=tk.W)
        
        scan_buttons_frame = tk.Frame(scan_frame, bg=self.colors['bg'])
        scan_buttons_frame.pack(pady=(5, 0))
        
        self.start_scan_btn = tk.Button(
            scan_buttons_frame,
            text="🔍 Start Scan",
            font=("Segoe UI", 10),
            bg=self.colors['success'],
            fg='white',
            activebackground='#28a745',
            activeforeground='white',
            relief='flat',
            padx=15,
            pady=5,
            command=self.start_face_scan
        )
        self.start_scan_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_scan_btn = tk.Button(
            scan_buttons_frame,
            text="⏹️ Stop Scan",
            font=("Segoe UI", 10),
            bg=self.colors['danger'],
            fg='white',
            activebackground='#dc3545',
            activeforeground='white',
            relief='flat',
            padx=15,
            pady=5,
            command=self.stop_face_scan,
            state=tk.DISABLED
        )
        self.stop_scan_btn.pack(side=tk.LEFT)
        
        # Main content area with two panels
        content_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Target image
        left_panel = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='solid', bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(
            left_panel,
            text="Target Image",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(pady=10)
        
        # Target image display
        self.target_image_frame = tk.Frame(left_panel, bg=self.colors['bg_secondary'], width=400, height=300)
        self.target_image_frame.pack(padx=20, pady=(0, 20), fill=tk.BOTH, expand=True)
        self.target_image_frame.pack_propagate(False)
        
        self.target_image_label = tk.Label(
            self.target_image_frame,
            text="No target image selected\n\nClick 'Upload Image' to select\na target face for recognition",
            font=("Segoe UI", 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER
        )
        self.target_image_label.pack(expand=True)
        
        # Right panel - CCTV footage
        right_panel = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='solid', bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            right_panel,
            text="CCTV Footage Analysis",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(pady=10)
        
        # CCTV footage display
        self.cctv_frame = tk.Frame(right_panel, bg=self.colors['bg_secondary'], width=400, height=300)
        self.cctv_frame.pack(padx=20, pady=(0, 10), fill=tk.BOTH, expand=True)
        self.cctv_frame.pack_propagate(False)
        
        self.cctv_label = tk.Label(
            self.cctv_frame,
            text="Ready for scanning\n\nStart scan to begin analyzing\nCCTV footage for face matches",
            font=("Segoe UI", 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER
        )
        self.cctv_label.pack(expand=True)
        
        # Results panel at bottom
        results_frame = tk.Frame(right_panel, bg=self.colors['card_bg'])
        results_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Results header with clear button
        results_header = tk.Frame(results_frame, bg=self.colors['card_bg'])
        results_header.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(
            results_header,
            text="Scan Results",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(side=tk.LEFT)
        
        # PDF download button
        self.download_pdf_btn = tk.Button(
            results_header,
            text="📄 Download PDF",
            font=("Segoe UI", 9),
            bg=self.colors['accent'],
            fg='black',
            activebackground=self.colors['accent_hover'],
            activeforeground='black',
            relief='flat',
            padx=10,
            pady=2,
            command=self.download_pdf_report,
            state=tk.DISABLED
        )
        self.download_pdf_btn.pack(side=tk.RIGHT, padx=(0, 5))
        
        clear_btn = tk.Button(
            results_header,
            text="🗑️ Clear",
            font=("Segoe UI", 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['card_bg'],
            activeforeground=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=2,
            command=self.clear_face_results
        )
        clear_btn.pack(side=tk.RIGHT)
        
        # Results text area with scrollbar
        results_container = tk.Frame(results_frame, bg=self.colors['card_bg'])
        results_container.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(
            results_container,
            height=6,
            font=("Consolas", 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        
        results_scrollbar = ttk.Scrollbar(results_container, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        print(f"DEBUG: results_text widget created and packed for user: {self.current_user}, role: {getattr(self, 'current_role', 'Unknown')}")
        print(f"DEBUG: results_text type: {type(self.results_text)}")
        
        # Test the widget immediately
        try:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, f"[TEST] Face recognition tab initialized for {self.current_user} ({getattr(self, 'current_role', 'Unknown')})\n")
            self.results_text.config(state=tk.DISABLED)
            print(f"DEBUG: Test message inserted successfully")
        except Exception as e:
            print(f"DEBUG: Failed to insert test message: {e}")
        
        # Status bar for face recognition
        self.face_status_var = tk.StringVar(value="Ready - Select target image and model to begin scanning")
        status_label = tk.Label(
            main_frame,
            textvariable=self.face_status_var,
            font=("Segoe UI", 10),
            bg=self.colors['bg'],
            fg=self.colors['text_secondary'],
            anchor=tk.W
        )
        status_label.pack(fill=tk.X, pady=(10, 0))

    def on_accuracy_changed(self, event=None):
        """Handle accuracy preset change"""
        try:
            preset = self.accuracy_var.get()
            if hasattr(self.face_engine, 'set_accuracy_preset'):
                success = self.face_engine.set_accuracy_preset(preset)
                if success:
                    # Get current thresholds to show user
                    thresholds = self.face_engine.get_thresholds()
                    self.add_face_result(f"🎯 Accuracy mode set to '{preset}' (Confidence: {thresholds['confidence_threshold']:.2f}, Similarity: {thresholds['similarity_threshold']:.2f})")
                else:
                    self.add_face_result(f"❌ Failed to set accuracy mode to '{preset}'")
        except Exception as e:
            self.add_face_result(f"❌ Error changing accuracy: {str(e)}")

    def upload_target_image(self):
        """Upload target image for face recognition"""
        from tkinter import filedialog
        from PIL import Image, ImageTk
        
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display the image
                image = Image.open(file_path)
                
                # Resize image to fit the display area while maintaining aspect ratio
                display_size = (350, 250)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Create PhotoImage and display
                photo = ImageTk.PhotoImage(image)
                
                self.target_image_label.configure(
                    image=photo,
                    text="",
                    compound=tk.CENTER
                )
                self.target_image_label.image = photo  # Keep a reference
                
                # Store the file path
                self.target_image_path = file_path
                
                # Load the reference image into the face engine for PDF generation
                if hasattr(self, 'face_engine') and self.face_engine:
                    if self.face_engine.load_reference_image(file_path):
                        self.add_face_result(f"✅ Target image loaded into recognition engine")
                    else:
                        self.add_face_result(f"⚠️ Warning: Could not load image into recognition engine")
                
                # Update status
                self.face_status_var.set(f"Target image loaded: {file_path.split('/')[-1]}")
                
                # Add to results
                self.add_face_result(f"✅ Target image loaded: {file_path.split('/')[-1]}")
                
            except Exception as e:
                self.add_face_result(f"❌ Error loading image: {str(e)}")
                self.face_status_var.set("Error loading target image")

    def start_face_scan(self):
        """Start face recognition scanning"""
        if not self.target_image_path:
            self.add_face_result("❌ Please upload a target image first")
            return
            
        if self.scanning_active:
            self.add_face_result("⚠️ Scanning already in progress")
            return
            
        # Update UI state
        self.scanning_active = True
        self.start_scan_btn.config(state=tk.DISABLED)
        self.stop_scan_btn.config(state=tk.NORMAL)
        
        # Update status
        model = self.model_var.get()
        self.face_status_var.set(f"Scanning active - Using {model} model...")
        
        # Apply current accuracy preset
        accuracy_preset = self.accuracy_var.get()
        self.face_engine.set_accuracy_preset(accuracy_preset)
        thresholds = self.face_engine.get_thresholds()
        
        # Add scan start message
        self.add_face_result(f"🔍 Starting face recognition scan with {model} model")
        self.add_face_result(f"📂 Target image: {self.target_image_path.split('/')[-1]}")
        self.add_face_result(f"🎯 Accuracy mode: {accuracy_preset} (Conf: {thresholds['confidence_threshold']:.2f}, Sim: {thresholds['similarity_threshold']:.2f})")
        
        # Start scanning animation
        self.start_scanning_animation()
        
        # Update CCTV display to show scanning is starting
        self.cctv_label.configure(
            text="🔍 Initializing scan...\n\nSearching for videos...",
            image="",
            compound=tk.CENTER,
            bg=self.colors['info'],
            fg='white',
            font=("Segoe UI", 11, "bold")
        )
        
        # Start actual face recognition in a separate thread
        import threading
        scan_thread = threading.Thread(target=self.perform_face_recognition, daemon=True)
        scan_thread.start()

    def stop_face_scan(self):
        """Stop face recognition scanning"""
        self.scanning_active = False
        self.start_scan_btn.config(state=tk.NORMAL)
        self.stop_scan_btn.config(state=tk.DISABLED)
        
        # Stop face recognition engine properly
        if hasattr(self.face_engine, 'stop_recognition'):
            self.face_engine.stop_recognition()
        elif hasattr(self.face_engine, 'stop_scanning_flag'):
            self.face_engine.stop_scanning_flag = True
        
        # Stop animation
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        
        # Check if there are matches for PDF generation
        matches_available = False
        if hasattr(self.face_engine, 'matches_found') and self.face_engine.matches_found:
            matches_available = True
            self.download_pdf_btn.config(state=tk.NORMAL)
            
        # Get summary from engine if available
        summary_text = "Scanning stopped\n\nReady for new scan"
        if hasattr(self.face_engine, 'get_match_summary'):
            try:
                summary = self.face_engine.get_match_summary()
                if summary['total_matches'] > 0:
                    summary_text = f"Scan stopped\n\n{summary['total_matches']} matches found\n{summary['popups_shown']}/3 popups shown"
                    if summary['max_popups_reached']:
                        summary_text += "\n\n⚠️ Max popups reached"
                    summary_text += "\n\n📄 PDF report available"
            except:
                pass
        
        # Reset CCTV display
        display_color = self.colors['success'] if matches_available else self.colors['bg_secondary']
        text_color = 'white' if matches_available else self.colors['text_secondary']
        
        self.cctv_label.configure(
            text=summary_text,
            image="",
            compound=tk.CENTER,
            bg=display_color,
            fg=text_color,
            font=("Segoe UI", 11, "bold" if matches_available else "normal")
        )
        
        # Reset target image display (remove scanning overlay)
        if self.target_image_path:
            try:
                from PIL import Image, ImageTk
                image = Image.open(self.target_image_path)
                display_size = (350, 250)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.target_image_label.configure(
                    image=photo,
                    text="",
                    compound=tk.CENTER,
                    fg=self.colors['text_primary'],
                    font=("Segoe UI", 11)
                )
                self.target_image_label.image = photo
            except Exception:
                pass  # Keep current state if reset fails
        
        status_msg = "Scanning stopped - Ready for new scan"
        if matches_available:
            status_msg = "Scanning stopped - Results available, PDF ready for download"
            
        self.face_status_var.set(status_msg)
        self.add_face_result("⏹️ Real-time face recognition pipeline stopped")
        
        if matches_available:
            self.add_face_result("📄 PDF report is ready for download")

    def start_scanning_animation(self):
        """Start the scanning animation effect on target image"""
        if not self.scanning_active:
            return
            
        # Update scan line position
        self.scan_line_position += self.scan_direction * 8
        
        # Reverse direction at boundaries
        if self.scan_line_position >= 280 or self.scan_line_position <= 0:
            self.scan_direction *= -1
            
        # Add scanning overlay to target image if it exists
        if hasattr(self.target_image_label, 'image') and self.target_image_label.image:
            try:
                # Create scanning line effect on target image
                from PIL import Image, ImageDraw, ImageTk
                
                # Get original image
                original_image = self.target_image_label.image._PhotoImage__photo.copy()
                
                # Add scanning line overlay
                overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Draw scanning line
                line_y = int((self.scan_line_position / 280) * original_image.height)
                draw.rectangle([0, line_y-2, original_image.width, line_y+2], fill=(0, 255, 0, 150))
                
                # Composite the overlay
                result = Image.alpha_composite(original_image.convert('RGBA'), overlay)
                photo = ImageTk.PhotoImage(result.convert('RGB'))
                
                self.target_image_label.configure(image=photo)
                self.target_image_label.image = photo
                
            except Exception:
                # Fallback to text-based animation
                progress_percent = (self.scan_line_position / 280) * 100
                scan_overlay = f"🔍 SCANNING TARGET\n\nProgress: {progress_percent:.0f}%"
                
                # Add visual progress bar
                bar_length = 20
                filled = int((progress_percent / 100) * bar_length)
                progress_bar = "█" * filled + "░" * (bar_length - filled)
                scan_overlay += f"\n[{progress_bar}]"
                
                self.target_image_label.configure(
                    text=scan_overlay,
                    compound=tk.CENTER,
                    fg='lime',
                    font=("Segoe UI", 10, "bold")
                )
        else:
            # No target image, show loading animation
            progress_percent = (self.scan_line_position / 280) * 100
            dots = "." * (int(progress_percent / 25) % 4)
            
            self.target_image_label.configure(
                text=f"🔍 Ready to scan{dots}\n\nWaiting for target image...",
                fg=self.colors['accent'],
                font=("Segoe UI", 11)
            )
        
        # Schedule next animation frame
        self.animation_after_id = self.root.after(150, self.start_scanning_animation)

    def perform_face_recognition(self):
        """Perform real-time face recognition using advanced pipeline"""
        import cv2
        import os
        from PIL import Image, ImageTk
        
        try:
            model = self.model_var.get()
            
            # Update UI from main thread  
            self.root.after(0, lambda: self.add_face_result(f"🔄 Initializing {model} model..."))
            
            # Set up face recognition engine
            self.face_engine.set_model(model)
            self.root.after(0, lambda: self.add_face_result(f"🔄 Loading reference image: {self.target_image_path}"))
            
            if not self.face_engine.load_reference_image(self.target_image_path):
                self.root.after(0, lambda: self.add_face_result("❌ Failed to load reference image"))
                self.root.after(0, self.stop_face_scan)
                return
            
            self.root.after(0, lambda: self.add_face_result("✅ Reference image loaded successfully"))
            
            # Get video files for scanning
            self.root.after(0, lambda: self.add_face_result(f"🔍 Searching for videos for user: {self.current_user}"))
            # Test debugging first
            self.test_debug_output()
            
            # Use simplified video search
            videos = self.get_available_videos_simple()
            
            if not videos:
                self.root.after(0, lambda: self.add_face_result("❌ No video files found for scanning"))
                self.root.after(0, lambda: self.add_face_result("💡 Please upload some videos first in the Video Upload tab"))
                self.root.after(0, self.stop_face_scan)
                return
                
            self.root.after(0, lambda: self.add_face_result(f"📹 Found {len(videos)} video files to scan"))
            self.root.after(0, lambda: self.add_face_result("🚀 Starting real-time face recognition pipeline..."))
            
            # Process each video with real-time pipeline
            total_matches = 0
            
            for i, video_path in enumerate(videos):
                if not self.scanning_active:
                    self.root.after(0, lambda: self.add_face_result("⏹️ Scanning stopped by user"))
                    break
                
                # Check if engine has reached max popups before processing next video
                if hasattr(self.face_engine, 'popups_shown') and hasattr(self.face_engine, 'max_popups'):
                    if self.face_engine.popups_shown >= self.face_engine.max_popups:
                        self.root.after(0, lambda: self.add_face_result(f"🎯 Max popups ({self.face_engine.max_popups}) reached, stopping entire scan"))
                        self.scanning_active = False
                        break
                    
                video_name = os.path.basename(video_path)
                
                # Update progress
                current_video = video_name
                video_index = i + 1
                total_videos = len(videos)
                self.root.after(0, lambda v=current_video, idx=video_index, tot=total_videos: 
                              self.add_face_result(f"🎬 Processing video {idx}/{tot}: {v}"))
                
                # Start real-time recognition for this video
                self.root.after(0, lambda v=video_name: self.add_face_result(f"🔄 Attempting to start recognition for: {v}"))
                
                if self.face_engine.start_real_time_recognition(video_path):
                    
                    # Monitor video playback and display frames
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        self.root.after(0, lambda v=video_name, d=duration, f=frame_count: 
                                      self.add_face_result(f"📊 Video info: {v} - {d:.1f}s duration, {f} frames"))
                        
                        frame_number = 0
                        
                        while cap.isOpened() and self.scanning_active:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Check if engine has reached max popups and should stop
                            if hasattr(self.face_engine, 'popups_shown') and hasattr(self.face_engine, 'max_popups'):
                                if self.face_engine.popups_shown >= self.face_engine.max_popups:
                                    self.root.after(0, lambda: self.add_face_result(f"🎯 Max popups ({self.face_engine.max_popups}) reached, stopping scan"))
                                    self.scanning_active = False
                                    break
                            
                            # Check engine's stop flag
                            if hasattr(self.face_engine, 'stop_scanning_flag') and self.face_engine.stop_scanning_flag:
                                self.root.after(0, lambda: self.add_face_result("🛑 Engine stop flag triggered, stopping scan"))
                                self.scanning_active = False
                                break
                                
                            # Update display every 10 frames for smooth playback
                            if frame_number % 10 == 0:
                                progress = (frame_number / frame_count) * 100 if frame_count > 0 else 0
                                timestamp = frame_number / fps if fps > 0 else frame_number / 30
                                
                                # Display current frame in CCTV panel
                                self.update_cctv_display(frame.copy(), video_name, timestamp, progress)
                                
                                # Update status
                                current_progress = progress
                                status_video = video_name
                                self.root.after(0, lambda p=current_progress, v=status_video: 
                                              self.face_status_var.set(f"Real-time scanning {v} - {p:.1f}% complete"))
                                
                                # Also check popup count during display updates
                                if hasattr(self.face_engine, 'popups_shown'):
                                    current_popups = self.face_engine.popups_shown
                                    self.root.after(0, lambda p=current_popups: 
                                                  self.add_face_result(f"🔔 Popups shown: {p}/3"))
                            
                            frame_number += 1
                            
                            # Control playback speed
                            time.sleep(1/fps if fps > 0 else 0.033)
                        
                        cap.release()
                        
                        # Get matches found for this video
                        video_matches = len(self.face_engine.matches_found)
                        total_matches += video_matches
                        
                        if video_matches > 0:
                            completed_video = video_name
                            self.root.after(0, lambda v=completed_video, m=video_matches: 
                                          self.add_face_result(f"✅ {v} complete: {m} matches found with real-time detection"))
                        else:
                            no_match_video = video_name
                            self.root.after(0, lambda v=no_match_video: 
                                          self.add_face_result(f"❌ No matches found in {v}"))
                    
                    # Stop the engine's scanning for this video
                    self.face_engine.stop_scanning_flag = True
                    time.sleep(0.5)  # Allow cleanup
                    self.face_engine.stop_scanning_flag = False
                
                else:
                    error_video = video_name
                    self.root.after(0, lambda v=error_video: 
                                  self.add_face_result(f"❌ Failed to start real-time recognition for {v}"))
            
            # Final results
            if self.scanning_active:
                # Get match summary from engine
                summary = self.face_engine.get_match_summary()
                total_detections = summary['total_matches']
                popups_shown = summary['popups_shown']
                max_popups_reached = summary['max_popups_reached']
                
                self.root.after(0, lambda: self.add_face_result(f"🎯 Real-time scan complete!"))
                self.root.after(0, lambda: self.add_face_result(f"📊 Total face detections: {total_detections}"))
                self.root.after(0, lambda: self.add_face_result(f"🔔 Popups shown: {popups_shown}/3 (max limit)"))
                
                if max_popups_reached:
                    self.root.after(0, lambda: self.add_face_result("⚠️ Scanning stopped - Maximum popups reached"))
                
                if total_detections > 0:
                    self.root.after(0, lambda: self.add_face_result("📄 PDF report ready for download"))
                    self.root.after(0, lambda: self.download_pdf_btn.config(state=tk.NORMAL))
                
                self.root.after(0, lambda: self.face_status_var.set(f"Scan complete - {total_detections} faces detected, {popups_shown} popups shown"))
                
                # Reset CCTV display with final results
                final_color = self.colors['success'] if total_detections > 0 else self.colors['bg_secondary']
                status_text = f"🎯 Real-time Scan Complete!\n\n{total_detections} faces detected\nacross {len(videos)} videos"
                
                if max_popups_reached:
                    status_text += f"\n\n⚠️ Stopped at {popups_shown} popups (max limit)"
                elif popups_shown > 0:
                    status_text += f"\n\n🔔 {popups_shown} popups shown"
                
                if total_detections > 0:
                    status_text += "\n\n📄 PDF report available"
                
                self.root.after(0, lambda: self.cctv_label.configure(
                    text=status_text,
                    image="",
                    compound=tk.CENTER,
                    bg=final_color,
                    fg='white' if total_detections > 0 else self.colors['text_secondary'],
                    font=("Segoe UI", 11, "bold" if total_detections > 0 else "normal")
                ))
            
        except Exception as e:
            self.root.after(0, lambda: self.add_face_result(f"❌ Face recognition error: {str(e)}"))
            self.root.after(0, lambda: self.face_status_var.set("Error during face recognition"))
        finally:
            # Ensure UI is reset
            if self.scanning_active:
                self.root.after(0, self.stop_face_scan)

    def update_cctv_display(self, frame, video_name, timestamp, progress):
        """Thread-safe method to update CCTV display with current frame"""
        from PIL import Image, ImageTk
        import cv2
        
        def _update_display():
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize to fit display area
                display_size = (350, 250)
                pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # Create display text
                display_text = f"🎬 {video_name}\n⏰ {timestamp:.1f}s\n📊 {progress:.1f}% complete"
                
                # Update the label
                self.cctv_label.configure(
                    image=photo,
                    text=display_text,
                    compound=tk.TOP,
                    bg=self.colors['accent'],
                    fg='white',
                    font=("Segoe UI", 10, "bold")
                )
                self.cctv_label.image = photo  # Keep reference
                
            except Exception as e:
                # Fallback to text display if image processing fails
                self.cctv_label.configure(
                    text=f"🎬 {video_name}\n⏰ {timestamp:.1f}s\n📊 {progress:.1f}% complete\n\n[Frame display error: {str(e)}]",
                    image="",
                    compound=tk.CENTER,
                    bg=self.colors['warning'],
                    fg='black',
                    font=("Segoe UI", 10)
                )
        
        # Schedule the update in the main thread
        self.root.after(0, _update_display)

    def display_current_frame(self, frame, video_name, timestamp, progress):
        """Display the current video frame being analyzed (legacy method - kept for compatibility)"""
        self.update_cctv_display(frame, video_name, timestamp, progress)

    def get_available_videos(self):
        """Get list of available video files for scanning from database"""
        import os
        
        # Test debugging output first
        self.add_face_result("=== STARTING VIDEO SEARCH ===")
        self.add_face_result(f"Current user: {self.current_user}")
        
        try:
            # Get user's videos from database
            self.add_face_result("Calling get_user_videos...")
            user_videos = get_user_videos(self.current_user)
            self.add_face_result("get_user_videos completed")
            
            # Debug information about the returned data
            self.root.after(0, lambda: self.add_face_result(f"� Raw database result type: {type(user_videos)}"))
            self.root.after(0, lambda: self.add_face_result(f"🔍 Raw database result: {str(user_videos)[:200]}..."))
            
            # Check if it's actually a list
            if not isinstance(user_videos, list):
                self.root.after(0, lambda: self.add_face_result(f"❌ Database returned unexpected type: {type(user_videos)}"))
                self.root.after(0, lambda: self.add_face_result(f"❌ Expected list, got: {user_videos}"))
                return []
            
            self.root.after(0, lambda: self.add_face_result(f"�📊 Database returned {len(user_videos)} video records"))
            
            video_files = []
            
            for i, video in enumerate(user_videos):
                try:
                    # Debug: Check what each video record looks like
                    self.root.after(0, lambda v=str(video), idx=i: self.add_face_result(f"🔍 Video {idx}: {v}"))
                    
                    # Check if video is a dictionary
                    if not isinstance(video, dict):
                        self.root.after(0, lambda v=video, idx=i: self.add_face_result(f"❌ Video {idx} is not a dict: {type(v)} = {v}"))
                        continue
                    
                    # Check required keys
                    if 'filepath' not in video or 'filename' not in video:
                        self.root.after(0, lambda v=video, idx=i: self.add_face_result(f"❌ Video {idx} missing keys: {list(v.keys())}"))
                        continue
                    
                    # video is a dictionary: {'id', 'filename', 'filepath', 'size', 'duration', 'width', 'height', 'fps', 'upload_time'}
                    filepath = video['filepath']  # get filepath from dict
                    filename = video['filename']  # get filename from dict
                    
                    # Create local copies for lambda capture
                    check_filename = filename
                    check_filepath = filepath
                    self.root.after(0, lambda f=check_filename, p=check_filepath: 
                                  self.add_face_result(f"📁 Checking video: {f} at path: {p}"))
                    
                    # Check if file actually exists
                    if os.path.exists(filepath):
                        video_files.append(filepath)
                        found_filename = filename
                        self.root.after(0, lambda f=found_filename: 
                                      self.add_face_result(f"✅ Video found: {f}"))
                    else:
                        missing_filename = filename
                        missing_filepath = filepath
                        self.root.after(0, lambda f=missing_filename, p=missing_filepath: 
                                      self.add_face_result(f"❌ Video file not found: {f} at {p}"))
                        
                except Exception as video_error:
                    error_msg = str(video_error)
                    self.root.after(0, lambda e=error_msg, idx=i: self.add_face_result(f"❌ Error processing video {idx}: {e}"))
                    continue
            
            self.root.after(0, lambda: self.add_face_result(f"📈 Total available videos for scanning: {len(video_files)}"))
            return video_files
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            full_traceback = traceback.format_exc()
            self.root.after(0, lambda: self.add_face_result(f"❌ Error getting videos from database: {error_msg}"))
            self.root.after(0, lambda: self.add_face_result(f"🔍 Full error details: {full_traceback}"))
            return []

    def add_face_result(self, message):
        """Add a message to the face recognition results"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        try:
            # Debug: Check if results_text widget exists
            if not hasattr(self, 'results_text'):
                print(f"ERROR: results_text widget does not exist! User: {self.current_user}, Role: {getattr(self, 'current_role', 'Unknown')}")
                return
            
            if self.results_text is None:
                print(f"ERROR: results_text is None! User: {self.current_user}, Role: {getattr(self, 'current_role', 'Unknown')}")
                return
                
            # Check if widget still exists (not destroyed)
            try:
                self.results_text.winfo_exists()
            except tk.TclError:
                print(f"ERROR: results_text widget was destroyed! User: {self.current_user}, Role: {getattr(self, 'current_role', 'Unknown')}")
                return
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, formatted_message)
            self.results_text.see(tk.END)  # Scroll to bottom
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"ERROR: Could not add to results text: {e}")
            print(f"User: {self.current_user}, Role: {getattr(self, 'current_role', 'Unknown')}")
            print(f"Message was: {formatted_message}")
            print(f"results_text exists: {hasattr(self, 'results_text')}")
            if hasattr(self, 'results_text'):
                print(f"results_text type: {type(self.results_text)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

    def test_debug_output(self):
        """Test method to verify debugging output works"""
        self.add_face_result("=== DEBUG TEST ===")
        self.add_face_result("If you see this, debugging is working!")
        self.add_face_result(f"Current user: {self.current_user}")
        
        # Test face engine status
        if hasattr(self, 'face_engine'):
            self.add_face_result(f"Face engine exists: {type(self.face_engine)}")
            self.add_face_result(f"Reference image path: {getattr(self.face_engine, 'reference_image_path', 'None')}")
            
            # Test if DEEPFACE is available
            try:
                from deepface import DeepFace
                self.add_face_result("✅ DeepFace is available")
            except ImportError:
                self.add_face_result("❌ DeepFace is not available")
        else:
            self.add_face_result("❌ Face engine not found")
            
        self.add_face_result("=== END DEBUG TEST ===")

    def get_available_videos_simple(self):
        """Simplified version for debugging"""
        import os
        
        self.add_face_result("=== SIMPLE VIDEO SEARCH ===")
        
        try:
            from local_db import get_user_videos
            self.add_face_result("Imported get_user_videos successfully")
            
            # For admin users, pass requesting_user parameter to see all videos
            user_videos = get_user_videos(self.current_user, requesting_user=self.current_user)
            self.add_face_result(f"get_user_videos returned: {type(user_videos)}")
            # Get actual role from database
            from local_db import get_user_role
            actual_role = get_user_role(self.current_user)
            self.add_face_result(f"User role from GUI: {getattr(self, 'current_role', 'Unknown')}")
            self.add_face_result(f"User role from DB: {actual_role}")
            
            if user_videos is None:
                self.add_face_result("ERROR: get_user_videos returned None")
                return []
            
            if not isinstance(user_videos, list):
                self.add_face_result(f"ERROR: Expected list, got {type(user_videos)}: {user_videos}")
                return []
                
            self.add_face_result(f"SUCCESS: Found {len(user_videos)} videos")
            
            video_files = []
            for i, video in enumerate(user_videos):
                self.add_face_result(f"Video {i}: {video}")
                if isinstance(video, dict) and 'filepath' in video:
                    filepath = video['filepath']
                    if os.path.exists(filepath):
                        video_files.append(filepath)
                        self.add_face_result(f"Added video: {video.get('filename', 'Unknown')}")
                    else:
                        self.add_face_result(f"File not found: {filepath}")
                else:
                    self.add_face_result(f"Invalid video format: {video}")
                    
            self.add_face_result(f"Total valid videos: {len(video_files)}")
            return video_files
            
        except Exception as e:
            import traceback
            self.add_face_result(f"EXCEPTION: {str(e)}")
            self.add_face_result(f"TRACEBACK: {traceback.format_exc()}")
            return []

    def clear_face_results(self):
        """Clear the face recognition results display"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Reset engine state and disable PDF button
        if hasattr(self.face_engine, 'reset'):
            self.face_engine.reset()
        self.download_pdf_btn.config(state=tk.DISABLED)
        
        # Add clear notification
        self.add_face_result("🧹 Results cleared - Engine reset")
    
    def download_pdf_report(self):
        """Download PDF report of face recognition results"""
        try:
            if not hasattr(self.face_engine, 'matches_found') or not self.face_engine.matches_found:
                messagebox.showwarning("No Results", "No face recognition results available for PDF generation.")
                return
            
            # Ask user where to save the PDF
            from tkinter import filedialog
            import os
            from datetime import datetime
            
            default_filename = f"face_recognition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Face Recognition Report",
                defaultextension=".pdf",
                initialfile=default_filename,
                filetypes=[
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Show progress
                self.add_face_result("📄 Generating PDF report...")
                self.face_status_var.set("Generating PDF report...")
                
                # Generate PDF using engine
                def generate_pdf():
                    try:
                        print(f"DEBUG: Starting PDF generation for path: {file_path}")
                        result_path = self.face_engine.generate_pdf_report(file_path)
                        print(f"DEBUG: PDF generation completed. Result: {result_path}")
                        
                        if result_path:
                            # Success - update UI from main thread
                            self.root.after(0, lambda: self.add_face_result(f"✅ PDF report saved: {os.path.basename(result_path)}"))
                            self.root.after(0, lambda: self.face_status_var.set(f"PDF report saved successfully"))
                            self.root.after(0, lambda: messagebox.showinfo(
                                "PDF Generated", 
                                f"Face recognition report has been saved to:\n\n{result_path}\n\nThe report includes detailed analysis of all face matches found during the scan."
                            ))
                        else:
                            # Error - update UI from main thread
                            self.root.after(0, lambda: self.add_face_result("❌ Failed to generate PDF report"))
                            self.root.after(0, lambda: self.face_status_var.set("PDF generation failed"))
                            self.root.after(0, lambda: messagebox.showerror(
                                "PDF Error", 
                                "Failed to generate PDF report. Please check the logs for more details."
                            ))
                    
                    except Exception as e:
                        error_msg = str(e)
                        self.root.after(0, lambda: self.add_face_result(f"❌ PDF generation error: {error_msg}"))
                        self.root.after(0, lambda: self.face_status_var.set("PDF generation error"))
                        self.root.after(0, lambda: messagebox.showerror("PDF Error", f"Error generating PDF report:\n\n{error_msg}"))
                
                # Run PDF generation in background thread
                import threading
                pdf_thread = threading.Thread(target=generate_pdf, daemon=True)
                pdf_thread.start()
        
        except Exception as e:
            self.add_face_result(f"❌ PDF download error: {str(e)}")
            messagebox.showerror("Download Error", f"Error initiating PDF download:\n\n{str(e)}")

    def create_security_tab(self):
        """Create security settings tab"""
        security_tab = ttk.Frame(self.notebook)
        self.notebook.add(security_tab, text="🔐 Security")
        
        # Split into two columns
        left_column = ttk.Frame(security_tab)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_column = ttk.Frame(security_tab)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # User Management Card
        user_card = tk.Frame(left_column, bg=self.colors['card_bg'],
                            highlightthickness=1, highlightbackground=self.colors['border'])
        user_card.pack(fill='x', padx=5, pady=5)
        
        # Card title
        user_title = tk.Frame(user_card, bg=self.colors['accent'])
        user_title.pack(fill='x')
        
        tk.Label(
            user_title,
            text="User Management",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # User content
        user_content = tk.Frame(user_card, bg=self.colors['card_bg'], padx=15, pady=15)
        user_content.pack(fill='x')
        
        # User info grid
        user_grid = tk.Frame(user_content, bg=self.colors['card_bg'])
        user_grid.pack(fill='x')
        
        tk.Label(user_grid, text="Current User:", bg=self.colors['card_bg'],
                fg=self.colors['text_secondary']).grid(row=0, column=0, sticky='w', pady=5)
        tk.Label(user_grid, text=self.current_user, bg=self.colors['card_bg'],
                fg=self.colors['text_primary'], font=('Segoe UI', 11, 'bold')).grid(row=0, column=1, sticky='w', pady=5, padx=10)
        
        tk.Label(user_grid, text="Role:", bg=self.colors['card_bg'],
                fg=self.colors['text_secondary']).grid(row=1, column=0, sticky='w', pady=5)
        
        # Role badge
        role_frame = tk.Frame(user_grid, bg=self.colors['card_bg'])
        role_frame.grid(row=1, column=1, sticky='w', pady=5, padx=10)
        
        role_colors = {
            "Admin": "#f43f5e",
            "Operator": "#f59e0b",
            "Viewer": "#3b82f6"
        }
        role_color = role_colors.get(self.current_role, "#64748b")
        
        role_badge = tk.Label(
            role_frame,
            text=self.current_role,
            font=('Segoe UI', 10, 'bold'),
            bg=role_color,
            fg='white',
            padx=8,
            pady=2
        )
        role_badge.pack()
        
        # Role Management
        if self.current_role == "Admin":
            tk.Label(user_grid, text="Change User Role:", bg=self.colors['card_bg'],
                    fg=self.colors['text_primary']).grid(row=2, column=0, sticky='w', pady=5)
            
            self.role_user_var = tk.StringVar()
            self.role_user_combo = ttk.Combobox(user_grid, textvariable=self.role_user_var, 
                                               values=get_all_users(), state='readonly', width=25)
            self.role_user_combo.grid(row=2, column=1, padx=10, pady=5, sticky='w')
            
            tk.Label(user_grid, text="New Role:", bg=self.colors['card_bg'],
                    fg=self.colors['text_primary']).grid(row=3, column=0, sticky='w', pady=5)
            
            self.role_new_var = tk.StringVar()
            self.role_new_combo = ttk.Combobox(user_grid, textvariable=self.role_new_var,
                                              values=["Admin", "Operator", "Viewer"], state='readonly', width=25)
            self.role_new_combo.grid(row=3, column=1, padx=10, pady=5, sticky='w')
            
            # Update role button
            update_role_btn = tk.Button(
                user_content,
                text="Update Role",
                font=('Segoe UI', 11),
                bg=self.colors['accent'],
                fg='white',
                activebackground=self.colors['accent_hover'],
                activeforeground='white',
                bd=0,
                padx=15,
                pady=8,
                cursor="hand2",
                command=self.update_user_role
            )
            update_role_btn.pack(pady=10)
        
        # Encryption Status Card
        encryption_card = tk.Frame(left_column, bg=self.colors['card_bg'],
                                 highlightthickness=1, highlightbackground=self.colors['border'])
        encryption_card.pack(fill='x', padx=5, pady=5)
        
        # Card title
        encryption_title = tk.Frame(encryption_card, bg=self.colors['accent'], height=40)
        encryption_title.pack(fill='x')
        
        tk.Label(
            encryption_title,
            text="Encryption Status",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Encryption content
        encryption_content = tk.Frame(encryption_card, bg=self.colors['card_bg'], padx=15, pady=15)
        encryption_content.pack(fill='x')
        
        # Encryption grid
        encryption_grid = tk.Frame(encryption_content, bg=self.colors['card_bg'])
        encryption_grid.pack(fill='x')
        
        # Data Encryption
        tk.Label(encryption_grid, text="Data Encryption:", 
                bg=self.colors['card_bg'], fg=self.colors['text_primary']).grid(row=0, column=0, sticky='w', pady=5)
        tk.Label(encryption_grid, text="✅ Active", 
                bg=self.colors['card_bg'], fg=self.colors['success'], 
                font=('Segoe UI', 11)).grid(row=0, column=1, sticky='w', pady=5, padx=10)
        
        # Video Encryption
        tk.Label(encryption_grid, text="Video Encryption:", 
                bg=self.colors['card_bg'], fg=self.colors['text_primary']).grid(row=1, column=0, sticky='w', pady=5)
        tk.Label(encryption_grid, text="✅ Enabled", 
                bg=self.colors['card_bg'], fg=self.colors['success'], 
                font=('Segoe UI', 11)).grid(row=1, column=1, sticky='w', pady=5, padx=10)
        
        # SSL/TLS Encryption
        tk.Label(encryption_grid, text="TLS/DTLS Encryption:", 
                bg=self.colors['card_bg'], fg=self.colors['text_primary']).grid(row=2, column=0, sticky='w', pady=5)
        
        ssl_frame = tk.Frame(encryption_grid, bg=self.colors['card_bg'])
        ssl_frame.grid(row=2, column=1, sticky='w', pady=5, padx=10)
        
        self.ssl_var = tk.BooleanVar(value=True)
        self.ssl_checkbox = ttk.Checkbutton(ssl_frame, text="Enabled", variable=self.ssl_var, 
                                          command=self.toggle_ssl_encryption)
        self.ssl_checkbox.pack(side='left')
        
        # After SSL/TLS Encryption row in encryption_grid
        
        #VPN Status 
        # Check VPN connection status
        tk.Label(encryption_grid, text="VPN Status:", 
                bg=self.colors['card_bg'], fg=self.colors['text_primary']).grid(row=3, column=0, sticky='w', pady=5)
        self.vpn_status_label = tk.Label(encryption_grid, text="Checking...", 
                bg=self.colors['card_bg'], fg=self.colors['warning'], font=('Segoe UI', 11))
        self.vpn_status_label.grid(row=3, column=1, sticky='w', pady=5, padx=10)
        
        # Certificate status
        cert_status = self.cert_manager.get_certificate_status()
        if cert_status['valid']:
            days = cert_status['days_until_expiry']
            if days is not None and days < 15:
                cert_text = f"⚠️ Expires in {days} days"
                cert_color = self.colors['warning']
            else:
                cert_text = f"✅ Valid (expires {cert_status['expiry'].date()})"
                cert_color = self.colors['success']
        else:
            cert_text = "❌ Invalid or Not Found"
            cert_color = self.colors['error']

        tk.Label(encryption_grid, text="Certificate Status:",
                bg=self.colors['card_bg'], fg=self.colors['text_primary']).grid(row=4, column=0, sticky='w', pady=5)
        tk.Label(encryption_grid, text=cert_text,
                bg=self.colors['card_bg'], fg=cert_color, font=('Segoe UI', 11)).grid(row=4, column=1, sticky='w', pady=5, padx=10)
        
        # System Monitoring Card
        monitor_card = tk.Frame(right_column, bg=self.colors['card_bg'],
                               highlightthickness=1, highlightbackground=self.colors['border'])
        monitor_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Card title
        monitor_title = tk.Frame(monitor_card, bg=self.colors['accent'], height=40)
        monitor_title.pack(fill='x')
        
        tk.Label(
            monitor_title,
            text="System Monitoring",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Monitor content
        monitor_content = tk.Frame(monitor_card, bg=self.colors['card_bg'], padx=15, pady=15)
        monitor_content.pack(fill=tk.BOTH, expand=True)
        
        # Description text
        desc = "Live system monitoring and security events.\n"
        if self.current_role == "Admin":
            desc += "You see all system events."
        elif self.current_role == "Operator":
            desc += "You see system and your own events."
        else:
            desc += "You see your own activity only."
            
        desc_label = tk.Label(
            monitor_content,
            text=desc,
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['info'],
            justify='left'
        )
        desc_label.pack(anchor='w', pady=(0, 10))
        
        # Monitor text
        self.monitor_text = tk.Text(monitor_content, height=20, bg='#f8fafc')
        monitor_scrollbar = ttk.Scrollbar(monitor_content, orient=tk.VERTICAL, command=self.monitor_text.yview)
        self.monitor_text.configure(yscrollcommand=monitor_scrollbar.set)
        
        self.monitor_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        monitor_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add initial monitoring message
        self.monitor_text.insert(tk.END, "System monitoring started...\n")
        self.monitor_text.insert(tk.END, f"User {self.current_user} logged in\n")
        self.monitor_text.insert(tk.END, "All systems operational\n")
        # Add this after user info grid in create_security_tab
        reset_btn = tk.Button(
            user_content,
            text="Reset Authenticator",
            font=('Segoe UI', 11),
            bg=self.colors['error'],
            fg='white',
            command=lambda: self.reset_totp_secret(self.current_user)
        )
        reset_btn.pack(pady=10)
        
        
    # Add this to your update_vpn_status method (replace the method with this version):

    def update_vpn_status(self):
        """Update VPN status label in the security tab, log status changes, and show threat alert if VPN is down."""
        connected = is_vpn_connected()
        if hasattr(self, "vpn_status_label"):
            if connected:
                self.vpn_status_label.config(text="✅ VPN Connected", fg=self.colors['success'])
            else:
                self.vpn_status_label.config(text="❌ VPN Not Connected", fg=self.colors['error'])
        # Log only on status change
        if self.vpn_last_status is None or self.vpn_last_status != connected:
            if connected:
                logging.info("VPN connected (10.0.0.2 detected)")
                self.vpn_warned = False  # Reset warning flag when VPN comes back
            else:
                logging.warning("VPN disconnected (10.0.0.2 not detected)")
                # Show warning in Threat Intelligence tab only once per disconnect
                if not self.vpn_warned and hasattr(self, "add_alert"):
                    self.add_alert("VPN is not connected. Secure transmission is not active.", severity="Medium")
                    self.vpn_warned = True
            self.vpn_last_status = connected
        
    def reset_totp_secret(self, username):
        import sqlite3
        conn = sqlite3.connect("secure_cctv.db")
        c = conn.cursor()
        c.execute("UPDATE users SET totp_secret=NULL WHERE username=?", (username,))
        conn.commit()
        conn.close()
        messagebox.showinfo("Reset", "Authenticator setup has been reset. Please log out and set up again.")
    
    def update_user_role(self):
        """Update user role"""
        user = self.role_user_var.get()
        new_role = self.role_new_var.get()
        if not user or not new_role:
            messagebox.showerror("Error", "Please select a user and a new role.")
            return
            
        if user == self.current_user:
            messagebox.showwarning("Warning", "You cannot change your own role.")
            return
            
        try:
            # Update role in database
            conn = sqlite3.connect("secure_cctv.db")
            c = conn.cursor()
            c.execute("UPDATE users SET role=? WHERE username=?", (new_role, user))
            conn.commit()
            conn.close()
            
            messagebox.showinfo("Success", f"Role for {user} updated to {new_role}")
            logging.info(f"Changed role for {user} to {new_role}")
            self.monitor_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Role updated for {user}\n")
            
            # Refresh user list
            self.role_user_combo['values'] = get_all_users()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update role: {str(e)}")
            logging.error(f"Error updating role: {str(e)}")
    
    def toggle_ssl_encryption(self):
        """Toggle SSL encryption"""
        if self.ssl_var.get():
            self.monitor_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] TLS encryption enabled\n")
            logging.info("TLS encryption enabled")
        else:
            self.monitor_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] TLS encryption disabled\n")
            logging.warning("TLS encryption disabled")
    
    def create_threat_tab(self):
        """Create threat intelligence tab with System Intelligence and AI Threat Detection subtabs"""
        threat_tab = ttk.Frame(self.notebook)
        self.notebook.add(threat_tab, text="🛡️ Threat Intelligence")
        
        # Create main notebook for threat intelligence subtabs
        threat_notebook = ttk.Notebook(threat_tab)
        
        # Configure subtab styling
        style = ttk.Style()
        style.configure('Subtab.TNotebook.Tab', 
                       font=('Segoe UI', 11, 'bold'),
                       padding=[15, 8],
                       foreground='black')  # Default text color
        style.map('Subtab.TNotebook.Tab',
                 background=[('selected', self.colors['info']),
                           ('active', self.colors['bg_secondary']),
                           ('!selected', 'SystemButtonFace')],
                 foreground=[('selected', 'black'),
                           ('active', 'black'),
                           ('!selected', 'black')])
        
        # Apply subtab style
        threat_notebook.configure(style='Subtab.TNotebook')
        
        threat_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create System Intelligence subtab (existing content)
        self.create_system_intelligence_tab(threat_notebook)
        
        # Create AI Threat Detection subtab (new facial recognition)
        self.create_ai_threat_detection_tab(threat_notebook)
        
        # Initialize failed login alerts for current user
        self.show_failed_login_alerts_for_user(self.current_user)
    
    def create_system_intelligence_tab(self, parent_notebook):
        """Create System Intelligence subtab with existing threat intelligence content"""
        system_tab = ttk.Frame(parent_notebook)
        parent_notebook.add(system_tab, text="📊 System Intelligence")
        
        # Create notebook for sections
        monitor_notebook = ttk.Notebook(system_tab)
        
        # Configure monitor notebook styling
        style = ttk.Style()
        style.configure('Monitor.TNotebook.Tab', 
                       font=('Segoe UI', 10, 'bold'),
                       padding=[12, 6],
                       foreground='black')  # Default text color
        style.map('Monitor.TNotebook.Tab',
                 background=[('selected', self.colors['warning']),
                           ('active', self.colors['border']),
                           ('!selected', 'SystemButtonFace')],
                 foreground=[('selected', 'black'),
                           ('active', 'black'),
                           ('!selected', 'black')])
        
        # Apply monitor style
        monitor_notebook.configure(style='Monitor.TNotebook')
        
        monitor_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Real-time Alerts Section
        alerts_frame = ttk.Frame(monitor_notebook)
        monitor_notebook.add(alerts_frame, text="Real-time Alerts")
        
        # Split into two columns
        alert_left = ttk.Frame(alerts_frame)
        alert_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        alert_right = ttk.Frame(alerts_frame)
        alert_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        
        # Alert display card
        alert_card = tk.Frame(alert_left, bg=self.colors['card_bg'],
                             highlightthickness=1, highlightbackground=self.colors['border'])
        alert_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Card title
        alert_title = tk.Frame(alert_card, bg=self.colors['error'], height=40)
        alert_title.pack(fill='x')
        
        tk.Label(
            alert_title,
            text="Active Alerts",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['error'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Alert content
        alert_content = tk.Frame(alert_card, bg=self.colors['card_bg'], padx=15, pady=15)
        alert_content.pack(fill=tk.BOTH, expand=True)
        
        # Alerts text
        self.alert_text = tk.Text(alert_content, height=20, bg='#fff8f8')  # Light red background
        alert_scrollbar = ttk.Scrollbar(alert_content, orient=tk.VERTICAL, command=self.alert_text.yview)
        self.alert_text.configure(yscrollcommand=alert_scrollbar.set)
        
        self.alert_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alert_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add initial alert
        self.alert_text.insert(tk.END, "No active threats detected\n")
        
        # Alert actions card
        action_card = tk.Frame(alert_right, bg=self.colors['card_bg'],
                              highlightthickness=1, highlightbackground=self.colors['border'])
        action_card.pack(fill='x', padx=5, pady=5)
        
        # Card title
        action_title = tk.Frame(action_card, bg=self.colors['accent'], height=40)
        action_title.pack(fill='x')
        
        tk.Label(
            action_title,
            text="Alert Actions",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Action content
        action_content = tk.Frame(action_card, bg=self.colors['card_bg'], padx=15, pady=15)
        action_content.pack(fill='x')
        
        # Action buttons
        clear_btn = tk.Button(
            action_content,
            text="Clear Alerts",
            font=('Segoe UI', 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.clear_alerts
        )
        clear_btn.pack(fill='x', pady=5)
        
        export_btn = tk.Button(
            action_content,
            text="Export Alerts",
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.export_alerts
        )
        export_btn.pack(fill='x', pady=5)
        
        # Alert statistics card
        stats_card = tk.Frame(alert_right, bg=self.colors['card_bg'],
                             highlightthickness=1, highlightbackground=self.colors['border'])
        stats_card.pack(fill='x', padx=5, pady=5)
        
        # Card title
        stats_title = tk.Frame(stats_card, bg=self.colors['accent'], height=40)
        stats_title.pack(fill='x')
        
        tk.Label(
            stats_title,
            text="Alert Statistics",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Stats content
        stats_content = tk.Frame(stats_card, bg=self.colors['card_bg'], padx=15, pady=15)
        stats_content.pack(fill='x')
        
        # Stats grid
        stats_grid = tk.Frame(stats_content, bg=self.colors['card_bg'])
        stats_grid.pack(fill='x')
        
        # Stat counters
        self.error_count = tk.StringVar(value="0")
        self.warning_count = tk.StringVar(value="0")
        self.info_count = tk.StringVar(value="0")
        
        # Error stat
        tk.Label(stats_grid, text="Errors:", bg=self.colors['card_bg'],
                fg=self.colors['text_primary']).grid(row=0, column=0, sticky='w', pady=5)
        tk.Label(stats_grid, textvariable=self.error_count, bg=self.colors['card_bg'],
                fg=self.colors['error'], font=('Segoe UI', 12, 'bold')).grid(row=0, column=1, sticky='w', pady=5, padx=10)
        
        # Warning stat
        tk.Label(stats_grid, text="Warnings:", bg=self.colors['card_bg'],
                fg=self.colors['text_primary']).grid(row=1, column=0, sticky='w', pady=5)
        tk.Label(stats_grid, textvariable=self.warning_count, bg=self.colors['card_bg'],
                fg=self.colors['warning'], font=('Segoe UI', 12, 'bold')).grid(row=1, column=1, sticky='w', pady=5, padx=10)
        
        # Info stat
        tk.Label(stats_grid, text="Info:", bg=self.colors['card_bg'],
                fg=self.colors['text_primary']).grid(row=2, column=0, sticky='w', pady=5)
        tk.Label(stats_grid, textvariable=self.info_count, bg=self.colors['card_bg'],
                fg=self.colors['info'], font=('Segoe UI', 12, 'bold')).grid(row=2, column=1, sticky='w', pady=5, padx=10)
        
        # Threat Detection Section
        threat_frame = ttk.Frame(monitor_notebook)
        monitor_notebook.add(threat_frame, text="Threat Detection")
        
        # System status card
        status_card = tk.Frame(threat_frame, bg=self.colors['card_bg'],
                              highlightthickness=1, highlightbackground=self.colors['border'])
        status_card.pack(fill='x', padx=5, pady=5)
        
        # Card title
        status_title = tk.Frame(status_card, bg=self.colors['accent'], height=40)
        status_title.pack(fill='x')
        
        tk.Label(
            status_title,
            text="System Status",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Status content
        status_content = tk.Frame(status_card, bg=self.colors['card_bg'], padx=15, pady=15)
        status_content.pack(fill='x')
        
        # Status grid
        status_grid = tk.Frame(status_content, bg=self.colors['card_bg'])
        status_grid.pack(fill='x')
        
        # Network security status
        tk.Label(status_grid, text="Network Security:", bg=self.colors['card_bg'],
                fg=self.colors['text_primary'], font=('Segoe UI', 11)).grid(row=0, column=0, sticky='w', pady=10)
        self.network_status = tk.Label(status_grid, text="✅ Secure", bg=self.colors['card_bg'],
                                      fg=self.colors['success'], font=('Segoe UI', 11, 'bold'))
        self.network_status.grid(row=0, column=1, sticky='w', pady=10, padx=10)
        
        # Video integrity status
        tk.Label(status_grid, text="Video Integrity:", bg=self.colors['card_bg'],
                fg=self.colors['text_primary'], font=('Segoe UI', 11)).grid(row=1, column=0, sticky='w', pady=10)
        self.video_status = tk.Label(status_grid, text="✅ Verified", bg=self.colors['card_bg'],
                                    fg=self.colors['success'], font=('Segoe UI', 11, 'bold'))
        self.video_status.grid(row=1, column=1, sticky='w', pady=10, padx=10)
        
        # System health status
        tk.Label(status_grid, text="System Health:", bg=self.colors['card_bg'],
                fg=self.colors['text_primary'], font=('Segoe UI', 11)).grid(row=2, column=0, sticky='w', pady=10)
        self.health_status = tk.Label(status_grid, text="✅ Normal", bg=self.colors['card_bg'],
                                     fg=self.colors['success'], font=('Segoe UI', 11, 'bold'))
        self.health_status.grid(row=2, column=1, sticky='w', pady=10, padx=10)
        
        # System metrics card
        metrics_card = tk.Frame(threat_frame, bg=self.colors['card_bg'],
                               highlightthickness=1, highlightbackground=self.colors['border'])
        metrics_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Card title
        metrics_title = tk.Frame(metrics_card, bg=self.colors['accent'], height=40)
        metrics_title.pack(fill='x')
        
        tk.Label(
            metrics_title,
            text="System Metrics",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Metrics content
        metrics_content = tk.Frame(metrics_card, bg=self.colors['card_bg'], padx=15, pady=15)
        metrics_content.pack(fill=tk.BOTH, expand=True)
        
        # Add gauges for CPU, memory, and disk usage
        gauge_frame = tk.Frame(metrics_content, bg=self.colors['card_bg'])
        gauge_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # CPU gauge
        cpu_frame = tk.Frame(gauge_frame, bg=self.colors['card_bg'])
        cpu_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(cpu_frame, text="CPU Usage", bg=self.colors['card_bg'],
                fg=self.colors['text_primary'], font=('Segoe UI', 11)).pack(anchor='center')
        
        cpu_canvas = tk.Canvas(cpu_frame, width=120, height=120, bg=self.colors['card_bg'],
                              highlightthickness=0)
        cpu_canvas.pack(anchor='center', pady=10)
        
        # Draw gauge
        cpu_canvas.create_arc(10, 10, 110, 110, start=0, extent=180, fill='', outline=self.colors['bg_secondary'],
                             width=15, style='arc')
        cpu_canvas.create_arc(10, 10, 110, 110, start=0, extent=45, fill='', outline=self.colors['accent'],
                             width=15, style='arc')
        
        # Add value text
        cpu_value = tk.Label(cpu_frame, text="25%", bg=self.colors['card_bg'],
                            fg=self.colors['accent'], font=('Segoe UI', 14, 'bold'))
        cpu_value.pack(anchor='center')
        
        # Memory gauge
        mem_frame = tk.Frame(gauge_frame, bg=self.colors['card_bg'])
        mem_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(mem_frame, text="Memory Usage", bg=self.colors['card_bg'],
                fg=self.colors['text_primary'], font=('Segoe UI', 11)).pack(anchor='center')
        
        mem_canvas = tk.Canvas(mem_frame, width=120, height=120, bg=self.colors['card_bg'],
                              highlightthickness=0)
        mem_canvas.pack(anchor='center', pady=10)
        
        # Draw gauge
        mem_canvas.create_arc(10, 10, 110, 110, start=0, extent=180, fill='', outline=self.colors['bg_secondary'],
                             width=15, style='arc')
        mem_canvas.create_arc(10, 10, 110, 110, start=0, extent=90, fill='', outline=self.colors['warning'],
                             width=15, style='arc')
        
        # Add value text
        mem_value = tk.Label(mem_frame, text="50%", bg=self.colors['card_bg'],
                            fg=self.colors['warning'], font=('Segoe UI', 14, 'bold'))
        mem_value.pack(anchor='center')
        
        # Disk gauge
        disk_frame = tk.Frame(gauge_frame, bg=self.colors['card_bg'])
        disk_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(disk_frame, text="Disk Usage", bg=self.colors['card_bg'],
                fg=self.colors['text_primary'], font=('Segoe UI', 11)).pack(anchor='center')
        
        disk_canvas = tk.Canvas(disk_frame, width=120, height=120, bg=self.colors['card_bg'],
                               highlightthickness=0)
        disk_canvas.pack(anchor='center', pady=10)
        
        # Draw gauge
        disk_canvas.create_arc(10, 10, 110, 110, start=0, extent=180, fill='', outline=self.colors['bg_secondary'],
                              width=15, style='arc')
        disk_canvas.create_arc(10, 10, 110, 110, start=0, extent=135, fill='', outline=self.colors['error'],
                              width=15, style='arc')
        
        # Add value text
        disk_value = tk.Label(disk_frame, text="75%", bg=self.colors['card_bg'],
                             fg=self.colors['error'], font=('Segoe UI', 14, 'bold'))
        disk_value.pack(anchor='center')
        
        self.update_vpn_status()
    
    def create_ai_threat_detection_tab(self, parent_notebook):
        """Create AI Threat Detection subtab with facial recognition interface"""
        ai_tab = ttk.Frame(parent_notebook)
        parent_notebook.add(ai_tab, text="🤖 AI Threat Detection")
        
        # Main container
        main_container = tk.Frame(ai_tab, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for viewers
        left_panel = tk.Frame(main_container, bg=self.colors['bg_primary'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right panel for controls with scrollbar
        right_panel_container = tk.Frame(main_container, bg=self.colors['bg_primary'], width=320)
        right_panel_container.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel_container.pack_propagate(False)
        
        # Create scrollable frame using canvas
        canvas = tk.Canvas(right_panel_container, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_panel_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configure canvas to expand content to full width
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas.find_all()[0], width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Store references for later use
        self.face_recognition_canvas = canvas
        self.face_recognition_scrollable_frame = scrollable_frame
        
        # Create dual viewer system
        self.create_face_viewer_system(left_panel)
        
        # Create facial recognition control panel (now in scrollable frame)
        self.create_facial_recognition_controls(scrollable_frame)
    
    def create_face_viewer_system(self, parent):
        """Create dual viewer system for face upload and CCTV footage"""
        # Title
        title_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        title_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            title_frame,
            text="🎯 Facial Recognition Surveillance",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        # Viewer container
        viewer_container = tk.Frame(parent, bg=self.colors['bg_primary'])
        viewer_container.pack(fill=tk.BOTH, expand=True)
        
        # Left viewer - Face Upload
        left_viewer_frame = tk.Frame(viewer_container, bg=self.colors['card_bg'],
                                   highlightthickness=2, highlightbackground=self.colors['border'])
        left_viewer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Left viewer header
        left_header = tk.Frame(left_viewer_frame, bg=self.colors['accent'], height=40)
        left_header.pack(fill='x')
        left_header.pack_propagate(False)
        
        tk.Label(
            left_header,
            text="👤 Target Face",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15
        ).pack(side='left', pady=8)
        
        # Left viewer content
        self.face_viewer_frame = tk.Frame(left_viewer_frame, bg=self.colors['card_bg'])
        self.face_viewer_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Face upload area
        self.face_upload_area = tk.Frame(self.face_viewer_frame, bg='#2d3748',
                                       highlightthickness=2, highlightbackground=self.colors['border'],
                                       relief='raised')
        self.face_upload_area.pack(fill=tk.BOTH, expand=True)
        
        # Upload instruction
        upload_instruction = tk.Label(
            self.face_upload_area,
            text="📷\n\nDrop target face image here\nor click to browse",
            font=('Segoe UI', 12),
            bg='#2d3748',
            fg=self.colors['text_secondary'],
            justify='center'
        )
        upload_instruction.pack(expand=True)
        
        # Make upload area clickable
        self.face_upload_area.bind("<Button-1>", lambda e: self.upload_target_face())
        upload_instruction.bind("<Button-1>", lambda e: self.upload_target_face())
        
        # Right viewer - CCTV Footage
        right_viewer_frame = tk.Frame(viewer_container, bg=self.colors['card_bg'],
                                    highlightthickness=2, highlightbackground=self.colors['border'])
        right_viewer_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Right viewer header
        right_header = tk.Frame(right_viewer_frame, bg=self.colors['accent'], height=40)
        right_header.pack(fill='x')
        right_header.pack_propagate(False)
        
        tk.Label(
            right_header,
            text="🎥 CCTV Analysis",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15
        ).pack(side='left', pady=8)
        
        # Right viewer content
        self.cctv_viewer_frame = tk.Frame(right_viewer_frame, bg=self.colors['card_bg'])
        self.cctv_viewer_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # CCTV analysis area
        self.cctv_analysis_area = tk.Frame(self.cctv_viewer_frame, bg='#1a202c',
                                         highlightthickness=2, highlightbackground=self.colors['border'],
                                         relief='sunken')
        self.cctv_analysis_area.pack(fill=tk.BOTH, expand=True)
        
        # Analysis instruction
        analysis_instruction = tk.Label(
            self.cctv_analysis_area,
            text="🔍\n\nSelect CCTV footage for\nfacial recognition analysis",
            font=('Segoe UI', 12),
            bg='#1a202c',
            fg=self.colors['text_secondary'],
            justify='center'
        )
        analysis_instruction.pack(expand=True)
    
    def create_facial_recognition_controls(self, parent):
        """Create control panel for facial recognition settings and results"""
        # Title
        title_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        title_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            title_frame,
            text="🎛️ Recognition Controls",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        # Target Information Card
        target_card = tk.Frame(parent, bg=self.colors['card_bg'],
                             highlightthickness=1, highlightbackground=self.colors['border'])
        target_card.pack(fill='x', pady=(0, 10))
        
        # Card header
        target_header = tk.Frame(target_card, bg=self.colors['accent'])
        target_header.pack(fill='x')
        
        tk.Label(
            target_header,
            text="🎯 Target Info",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        target_content = tk.Frame(target_card, bg=self.colors['card_bg'])
        target_content.pack(fill='x', padx=12, pady=10)
        
        # Target status
        self.target_status_label = tk.Label(
            target_content,
            text="❌ No target loaded",
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['error']
        )
        self.target_status_label.pack(anchor='w', pady=(0, 5))
        
        # Upload button
        upload_btn = tk.Button(
            target_content,
            text="📁 Upload Target Face",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.upload_target_face
        )
        upload_btn.pack(fill='x', pady=(0, 5))
        
        # Clear button
        clear_btn = tk.Button(
            target_content,
            text="🗑️ Clear Target",
            font=('Segoe UI', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.clear_target_face
        )
        clear_btn.pack(fill='x')
        
        # Detection Settings Card
        settings_card = tk.Frame(parent, bg=self.colors['card_bg'],
                               highlightthickness=1, highlightbackground=self.colors['border'])
        settings_card.pack(fill='x', pady=(0, 10))
        
        # Card header
        settings_header = tk.Frame(settings_card, bg=self.colors['accent'], height=35)
        settings_header.pack(fill='x')
        settings_header.pack_propagate(False)
        
        tk.Label(
            settings_header,
            text="⚙️ Detection Settings",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        settings_content = tk.Frame(settings_card, bg=self.colors['card_bg'])
        settings_content.pack(fill='x', padx=12, pady=10)
        
        # Confidence threshold
        tk.Label(
            settings_content,
            text="Confidence Threshold:",
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        confidence_frame = tk.Frame(settings_content, bg=self.colors['card_bg'])
        confidence_frame.pack(fill='x', pady=(2, 8))
        
        self.confidence_var = tk.DoubleVar(value=0.8)
        confidence_scale = tk.Scale(
            confidence_frame,
            from_=0.5,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_secondary']
        )
        confidence_scale.pack(side='left', fill='x', expand=True)
        
        confidence_label = tk.Label(
            confidence_frame,
            textvariable=self.confidence_var,
            font=('Segoe UI', 9),
            bg=self.colors['card_bg'],
            fg=self.colors['accent'],
            width=5
        )
        confidence_label.pack(side='right', padx=(5, 0))
        
        # Real-time detection toggle
        self.realtime_var = tk.BooleanVar(value=False)
        realtime_check = tk.Checkbutton(
            settings_content,
            text="Real-time Detection",
            variable=self.realtime_var,
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent'],
            activebackground=self.colors['card_bg'],
            activeforeground=self.colors['text_primary']
        )
        realtime_check.pack(anchor='w', pady=(0, 5))
        
        # Alert notifications toggle
        self.alerts_var = tk.BooleanVar(value=True)
        alerts_check = tk.Checkbutton(
            settings_content,
            text="Alert Notifications",
            variable=self.alerts_var,
            font=('Segoe UI', 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['accent'],
            activebackground=self.colors['card_bg'],
            activeforeground=self.colors['text_primary']
        )
        alerts_check.pack(anchor='w')
        
        # Analysis Controls Card
        controls_card = tk.Frame(parent, bg=self.colors['card_bg'],
                               highlightthickness=1, highlightbackground=self.colors['border'])
        controls_card.pack(fill='x', pady=(0, 10))
        
        # Card header
        controls_header = tk.Frame(controls_card, bg=self.colors['accent'])
        controls_header.pack(fill='x')
        
        tk.Label(
            controls_header,
            text="🎬 Analysis Controls",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        controls_content = tk.Frame(controls_card, bg=self.colors['card_bg'])
        controls_content.pack(fill='x', padx=12, pady=10)
        
        # Select footage button
        select_footage_btn = tk.Button(
            controls_content,
            text="🎥 Select CCTV Footage",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.select_cctv_footage
        )
        select_footage_btn.pack(fill='x', pady=(0, 5))
        
        # Start analysis button
        self.start_analysis_btn = tk.Button(
            controls_content,
            text="🔍 Start Analysis",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['success'],
            fg='white',
            activebackground='#1e7e34',
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            state='disabled',
            command=self.start_facial_analysis
        )
        self.start_analysis_btn.pack(fill='x', pady=(0, 5))
        
        # Stop analysis button
        self.stop_analysis_btn = tk.Button(
            controls_content,
            text="⏹️ Stop Analysis",
            font=('Segoe UI', 10),
            bg=self.colors['error'],
            fg='white',
            activebackground='#c82333',
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            state='disabled',
            command=self.stop_facial_analysis
        )
        self.stop_analysis_btn.pack(fill='x')
        
        # Results Card
        results_card = tk.Frame(parent, bg=self.colors['card_bg'],
                              highlightthickness=1, highlightbackground=self.colors['border'])
        results_card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        results_header = tk.Frame(results_card, bg=self.colors['accent'])
        results_header.pack(fill='x')
        
        tk.Label(
            results_header,
            text="📊 Detection Results",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=12
        ).pack(side='left', pady=6)
        
        # Card content
        results_content = tk.Frame(results_card, bg=self.colors['card_bg'])
        results_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        # Results text area
        self.results_text = tk.Text(
            results_content,
            height=8,
            font=('Consolas', 9),
            bg='#1a202c',
            fg=self.colors['text_secondary'],
            insertbackground=self.colors['text_primary'],
            relief='flat',
            wrap=tk.WORD
        )
        
        results_scrollbar = ttk.Scrollbar(results_content, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial message
        self.results_text.insert(tk.END, "🔍 Ready for facial recognition analysis...\n")
        self.results_text.insert(tk.END, "📝 Upload a target face to begin\n")
        self.results_text.config(state='disabled')
    
    # Placeholder methods for facial recognition functionality
    def upload_target_face(self):
        """Upload target face for recognition"""
        file_path = filedialog.askopenfilename(
            title="Select Target Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.target_status_label.config(text="✅ Target face loaded", fg=self.colors['success'])
            self.start_analysis_btn.config(state='normal')
            # Add results log
            self.results_text.config(state='normal')
            self.results_text.insert(tk.END, f"📁 Target face loaded: {file_path.split('/')[-1]}\n")
            self.results_text.see(tk.END)
            self.results_text.config(state='disabled')
    
    def clear_target_face(self):
        """Clear loaded target face"""
        self.target_status_label.config(text="❌ No target loaded", fg=self.colors['error'])
        self.start_analysis_btn.config(state='disabled')
        self.results_text.config(state='normal')
        self.results_text.insert(tk.END, "🗑️ Target face cleared\n")
        self.results_text.see(tk.END)
        self.results_text.config(state='disabled')
    
    def select_cctv_footage(self):
        """Select CCTV footage for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select CCTV Footage",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.results_text.config(state='normal')
            self.results_text.insert(tk.END, f"🎥 CCTV footage selected: {file_path.split('/')[-1]}\n")
            self.results_text.see(tk.END)
            self.results_text.config(state='disabled')
    
    def start_facial_analysis(self):
        """Start facial recognition analysis"""
        self.start_analysis_btn.config(state='disabled')
        self.stop_analysis_btn.config(state='normal')
        self.results_text.config(state='normal')
        self.results_text.insert(tk.END, "🔍 Starting facial recognition analysis...\n")
        self.results_text.insert(tk.END, f"⚙️ Confidence threshold: {self.confidence_var.get()}\n")
        self.results_text.insert(tk.END, f"📡 Real-time detection: {'Enabled' if self.realtime_var.get() else 'Disabled'}\n")
        self.results_text.see(tk.END)
        self.results_text.config(state='disabled')
        
        # Update canvas scroll region
        self.update_face_recognition_scroll()
    
    def stop_facial_analysis(self):
        """Stop facial recognition analysis"""
        self.start_analysis_btn.config(state='normal')
        self.stop_analysis_btn.config(state='disabled')
        self.results_text.config(state='normal')
        self.results_text.insert(tk.END, "⏹️ Analysis stopped\n")
        self.results_text.see(tk.END)
        self.results_text.config(state='disabled')
        
        # Update canvas scroll region
        self.update_face_recognition_scroll()
    
    def update_face_recognition_scroll(self):
        """Update the scroll region for the face recognition panel"""
        if hasattr(self, 'face_recognition_canvas'):
            self.face_recognition_canvas.update_idletasks()
            self.face_recognition_canvas.configure(scrollregion=self.face_recognition_canvas.bbox("all"))
    
    def show_failed_login_alerts_for_user(self, username):
        """Show only the last 20 failed login attempts (IP only, WARNING only) as alerts in Threat Intelligence tab, with delay to avoid GUI freeze."""
        import os
        log_file = 'cctv_system.log'
        if not os.path.exists(log_file):
            return
        try:
            # Read only the last 1000 lines for performance
            with open(log_file, 'rb') as f:
                try:
                    f.seek(-1024 * 100, os.SEEK_END)  # Read last ~100KB
                except OSError:
                    f.seek(0)
                lines = f.read().decode(errors='ignore').splitlines()[-1000:]
            import re
            failed_alerts = []
            for line in lines:
                # Only match WARNING lines with failed login attempts for this user
                if "Failed login attempt" in line and "WARNING" in line:
                    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d* - WARNING - Failed login attempt for [^\s]+ from IP ([^\s]+)", line)
                    if match:
                        timestamp, ip = match.groups()
                        msg = f"[{timestamp}] Failed login attempt from IP {ip}"
                    else:
                        msg = line.strip()
                    failed_alerts.append(msg)
            # Only show the last 20 alerts
            failed_alerts = failed_alerts[-20:]

            def add_next_alert(i=0):
                if i < len(failed_alerts):
                    if hasattr(self, 'add_alert'):
                        self.add_alert(failed_alerts[i], severity="Warning")
                    self.root.after(50, add_next_alert, i+1)  # 50ms delay between alerts

            self.root.after(0, add_next_alert)
        except Exception as e:
            self.add_log(f"Error reading failed login attempts: {str(e)}")
    def clear_alerts(self):
        """Clear alerts"""
        self.alert_text.delete(1.0, tk.END)
        self.alert_text.insert(tk.END, "Alerts cleared\n")
        logging.info("Alerts cleared")
    
    def export_alerts(self):
        """Export alerts to file"""
        alerts = self.alert_text.get(1.0, tk.END)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(alerts)
            messagebox.showinfo("Export Successful", "Alerts exported successfully")
            logging.info("Exported alerts")
    
    def create_logs_tab(self):
        """Create system logs tab"""
        logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(logs_tab, text="📋 System Logs")
        
        # Logs display
        logs_frame = tk.Frame(logs_tab, bg=self.colors['card_bg'],
                             highlightthickness=1, highlightbackground=self.colors['border'])
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Logs title
        logs_title = tk.Frame(logs_frame, bg=self.colors['accent'], height=40)
        logs_title.pack(fill='x')
        
        tk.Label(
            logs_title,
            text="System Logs",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left')
        
        # Search box
        search_frame = tk.Frame(logs_title, bg=self.colors['accent'])
        search_frame.pack(side='right', padx=10)
        
        self.log_search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.log_search_var, width=20)
        search_entry.pack(side='left')
        
        search_btn = tk.Button(
            search_frame,
            text="🔍",
            font=('Segoe UI', 10),
            bg=self.colors['bg_accent'],
            fg='white',
            activebackground=self.colors['bg_accent'],
            activeforeground='white',
            bd=0,
            padx=5,
            pady=0,
            cursor="hand2",
            command=lambda: self.filter_logs(self.log_search_var.get())
        )
        search_btn.pack(side='right')
        
        # Logs content
        logs_content = tk.Frame(logs_frame, bg=self.colors['card_bg'], padx=10, pady=10)
        logs_content.pack(fill=tk.BOTH, expand=True)
        
        # Logs text
        self.logs_text = tk.Text(logs_content, bg='#f8fafc')
        logs_scrollbar = ttk.Scrollbar(logs_content, orient=tk.VERTICAL, command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=logs_scrollbar.set)
        
        self.logs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        logs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load logs
        self.refresh_logs()
        
        # Log controls
        log_controls = tk.Frame(logs_tab)
        log_controls.pack(fill='x', padx=10, pady=5)
        
        clear_btn = tk.Button(
            log_controls,
            text="Clear Logs",
            font=('Segoe UI', 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.clear_logs
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = tk.Button(
            log_controls,
            text="Export Logs",
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
                       fg='white',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.export_logs
        )
        export_btn.pack(side=tk.LEFT, padx=5)
        
        refresh_btn = tk.Button(
            log_controls,
            text="Refresh",
            font=('Segoe UI', 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['menu_hover'],
            activeforeground=self.colors['text_primary'],
            bd=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.refresh_logs
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)
    
    def refresh_logs(self):
        """Refresh logs display"""
        self.logs_text.delete(1.0, tk.END)
        
        # Read log file
        log_file = 'cctv_system.log'
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = f.readlines()
                for log in logs:
                    self.logs_text.insert(tk.END, log)
        else:
            self.logs_text.insert(tk.END, "No log file found")
    
    def filter_logs(self, search_term):
        """Filter logs based on search term"""
        self.logs_text.delete(1.0, tk.END)
        
        # Read log file
        log_file = 'cctv_system.log'
        if not os.path.exists(log_file):
            self.logs_text.insert(tk.END, "No log file found")
            return
        
        search_term = search_term.lower()
        filtered_logs = []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.readlines()
            for log in logs:
                if search_term in log.lower():
                    filtered_logs.append(log)
        
        if filtered_logs:
            for log in filtered_logs:
                self.logs_text.insert(tk.END, log)
        else:
            self.logs_text.insert(tk.END, "No matching logs found")
    
    def clear_logs(self):
        """Clear logs display"""
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.insert(tk.END, "Logs cleared\n")
        logging.info("Logs cleared")
    
    def export_logs(self):
        """Export logs to file"""
        logs = self.logs_text.get(1.0, tk.END)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(logs)
            messagebox.showinfo("Export Successful", "Logs exported successfully")
            logging.info("Exported logs")
    
    def start_monitoring(self):
        """Start monitoring thread"""
        self.monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def monitor_system(self):
        """Monitor system for alerts and VPN status."""
        while True:
            try:
                # Simulate random alerts
                if random.random() < 0.1:  # 10% chance of alert
                    alerts = [
                        "Unusual network activity detected",
                        "Video feed interruption",
                        "High CPU usage detected",
                        "Storage capacity warning",
                        "Unauthorized access attempt"
                    ]
                    alert = random.choice(alerts)
                    
                    severities = ["Low", "Medium", "High"]
                    severity = random.choice(severities)
                    
                    # Update alert text
                    self.alert_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] [{severity}] {alert}\n")
                    self.alert_text.see(tk.END)
                    
                    # Update stats
                    if severity == "High":
                        self.error_count.set(str(int(self.error_count.get()) + 1))
                    elif severity == "Medium":
                        self.warning_count.set(str(int(self.warning_count.get()) + 1))
                    else:
                        self.info_count.set(str(int(self.info_count.get()) + 1))
                
                # Update monitoring text
                self.monitor_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] System check completed\n")
                self.monitor_text.see(tk.END)

                # --- VPN STATUS CHECK ---
                self.root.after(0, self.update_vpn_status)
                # -----------------------

                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Monitoring error: {str(e)}")
                time.sleep(30)
    
    def logout(self):
        """Logout current user"""
        logging.info(f"User {self.current_user} logged out")
        
        # Stop all video players
        if hasattr(self, 'video_players'):
            for i, player in enumerate(self.video_players):
                if player:
                    player.stop()
                    self.video_players[i] = None
        
        self.root.destroy()
        # Restart application
        root = tk.Tk()
        app = CCTVApp(root)
        root.mainloop()

# Main entry point
if __name__ == "__main__":
    # Start application
    root = tk.Tk()
    app = CCTVApp(root)
    root.mainloop()