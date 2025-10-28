import sqlite3
import os
import ssl
import logging
from hashlib import sha256
from cryptography.fernet import Fernet
import hmac
from datetime import datetime
import ssl
import logging

DB_PATH = "secure_cctv.db"

def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                totp_secret TEXT,
                backup_codes TEXT
            )
        ''')        
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_certificates (
                username TEXT PRIMARY KEY,
                certificate TEXT NOT NULL,
                private_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_valid INTEGER DEFAULT 1
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                size INTEGER,
                duration REAL,
                width INTEGER,
                height INTEGER,
                fps REAL,
                upload_time TEXT NOT NULL,
                encrypted INTEGER DEFAULT 1,
                video_data BLOB
            )
        ''')
        conn.commit()
        conn.close()
        os.chmod(DB_PATH, 0o600)
    else:
        # Check if backup_codes column exists, add it if not
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("SELECT backup_codes FROM users LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            c.execute("ALTER TABLE users ADD COLUMN backup_codes TEXT")
            conn.commit()
        
        try:
            c.execute("SELECT totp_secret FROM users LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            c.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
            conn.commit()
        
        conn.close()

def generate_encryption_key():
    """Generate and save a new encryption key"""
    key = Fernet.generate_key()
    with open('encryption.key', 'wb') as f:
        f.write(key)
    # Set secure permissions on the key file
    os.chmod('encryption.key', 0o600)
    return key

def load_encryption_key():
    """Load encryption key, generate if it doesn't exist"""
    if not os.path.exists('encryption.key'):
        return generate_encryption_key()
    with open('encryption.key', 'rb') as f:
        return f.read()

def encrypt_data(data: str) -> str:
    key = load_encryption_key()
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(token: str) -> str:
    key = load_encryption_key()
    f = Fernet(key)
    return f.decrypt(token.encode()).decode()

def hash_password(password: str) -> str:
    # Add a salt for even more security in production!
    return sha256(password.encode('utf-8')).hexdigest()

def add_user(username: str, password: str, role: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    password_hash = hash_password(password)
    encrypted_hash = encrypt_data(password_hash)
    c.execute('''
        INSERT INTO users (username, password_hash, role)
        VALUES (?, ?, ?)
    ''', (username, encrypted_hash, role))
    conn.commit()
    conn.close()

def verify_user(username, password):
    import sqlite3
    conn = sqlite3.connect("secure_cctv.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if row:
        # You may need to check password hash here!
        columns = [desc[0] for desc in c.description]
        user_dict = dict(zip(columns, row))
        # Check password hash here if needed
        conn.close()
        return user_dict
    conn.close()
    return None

def encrypt_file_bytes(file_path: str) -> bytes:
    key = load_encryption_key()
    f = Fernet(key)
    with open(file_path, 'rb') as file:
        data = file.read()
    return f.encrypt(data)

def encrypt_file_in_chunks(file_path: str, chunk_size: int = 1024 * 1024) -> bytes:
    """Encrypt a file in chunks and return the encrypted data."""
    key = load_encryption_key()
    f = Fernet(key)
    encrypted_chunks = []
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            encrypted_chunks.append(f.encrypt(chunk))
    return b"".join(encrypted_chunks)

def get_user_role(username: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def add_video(username, video_info):
    """Add video to database (no SSL chunk encryption)"""
    role = get_user_role(username)
    if role not in ("Admin", "Operator"):
        raise PermissionError("Only Admin or Operator can upload videos.")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        encrypted_path = encrypt_data(video_info.filepath)

        # Read video file as bytes (no SSL encryption)
        with open(video_info.filepath, 'rb') as f:
            video_data = f.read()

        # Store the video data as-is
        c.execute('''
            INSERT INTO videos (username, filename, filepath, size, duration, width, height, fps, upload_time, encrypted, video_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            username,
            video_info.filename,
            encrypted_path,
            video_info.size,
            video_info.duration,
            video_info.width,
            video_info.height,
            video_info.fps,
            datetime.now().isoformat(),
            0,  # Not SSL-encrypted
            video_data
        ))
        conn.commit()
        logging.info(f"Video {video_info.filename} uploaded successfully (no SSL encryption)")
    except Exception as e:
        logging.error(f"Error storing video: {str(e)}")
        raise
    finally:
        conn.close()

def get_user_videos(username, requesting_user=None):
    """Return videos uploaded by username. Admin can see all, operator their own, viewer only their own."""
    try:
        role = get_user_role(requesting_user or username)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        if role == "Admin":
            c.execute('SELECT id, filename, filepath, size, duration, width, height, fps, upload_time FROM videos')
        else:
            c.execute('SELECT id, filename, filepath, size, duration, width, height, fps, upload_time FROM videos WHERE username=?', (username,))
        rows = c.fetchall()
        conn.close()
        
        videos = []
        for row in rows:
            try:
                # Try to decrypt filepath
                decrypted_filepath = decrypt_data(row[2])
                videos.append({
                    'id': row[0],
                    'filename': row[1],
                    'filepath': decrypted_filepath,
                    'size': row[3],
                    'duration': row[4],
                    'width': row[5],
                    'height': row[6],
                    'fps': row[7],
                    'upload_time': row[8]
                })
            except Exception as decrypt_error:
                logging.warning(f"Failed to decrypt filepath for video {row[0]}: {decrypt_error}")
                # Try using the filepath as-is (might not be encrypted)
                try:
                    videos.append({
                        'id': row[0],
                        'filename': row[1],
                        'filepath': row[2],  # Use unencrypted filepath
                        'size': row[3],
                        'duration': row[4],
                        'width': row[5],
                        'height': row[6],
                        'fps': row[7],
                        'upload_time': row[8]
                    })
                except Exception as fallback_error:
                    logging.error(f"Failed to process video {row[0]}: {fallback_error}")
                    continue
        
        return videos
        
    except Exception as e:
        logging.error(f"Error in get_user_videos: {e}")
        return []

def get_video_data(video_id):
    """Get video data with SSL decryption"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('SELECT video_data FROM videos WHERE id=?', (video_id,))
        row = c.fetchone()
        if row and row[0]:
            # Set up SSL context for decryption
            cert_path = os.path.join('certificates', 'server.crt')
            key_path = os.path.join('certificates', 'server.key')
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(certfile=cert_path, keyfile=key_path)
            
            # Decrypt video data using SSL
            encrypted_data = row[0]
            bio = ssl.MemoryBIO()
            ssl_obj = context.wrap_bio(bio, server_side=True)
            bio.write(encrypted_data)
            decrypted_data = bio.read()
            
            return decrypted_data
        return None
    except Exception as e:
        logging.error(f"Error retrieving video with SSL decryption: {str(e)}")
        return None
    finally:
        conn.close()

def store_user_certificate(username: str, cert_data: str, key_data: str, expiry_date):
    """Store user certificate in database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Encrypt sensitive data before storing
        encrypted_cert = encrypt_data(cert_data)
        encrypted_key = encrypt_data(key_data)
        
        c.execute('''
            INSERT OR REPLACE INTO user_certificates 
            (username, certificate, private_key, expires_at, is_valid)
            VALUES (?, ?, ?, ?, 1)
        ''', (username, encrypted_cert, encrypted_key, expiry_date))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing certificate: {e}")
        return False
    finally:
        conn.close()

def get_user_certificate(username: str):
    """Get user certificate from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            SELECT certificate, private_key, expires_at, is_valid 
            FROM user_certificates 
            WHERE username = ?
        ''', (username,))
        result = c.fetchone()
        if result:
            cert_data, key_data, expires_at, is_valid = result
            return {
                'certificate': decrypt_data(cert_data),
                'private_key': decrypt_data(key_data),
                'expires_at': expires_at,
                'is_valid': bool(is_valid)
            }
        return None
    except Exception as e:
        print(f"Error retrieving certificate: {e}")
        return None
    finally:
        conn.close()

def invalidate_user_certificate(username: str):
    """Mark a user's certificate as invalid"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            UPDATE user_certificates 
            SET is_valid = 0 
            WHERE username = ?
        ''', (username,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error invalidating certificate: {e}")
        return False
    finally:
        conn.close()

def check_certificate_validity(username: str):
    """Check if a user's certificate is valid and not expired"""
    cert_data = get_user_certificate(username)
    if not cert_data:
        return False
    
    try:
        expiry = datetime.strptime(cert_data['expires_at'], '%Y-%m-%d %H:%M:%S')
        return cert_data['is_valid'] and datetime.now() < expiry
    except Exception as e:
        print(f"Error checking certificate validity: {e}")
        return False

def get_all_users():
    """Return a list of all usernames in the users table."""
    import sqlite3
    conn = sqlite3.connect("secure_cctv.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users
def update_user_totp(username, secret):
    conn = sqlite3.connect("secure_cctv.db")
    c = conn.cursor()
    c.execute("UPDATE users SET totp_secret=? WHERE username=?", (secret, username))
    conn.commit()
    conn.close()

def generate_backup_codes(count=3):
    """Generate backup codes for a user"""
    import secrets
    return [secrets.token_urlsafe(8) for _ in range(count)]

def set_user_backup_codes(username, backup_codes):
    """Set backup codes for a user (stored as encrypted JSON)"""
    import json
    conn = sqlite3.connect("secure_cctv.db")
    c = conn.cursor()
    
    # Convert list to JSON and encrypt
    codes_json = json.dumps(backup_codes)
    encrypted_codes = encrypt_data(codes_json)
    
    c.execute("UPDATE users SET backup_codes=? WHERE username=?", (encrypted_codes, username))
    conn.commit()
    conn.close()
    print(f"✅ Set {len(backup_codes)} backup codes for user: {username}")

def get_user_backup_codes(username):
    """Get backup codes for a user"""
    import json
    conn = sqlite3.connect("secure_cctv.db")
    c = conn.cursor()
    c.execute("SELECT backup_codes FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    
    if row and row[0]:
        try:
            # Decrypt and parse JSON
            decrypted_codes = decrypt_data(row[0])
            return json.loads(decrypted_codes)
        except Exception as e:
            print(f"Error decrypting backup codes: {e}")
            return []
    return []

def verify_backup_code(username, code):
    """Verify a backup code and remove it from the list if valid"""
    backup_codes = get_user_backup_codes(username)
    
    if code in backup_codes:
        # Remove the used code
        backup_codes.remove(code)
        set_user_backup_codes(username, backup_codes)
        print(f"✅ Backup code used for {username}. Remaining codes: {len(backup_codes)}")
        return True
    
    return False

def setup_backup_codes_for_all_users():
    """Set up 3 backup codes for all existing users"""
    users = get_all_users()
    all_codes = {}
    
    for username in users:
        # Check if user already has backup codes
        existing_codes = get_user_backup_codes(username)
        if not existing_codes:
            # Generate new backup codes
            new_codes = generate_backup_codes(3)
            set_user_backup_codes(username, new_codes)
            all_codes[username] = new_codes
            print(f"🔑 Generated backup codes for {username}: {new_codes}")
        else:
            all_codes[username] = existing_codes
            print(f"🔑 {username} already has {len(existing_codes)} backup codes")
    
    return all_codes
def get_user(username):
    """Return user record as a dict for the given username, or None if not found."""
    import sqlite3
    conn = sqlite3.connect("secure_cctv.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if row:
        # Adjust the keys to match your users table columns
        columns = [desc[0] for desc in c.description]
        user_dict = dict(zip(columns, row))
        conn.close()
        return user_dict
    conn.close()
    return None

def delete_videos(video_ids, requesting_user):
    """Delete videos by their database IDs. Only admins and operators can delete videos."""
    role = get_user_role(requesting_user)
    if role not in ("Admin", "Operator"):
        raise PermissionError("Only Admin or Operator can delete videos.")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    deleted_files = []
    errors = []
    
    try:
        for video_id in video_ids:
            # First get the video info to delete the physical file
            c.execute('SELECT filename, filepath, username FROM videos WHERE id = ?', (video_id,))
            video_row = c.fetchone()
            
            if not video_row:
                errors.append(f"Video with ID {video_id} not found in database")
                continue
                
            filename, encrypted_filepath, video_owner = video_row
            
            # Check permissions - operators can only delete their own videos
            if role == "Operator" and video_owner != requesting_user:
                errors.append(f"Operator can only delete their own videos: {filename}")
                continue
            
            try:
                # Decrypt the file path
                filepath = decrypt_data(encrypted_filepath)
                
                # Delete the physical file if it exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logging.info(f"Deleted file: {filepath}")
                else:
                    logging.warning(f"File not found on disk: {filepath}")
                
                # Delete from database
                c.execute('DELETE FROM videos WHERE id = ?', (video_id,))
                deleted_files.append(filename)
                logging.info(f"Deleted video from database: {filename} (ID: {video_id})")
                
            except Exception as e:
                error_msg = f"Error deleting {filename}: {str(e)}"
                errors.append(error_msg)
                logging.error(error_msg)
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        errors.append(f"Database error: {str(e)}")
        logging.error(f"Database error during video deletion: {str(e)}")
    finally:
        conn.close()
    
    return {
        'deleted': deleted_files,
        'errors': errors,
        'success_count': len(deleted_files),
        'error_count': len(errors)
    }

def get_video_id_by_filepath(filepath, username=None):
    """Get video database ID by filepath"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    encrypted_filepath = encrypt_data(filepath)
    
    if username:
        c.execute('SELECT id FROM videos WHERE filepath = ? AND username = ?', (encrypted_filepath, username))
    else:
        c.execute('SELECT id FROM videos WHERE filepath = ?', (encrypted_filepath,))
    
    row = c.fetchone()
    conn.close()
    
    return row[0] if row else None

if __name__ == "__main__":
    init_db()
    # Add users securely
    add_user('admin', 'Admin@123', 'Admin')
    add_user('operator1', 'Operator@123', 'Operator')
    add_user('viewer1', 'Viewer@123', 'Viewer')
    print("Database initialized and users added.")
    
    # Set up backup codes for all users
    print("\n🔐 Setting up backup codes...")
    backup_codes = setup_backup_codes_for_all_users()
    
    print("\n" + "="*60)
    print("🛡️  DEFENSE CCTV BACKUP CODES")
    print("="*60)
    for username, codes in backup_codes.items():
        print(f"\n👤 USER: {username.upper()}")
        print(f"   Role: {get_user_role(username)}")
        print("   Backup Codes:")
        for i, code in enumerate(codes, 1):
            print(f"   {i}. {code}")
    print("\n" + "="*60)
    print("⚠️  IMPORTANT: Save these codes securely!")
    print("⚠️  Each code can only be used once!")
    print("="*60)