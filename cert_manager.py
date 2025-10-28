import os
import ssl
import datetime
import OpenSSL
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import logging
from pathlib import Path
from local_db import (store_user_certificate, get_user_certificate, 
                     invalidate_user_certificate, check_certificate_validity)

class CertificateManager:
    def __init__(self, cert_dir='certificates'):
        self.cert_dir = cert_dir
        self.cert_path = os.path.join(cert_dir, 'server.crt')
        self.key_path = os.path.join(cert_dir, 'server.key')
        self.user_cert_dir = os.path.join(cert_dir, 'users')
        self.cert_valid = False
        self.cert_expiry = None
        
        # Create certificates directories if they don't exist
        for directory in [cert_dir, self.user_cert_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize logging
        logging.basicConfig(
            filename='cctv_system.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Check certificates on initialization
        self.check_certificates()
        
    def get_user_cert_paths(self, username):
        """Get certificate paths for a specific user"""
        user_dir = os.path.join(self.user_cert_dir, username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        return {
            'cert': os.path.join(user_dir, 'user.crt'),
            'key': os.path.join(user_dir, 'user.key')        }
        
    def generate_certificates(self, username=None):
        """Generate new SSL certificates"""
        try:
            # Determine certificate paths            
            if username:
                paths = self.get_user_cert_paths(username)
                cert_path = paths['cert']
                key_path = paths['key']
                common_name = f"CCTV User: {username}"
                # First invalidate any existing certificate
                invalidate_user_certificate(username)
            else:
                cert_path = self.cert_path
                key_path = self.key_path
                common_name = "CCTV Security System"

            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Generate public key
            public_key = private_key.public_key()
            
            # Create certificate subject and issuer
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Local Security"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, u"CCTV Unit")
            ])
            
            # Set certificate validity period (1 year)
            not_valid_before = datetime.datetime.utcnow()
            not_valid_after = not_valid_before + datetime.timedelta(days=365)
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                public_key
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                not_valid_before
            ).not_valid_after(
                not_valid_after
            ).add_extension(
                x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
                critical=False
            ).sign(private_key, hashes.SHA256())
            
            # Get PEM format for storage
            private_key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            certificate_pem = cert.public_bytes(
                serialization.Encoding.PEM
            ).decode('utf-8')
            
            # Save to filesystem
            with open(key_path, "w") as f:
                f.write(private_key_pem)
            
            with open(cert_path, "w") as f:
                f.write(certificate_pem)
            
            # If it's a user certificate, store in database
            if username:
                if not store_user_certificate(
                    username, 
                    certificate_pem, 
                    private_key_pem,
                    not_valid_after.strftime('%Y-%m-%d %H:%M:%S')
                ):
                    logging.error(f"Failed to store certificate in database for user {username}")
                    return False
            
            self.cert_valid = True
            self.cert_expiry = not_valid_after
            logging.info(f"New SSL certificates generated successfully for {'user ' + username if username else 'server'}")
            return True
            
        except Exception as e:
            logging.error(f"Error generating certificates: {str(e)}")
            self.cert_valid = False
            return False
    
    def check_certificates(self):
        """Check if certificates exist and are valid"""
        try:
            if not (os.path.exists(self.cert_path) and os.path.exists(self.key_path)):
                logging.warning("Base certificates not found. Generating new ones...")
                return self.generate_certificates()
            
            # Load and verify certificate
            with open(self.cert_path, 'rb') as f:
                cert_data = f.read()
                cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert_data)
                
                # Check expiry
                expiry = datetime.datetime.strptime(cert.get_notAfter().decode('ascii'), '%Y%m%d%H%M%SZ')
                self.cert_expiry = expiry
                
                if datetime.datetime.utcnow() > expiry:
                    logging.warning("Base certificate expired. Generating new one...")
                    return self.generate_certificates()
                
                # Check if certificate will expire soon (within 30 days)
                days_until_expiry = (expiry - datetime.datetime.utcnow()).days
                if days_until_expiry < 30:
                    logging.warning(f"Base certificate will expire in {days_until_expiry} days")
                
                self.cert_valid = True
                logging.info("Base certificates validated successfully")
                return True
                
        except Exception as e:
            logging.error(f"Error checking certificates: {str(e)}")
            self.cert_valid = False
            return False
            
    def get_ssl_context(self):
        """Get SSL context for secure communications"""
        if not self.cert_valid:
            self.check_certificates()
            
        if not self.cert_valid:
            logging.error("Cannot create SSL context - invalid certificates")
            return None
            
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path)
            return context
        except Exception as e:
            logging.error(f"Error creating SSL context: {str(e)}")
            return None
            
    def get_certificate_status(self):
        """Get current certificate status"""
        return {
            'valid': self.cert_valid,
            'expiry': self.cert_expiry,
            'cert_path': self.cert_path,
            'key_path': self.key_path,
            'days_until_expiry': (self.cert_expiry - datetime.datetime.utcnow()).days if self.cert_expiry else None
        }
    
    def generate_user_certificates(self, username):
        """Generate certificates for a specific user"""
        try:
            cert_paths = self.get_user_cert_paths(username)
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Generate public key
            public_key = private_key.public_key()
            
            # Create certificate subject and issuer
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, f"CCTV User: {username}"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Local Security"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, u"CCTV Unit")
            ])
            
            # Certificate validity period (1 year)
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                public_key
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
                critical=False
            ).sign(private_key, hashes.SHA256())
            
            # Save private key
            with open(cert_paths['key'], "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save certificate
            with open(cert_paths['cert'], "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            logging.info(f"Generated certificates for user: {username}")
            return True
            
        except Exception as e:
            logging.error(f"Error generating certificates for user {username}: {str(e)}")
            return False
            
    def verify_user_certificate(self, username):
        """Verify if a user's certificates exist and are valid"""
        if not username:
            logging.error("Cannot verify certificate: no username provided")
            return False
            
        try:
            # First check the database
            if check_certificate_validity(username):
                cert_data = get_user_certificate(username)
                if not cert_data:
                    logging.warning(f"No certificate data found in database for user {username}")
                    return False
                    
                # Write certificate to file system for SSL context
                cert_paths = self.get_user_cert_paths(username)
                try:
                    with open(cert_paths['cert'], 'w') as f:
                        f.write(cert_data['certificate'])
                    with open(cert_paths['key'], 'w') as f:
                        f.write(cert_data['private_key'])
                    return True
                except Exception as e:
                    logging.error(f"Error writing certificate files: {e}")
                    return False
            
            return False
                    
        except Exception as e:
            logging.error(f"Error verifying certificate for user {username}: {str(e)}")
            return False

def rotate_certificates(cert_manager):
    """Rotate certificates if they're approaching expiry"""
    status = cert_manager.get_certificate_status()
    if status['valid'] and status['days_until_expiry'] < 30:
        logging.info("Certificate rotation needed - generating new certificates")
        return cert_manager.generate_certificates()
    return True
