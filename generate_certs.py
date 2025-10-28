from cert_manager import CertificateManager
from local_db import init_db, store_user_certificate
import sqlite3
from datetime import datetime, timedelta

def get_all_users():
    """Get list of all users from database"""
    conn = sqlite3.connect("secure_cctv.db")
    c = conn.cursor()
    c.execute('SELECT username FROM users')
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def main():
    """Generate initial SSL certificates and user certificates"""
    cert_manager = CertificateManager()
    
    # First generate server certificates
    if cert_manager.generate_certificates():
        print("Server SSL certificates generated successfully!")
        print(f"Certificate location: {cert_manager.cert_path}")
        print(f"Key location: {cert_manager.key_path}")
        
        status = cert_manager.get_certificate_status()
        print("\nCertificate Status:")
        print(f"Valid: {status['valid']}")
        print(f"Expiry: {status['expiry']}")
        print(f"Days until expiry: {status['days_until_expiry']}")
    else:
        print("Failed to generate server certificates!")
        return

    # Now generate certificates for all users
    init_db()
    users = get_all_users()
    print(f"\nGenerating certificates for {len(users)} users...")
    
    for username in users:
        print(f"\nGenerating certificates for user: {username}")
        if cert_manager.generate_certificates(username):
            print(f"Successfully generated certificates for {username}")
            if cert_manager.verify_user_certificate(username):
                print(f"Successfully verified certificate for {username}")
        else:
            print(f"Failed to generate certificates for {username}")

if __name__ == "__main__":
    main()
