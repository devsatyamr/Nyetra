import sqlite3
import os

DB_PATH = "secure_cctv.db"

def alter_users_table():
    """Add totp_secret and backup_codes columns if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in c.fetchall()]
    if "totp_secret" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
        print("Added column: totp_secret")
    if "backup_codes" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN backup_codes TEXT")
        print("Added column: backup_codes")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} does not exist. Please run your main app to initialize it first.")
    else:
        alter_users_table()
        print("2FA columns ensured in users table. No secrets or backup codes generated.")