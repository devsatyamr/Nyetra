#!/usr/bin/env python3
"""
Defense CCTV System - Backup Code Setup
Generate and display backup codes for all users
"""

import sys
sys.path.append('.')

from local_db import init_db, setup_backup_codes_for_all_users, get_user_role

def main():
    print("🛡️  DEFENSE CCTV SYSTEM - BACKUP CODE SETUP")
    print("=" * 50)
    
    # Initialize database (will add columns if needed)
    init_db()
    
    # Set up backup codes for all users
    print("🔐 Setting up backup codes...")
    backup_codes = setup_backup_codes_for_all_users()
    
    print("\n" + "=" * 60)
    print("🛡️  DEFENSE CCTV BACKUP CODES")
    print("=" * 60)
    for username, codes in backup_codes.items():
        print(f"\n👤 USER: {username.upper()}")
        print(f"   Role: {get_user_role(username)}")
        print("   Backup Codes:")
        for i, code in enumerate(codes, 1):
            print(f"   {i}. {code}")
    
    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT SECURITY NOTICES:")
    print("⚠️  • Save these codes securely!")
    print("⚠️  • Each code can only be used once!")
    print("⚠️  • Use backup codes when OTP is unavailable!")
    print("⚠️  • Check 'Use backup code' option in login!")
    print("=" * 60)
    
    print("\n✅ Backup codes setup complete!")
    print("🔐 You can now login using backup codes instead of OTP when needed.")

if __name__ == "__main__":
    main()