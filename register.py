"""
Registration module untuk sistem absensi
Handles user registration logic and validation
"""

import sqlite3
import re
from werkzeug.security import generate_password_hash
from datetime import datetime

class UserRegistration:
    def __init__(self, db_path='database.db'):
        self.db_path = db_path
    
    def get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def validate_username(self, username):
        """Validate username format and uniqueness"""
        if not username:
            return False, "Username tidak boleh kosong"
        
        if len(username) < 3:
            return False, "Username minimal 3 karakter"
        
        if len(username) > 50:
            return False, "Username maksimal 50 karakter"
        
        # Check alphanumeric and underscore only
        if not re.match("^[a-zA-Z0-9_]+$", username):
            return False, "Username hanya boleh huruf, angka, dan underscore"
        
        # Check if username exists
        conn = self.get_db_connection()
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if existing_user:
            return False, "Username sudah digunakan"
        
        return True, "Username valid"
    
    def validate_email(self, email):
        """Validate email format"""
        if not email:
            return True, "Email optional"  # Email is optional
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return False, "Format email tidak valid"
        
        # Check if email exists
        conn = self.get_db_connection()
        existing_email = conn.execute(
            'SELECT id FROM users WHERE email = ?', (email,)
        ).fetchone()
        conn.close()
        
        if existing_email:
            return False, "Email sudah terdaftar"
        
        return True, "Email valid"
    
    def validate_password(self, password):
        """Validate password strength"""
        if not password:
            return False, "Password tidak boleh kosong"
        
        if len(password) < 6:
            return False, "Password minimal 6 karakter"
        
        if len(password) > 128:
            return False, "Password maksimal 128 karakter"
        
        # Check for at least one letter and one number
        if not re.search(r'[A-Za-z]', password):
            return False, "Password harus mengandung minimal satu huruf"
        
        if not re.search(r'[0-9]', password):
            return False, "Password harus mengandung minimal satu angka"
        
        return True, "Password valid"
    
    def validate_full_name(self, full_name):
        """Validate full name"""
        if not full_name:
            return False, "Nama lengkap tidak boleh kosong"
        
        if len(full_name.strip()) < 2:
            return False, "Nama lengkap minimal 2 karakter"
        
        if len(full_name) > 100:
            return False, "Nama lengkap maksimal 100 karakter"
        
        # Only letters, spaces, and common punctuation
        if not re.match(r"^[a-zA-Z\s.,'-]+$", full_name):
            return False, "Nama lengkap hanya boleh huruf dan tanda baca umum"
        
        return True, "Nama lengkap valid"
    
    def register_user(self, username, password, full_name, email=None):
        """Register a new user"""
        try:
            # Validate all inputs
            username_valid, username_msg = self.validate_username(username)
            if not username_valid:
                return False, username_msg
            
            password_valid, password_msg = self.validate_password(password)
            if not password_valid:
                return False, password_msg
            
            name_valid, name_msg = self.validate_full_name(full_name)
            if not name_valid:
                return False, name_msg
            
            if email:
                email_valid, email_msg = self.validate_email(email)
                if not email_valid:
                    return False, email_msg
            
            # Hash password
            hashed_password = generate_password_hash(password)
            
            # Insert user into database
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, password, full_name, email, created_at, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (username.lower().strip(), hashed_password, full_name.strip(), email))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return True, f"User berhasil didaftarkan dengan ID: {user_id}"
            
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_user_stats(self):
        """Get user registration statistics"""
        try:
            conn = self.get_db_connection()
            
            # Total users
            total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
            
            # Active users
            active_users = conn.execute('SELECT COUNT(*) FROM users WHERE active = 1').fetchone()[0]
            
            # Users registered today
            today_users = conn.execute('''
                SELECT COUNT(*) FROM users 
                WHERE DATE(created_at) = DATE('now')
            ''').fetchone()[0]
            
            # Users registered this month
            month_users = conn.execute('''
                SELECT COUNT(*) FROM users 
                WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
            ''').fetchone()[0]
            
            conn.close()
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'today_users': today_users,
                'month_users': month_users
            }
            
        except Exception as e:
            return None
    
    def check_user_exists(self, username):
        """Check if user exists"""
        try:
            conn = self.get_db_connection()
            user = conn.execute(
                'SELECT id, username, full_name FROM users WHERE username = ?', 
                (username.lower().strip(),)
            ).fetchone()
            conn.close()
            
            return user is not None
            
        except Exception:
            return False

# Usage example and testing functions
def test_registration():
    """Test function untuk validasi"""
    reg = UserRegistration()
    
    # Test cases
    test_cases = [
        ("john_doe", "password123", "John Doe", "john@email.com"),
        ("jane", "pass123", "Jane Smith", "jane@email.com"),
        ("", "password123", "Test User", "test@email.com"),  # Invalid username
        ("valid_user", "123", "Valid User", "valid@email.com"),  # Invalid password
        ("another_user", "password123", "", "another@email.com"),  # Invalid name
    ]
    
    print("Testing Registration Module:")
    print("-" * 50)
    
    for username, password, full_name, email in test_cases:
        success, message = reg.register_user(username, password, full_name, email)
        status = "✓" if success else "✗"
        print(f"{status} {username:<12} | {message}")

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_registration()