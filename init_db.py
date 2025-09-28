import sqlite3
import os
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize database with tables and admin user"""
    
    # Create database connection
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    print("Creating database tables...")
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create coordinates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coordinates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius INTEGER DEFAULT 100,
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date DATE NOT NULL,
            time_in TIME,
            time_out TIME,
            latitude REAL,
            longitude REAL,
            latitude_out REAL,
            longitude_out REAL,
            photo_path TEXT,
            photo_path_out TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create attendance_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            success BOOLEAN DEFAULT 0,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create face_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            face_encoding TEXT,
            photo_path TEXT,
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create default admin user
    admin_password = generate_password_hash('admin.admin')
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, full_name, email, password, role)
        VALUES (?, ?, ?, ?, ?)
    ''', ('admin', 'Administrator', 'admin@example.com', admin_password, 'admin'))
    

    
    conn.commit()
    conn.close()
    
    print("Database initialized successfully!")
    print("Default admin user: admin / admin.admin")

if __name__ == '__main__':
    init_database()


