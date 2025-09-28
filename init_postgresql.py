# Create new file: init_postgresql.py

import os
import psycopg2
from urllib.parse import urlparse
from werkzeug.security import generate_password_hash

def get_postgres_connection():
    """Get PostgreSQL connection"""
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        raise Exception("DATABASE_URL environment variable not found")
    
    # Parse DATABASE_URL
    url = urlparse(DATABASE_URL)
    
    conn = psycopg2.connect(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )
    return conn

def init_postgresql_database():
    """Initialize PostgreSQL database with all required tables"""
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        print("Initializing PostgreSQL database...")
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                email VARCHAR(100),
                password VARCHAR(255) NOT NULL,
                role VARCHAR(20) DEFAULT 'user',
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create coordinates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coordinates (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                latitude DECIMAL(10, 8) NOT NULL,
                longitude DECIMAL(11, 8) NOT NULL,
                radius INTEGER DEFAULT 100,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                date DATE NOT NULL,
                time_in TIME,
                time_out TIME,
                latitude DECIMAL(10, 8),
                longitude DECIMAL(11, 8),
                latitude_out DECIMAL(10, 8),
                longitude_out DECIMAL(11, 8),
                photo_path VARCHAR(255),
                photo_path_out VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, date)
            )
        ''')
        
        # Create face_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_data (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                face_encoding TEXT,
                photo_path VARCHAR(255),
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create attendance_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                action VARCHAR(20) NOT NULL,
                latitude DECIMAL(10, 8),
                longitude DECIMAL(11, 8),
                success BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_user_date ON attendance(user_id, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_data_user ON face_data(user_id, active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_coordinates_active ON coordinates(active)')
        
        # Create admin user if not exists
        admin_password = generate_password_hash('admin123')
        cursor.execute('''
            INSERT INTO users (username, full_name, email, password, role, active)
            SELECT 'admin', 'Administrator', 'admin@system.local', %s, 'admin', TRUE
            WHERE NOT EXISTS (SELECT 1 FROM users WHERE username = 'admin')
        ''', (admin_password,))
        
        # Add default coordinate if none exists
        cursor.execute('''
            INSERT INTO coordinates (name, latitude, longitude, radius, active)
            SELECT 'Default Location', -6.2088, 106.8456, 100, TRUE
            WHERE NOT EXISTS (SELECT 1 FROM coordinates)
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✓ PostgreSQL database initialized successfully")
        print("✓ Admin user created (username: admin, password: admin123)")
        print("✓ Default coordinate added")
        return True
        
    except Exception as e:
        print(f"✗ Error initializing PostgreSQL database: {e}")
        return False

if __name__ == "__main__":
    init_postgresql_database()
