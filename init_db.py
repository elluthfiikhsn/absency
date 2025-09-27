import sqlite3
from datetime import datetime, date, timedelta
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize the SQLite database with all required tables"""
    
    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    print("Creating database tables...")
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user',
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date DATE NOT NULL,
            time_in TIME,
            time_out TIME,
            latitude REAL,
            longitude REAL,
            photo_path TEXT,
            status TEXT DEFAULT 'present',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id, date)
        )
    ''')
    
    # Create coordinates table for allowed locations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coordinates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius INTEGER DEFAULT 100,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create face_data table for face recognition
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            face_encoding TEXT NOT NULL,
            photo_path TEXT,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create settings table for app configuration
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT UNIQUE NOT NULL,
            setting_value TEXT,
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create attendance_logs table for detailed logging
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            latitude REAL,
            longitude REAL,
            device_info TEXT,
            ip_address TEXT,
            success INTEGER DEFAULT 1,
            error_message TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    print("Tables created successfully!")
    
    # Insert default admin user
    admin_password = generate_password_hash('admin123')  # Change this password!
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password, full_name, email, role)
        VALUES (?, ?, ?, ?, ?)
    ''', ('admin', admin_password, 'Administrator', 'admin@example.com', 'admin'))
    
    # # Insert sample coordinates (you can modify these)
    # sample_coordinates = [
    #     ('Kantor Pusat', -6.2088, 106.8456, 100),  # Jakarta coordinates example
    #     ('Kantor Cabang 1', -7.2575, 112.7521, 150),  # Surabaya coordinates example  
    #     ('Kantor Cabang 2', -6.9175, 107.6191, 100),  # Bandung coordinates example
    # ]
    
    # for coord in sample_coordinates:
    #     cursor.execute('''
    #         INSERT OR IGNORE INTO coordinates (name, latitude, longitude, radius)
    #         VALUES (?, ?, ?, ?)
    #     ''', coord)
    
    # Insert default settings
    default_settings = [
        ('app_name', 'Sistem Absensi', 'Nama aplikasi'),
        ('work_start_time', '08:00', 'Jam masuk kerja'),
        ('work_end_time', '17:00', 'Jam pulang kerja'),
        ('late_tolerance', '15', 'Toleransi keterlambatan (menit)'),
        ('location_radius', '100', 'Radius lokasi absensi (meter)'),
        ('require_photo', '1', 'Wajib foto saat absensi (1=ya, 0=tidak)'),
        ('face_recognition', '0', 'Aktifkan face recognition (1=ya, 0=tidak)'),
    ]
    
    for setting in default_settings:
        cursor.execute('''
            INSERT OR IGNORE INTO settings (setting_key, setting_value, description)
            VALUES (?, ?, ?)
        ''', setting)
    
    # Create indexes for better performance
    print("Creating indexes...")
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_user_date ON attendance(user_id, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coordinates_active ON coordinates(active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_data_user ON face_data(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_user_timestamp ON attendance_logs(user_id, timestamp)')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database initialization completed successfully!")
    print("\nDefault admin credentials:")
    print("Username: admin")
    print("Password: admin123")
    print("\nPlease change the admin password after first login!")

def reset_database():
    """Reset database by dropping all tables and recreating them"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    print("Resetting database...")
    
    # Drop all tables
    tables = ['attendance_logs', 'face_data', 'attendance', 'coordinates', 'settings', 'users']
    for table in tables:
        cursor.execute(f'DROP TABLE IF EXISTS {table}')
    
    conn.commit()
    conn.close()
    
    print("All tables dropped. Reinitializing...")
    init_database()

def add_sample_data():
    """Add sample data for testing"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    print("Adding sample data...")
    
    # Add sample users
    sample_users = [
        ('john_doe', generate_password_hash('password123'), 'John Doe', 'john@example.com'),
        ('jane_smith', generate_password_hash('password123'), 'Jane Smith', 'jane@example.com'),
        ('agus_setiawan', generate_password_hash('password123'), 'Agus Setiawan', 'agus@example.com'),
    ]
    
    for user in sample_users:
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password, full_name, email)
            VALUES (?, ?, ?, ?)
        ''', user)
    
    # Add sample attendance records
    today = date.today()
    
    # Get user IDs
    cursor.execute('SELECT id FROM users WHERE username IN ("john_doe", "jane_smith", "agus_setiawan")')
    user_ids = [row[0] for row in cursor.fetchall()]
    
    for user_id in user_ids:
        for i in range(5):  # Last 5 days
            attendance_date = today - timedelta(days=i)
            cursor.execute('''
                INSERT OR IGNORE INTO attendance (user_id, date, time_in, time_out, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, attendance_date, '08:30:00', '17:15:00', -6.2088, 106.8456))
    
    conn.commit()
    conn.close()
    
    print("Sample data added successfully!")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'reset':
            reset_database()
        elif sys.argv[1] == 'sample':
            add_sample_data()
        else:
            print("Usage: python init_db.py [reset|sample]")
    else:
        init_database()
    
    print("\nDatabase setup completed!")
    print("You can now run: python app.py")