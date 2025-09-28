from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
from datetime import datetime
import json
import pandas as pd
from io import BytesIO
from flask import send_file
import sys
import platform

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FACES_FOLDER'] = 'faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACES_FOLDER'], exist_ok=True)

# Face recognition disabled untuk Railway
FACE_RECOGNITION_AVAILABLE = False
FACE_RECOGNITION_ERROR = "Running in fallback mode - photo upload only"

print("Application starting in fallback mode (no AI face recognition)")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def init_database():
    """Initialize basic database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance table
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
    
    # Face data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            face_encoding TEXT,
            photo_path TEXT,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Coordinates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coordinates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius INTEGER DEFAULT 100,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            latitude REAL,
            longitude REAL,
            success INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create admin user if not exists
    cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        admin_password = generate_password_hash('admin123')
        cursor.execute('''
            INSERT INTO users (username, full_name, email, password, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'Administrator', 'admin@localhost', admin_password, 'admin'))
    
    conn.commit()
    conn.close()

# Initialize database on startup
if not os.path.exists('database.db'):
    init_database()
    print("Database initialized")

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    today = datetime.now().strftime("%Y-%m-%d")
    attendance = conn.execute(
        'SELECT * FROM attendance WHERE user_id = ? AND date = ?',
        (session['user_id'], today)
    ).fetchone()
    
    stats = conn.execute(
        '''SELECT 
           COUNT(*) as total_days,
           SUM(CASE WHEN time_in IS NOT NULL THEN 1 ELSE 0 END) as present_days
           FROM attendance WHERE user_id = ?''',
        (session['user_id'],)
    ).fetchone()
    
    face_enabled = conn.execute(
        'SELECT COUNT(*) FROM face_data WHERE user_id = ? AND active = 1',
        (session['user_id'],)
    ).fetchone()[0] > 0
    
    conn.close()
    
    return render_template('index.html', 
                         user=user, 
                         attendance=attendance, 
                         stats=stats,
                         face_enabled=face_enabled,
                         face_recognition_available=FACE_RECOGNITION_AVAILABLE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            flash('Login berhasil!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Username atau password salah!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('login'))

@app.route('/profil', methods=['GET', 'POST'])
@login_required
def profil():
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    face_data = conn.execute(
        'SELECT * FROM face_data WHERE user_id = ? AND active = 1',
        (session['user_id'],)
    ).fetchone()
    
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']

        if password:
            hashed_pw = generate_password_hash(password)
            conn.execute("""
                UPDATE users SET full_name=?, email=?, password=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (full_name, email, hashed_pw, session['user_id']))
        else:
            conn.execute("""
                UPDATE users SET full_name=?, email=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (full_name, email, session['user_id']))
        
        session['full_name'] = full_name
        conn.commit()
        conn.close()
        flash("Profil berhasil diperbarui!", "success")
        return redirect(url_for('profil'))

    user_dict = dict(user) if user else {}
    if user_dict.get('created_at'):
        try:
            user_dict['created_at'] = datetime.fromisoformat(user_dict['created_at'].replace('Z', '+00:00'))
        except:
            user_dict['created_at'] = None

    conn.close()
    return render_template('profil.html', 
                         user=user_dict, 
                         face_data=face_data,
                         face_recognition_available=FACE_RECOGNITION_AVAILABLE)

@app.route('/setup_face', methods=['POST'])
@login_required
def setup_face():
    if 'face_image' not in request.files:
        return jsonify({'success': False, 'message': 'Tidak ada foto yang diupload'})
    
    face_file = request.files['face_image']
    if face_file.filename == '':
        return jsonify({'success': False, 'message': 'Tidak ada file yang dipilih'})
    
    if not allowed_file(face_file.filename):
        return jsonify({'success': False, 'message': 'Format file tidak didukung'})
    
    try:
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        
        user_folder = os.path.join(app.config['FACES_FOLDER'], f"{user['full_name']}_{user['id']}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
        filename = secure_filename(f"{user['id']}_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        image_path = os.path.join(user_folder, filename)
        face_file.save(image_path)
        
        conn.execute('UPDATE face_data SET active = 0 WHERE user_id = ?', (session['user_id'],))
        conn.execute('''
            INSERT INTO face_data (user_id, photo_path, active)
            VALUES (?, ?, 1)
        ''', (session['user_id'], image_path))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'Foto wajah berhasil disimpan! (Mode manual - tanpa AI processing)'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/remove_face', methods=['POST'])
@login_required
def remove_face():
    try:
        conn = get_db_connection()
        face_data = conn.execute(
            'SELECT photo_path FROM face_data WHERE user_id = ? AND active = 1',
            (session['user_id'],)
        ).fetchall()
        
        conn.execute('UPDATE face_data SET active = 0 WHERE user_id = ?', (session['user_id'],))
        conn.commit()
        conn.close()
        
        for data in face_data:
            if data['photo_path'] and os.path.exists(data['photo_path']):
                try:
                    os.remove(data['photo_path'])
                except Exception:
                    pass
        
        return jsonify({'success': True, 'message': 'Face recognition berhasil dihapus!'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/debug/status')
def debug_status():
    return jsonify({
        'status': 'running',
        'mode': 'fallback',
        'face_recognition_available': FACE_RECOGNITION_AVAILABLE,
        'message': 'Application running in fallback mode - photo upload only'
    })

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
