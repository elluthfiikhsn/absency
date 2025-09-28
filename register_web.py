from flask import request, render_template, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
import os
import numpy as np
import json
import psycopg2
import psycopg2.extras
import sqlite3
from urllib.parse import urlparse

# Coba import face recognition libs
try:
    import cv2
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️ Face recognition libraries not available. Running without face recognition.")

class WebRegistration:
    def __init__(self, app, upload_folder='faces'):
        self.app = app
        self.upload_folder = upload_folder
        self.DATABASE_URL = os.environ.get('DATABASE_URL')
        
        # Ensure upload folder exists
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
    
    def get_db_connection(self):
        """Get database connection - supports both SQLite and PostgreSQL"""
        if self.DATABASE_URL:
            # PostgreSQL connection for production
            try:
                url = urlparse(self.DATABASE_URL)
                conn = psycopg2.connect(
                    database=url.path[1:],
                    user=url.username,
                    password=url.password,
                    host=url.hostname,
                    port=url.port,
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                conn.autocommit = True
                return conn
            except Exception as e:
                print(f"PostgreSQL connection failed: {e}")
                raise
        else:
            # SQLite connection for local development
            conn = sqlite3.connect('database.db')
            conn.row_factory = sqlite3.Row
            return conn
    
    def allowed_file(self, filename):
        """Check if uploaded file is allowed"""
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def validate_username(self, username):
        """Validate username availability and format"""
        if len(username) < 3:
            return False, "Username minimal 3 karakter"
        
        if len(username) > 50:
            return False, "Username maksimal 50 karakter"
        
        # Check if username already exists
        conn = self.get_db_connection()
        try:
            if self.DATABASE_URL:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
                existing = cursor.fetchone()
                cursor.close()
            else:
                existing = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            
            conn.close()
            
            if existing:
                return False, "Username sudah digunakan"
            
            return True, "Username tersedia"
            
        except Exception as e:
            conn.close()
            return False, f"Error checking username: {str(e)}"
    
    def register_user(self, username, password, full_name, email=None):
        """Register new user in database"""
        try:
            # Validate input
            if len(password) < 6:
                return False, "Password minimal 6 karakter"
            
            if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
                return False, "Password harus mengandung huruf dan angka"
            
            # Hash password
            hashed_password = generate_password_hash(password)
            
            # Insert user
            conn = self.get_db_connection()
            
            if self.DATABASE_URL:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, full_name, email, password, role, active)
                    VALUES (%s, %s, %s, %s, 'user', TRUE)
                    RETURNING id
                ''', (username, full_name, email, hashed_password))
                result = cursor.fetchone()
                user_id = result['id'] if result else None
                cursor.close()
            else:
                cursor = conn.execute('''
                    INSERT INTO users (username, full_name, email, password, role, active)
                    VALUES (?, ?, ?, ?, 'user', 1)
                ''', (username, full_name, email, hashed_password))
                user_id = cursor.lastrowid
                conn.commit()
            
            conn.close()
            
            if user_id:
                return True, "Registrasi berhasil!"
            else:
                return False, "Gagal membuat user"
                
        except Exception as e:
            return False, f"Error registrasi: {str(e)}"
    
    def process_face_image(self, image_file, user_id, full_name):
        """Process uploaded face image for recognition"""
        if not FACE_RECOGNITION_AVAILABLE:
            return False, "Face recognition not available in this environment"
        
        try:
            # Create user-specific folder
            user_folder = os.path.join(self.upload_folder, f"{full_name}_{user_id}")
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)
            
            # Save original image
            filename = secure_filename(f"{user_id}_face.jpg")
            image_path = os.path.join(user_folder, filename)
            image_file.save(image_path)
            
            # Process with face_recognition
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                os.remove(image_path)  # Clean up
                return False, "Tidak ada wajah terdeteksi dalam gambar"
            
            if len(face_encodings) > 1:
                return False, "Terdeteksi lebih dari satu wajah. Gunakan foto dengan satu wajah saja"
            
            face_encoding = face_encodings[0]
            
            # Convert to list for JSON serialization
            encoding_list = face_encoding.tolist()
            
            # Save encoding to database
            conn = self.get_db_connection()
            
            if self.DATABASE_URL:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO face_data (user_id, face_encoding, photo_path, active)
                    VALUES (%s, %s, %s, TRUE)
                ''', (user_id, json.dumps(encoding_list), image_path))
                cursor.close()
            else:
                conn.execute('''
                    INSERT INTO face_data (user_id, face_encoding, photo_path, active)
                    VALUES (?, ?, ?, 1)
                ''', (user_id, json.dumps(encoding_list), image_path))
                conn.commit()
            
            conn.close()
            
            return True, "Wajah berhasil didaftarkan"
            
        except Exception as e:
            return False, f"Error processing face: {str(e)}"
    
    def handle_registration(self):
        """Handle web registration form submission"""
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip() or None
            
            # Basic validation
            if not all([username, full_name, password, confirm_password]):
                flash('Semua field wajib harus diisi!', 'error')
                return render_template('register.html')
            
            if password != confirm_password:
                flash('Password dan konfirmasi password tidak cocok!', 'error')
                return render_template('register.html')
            
            # Check username availability
            valid_username, username_message = self.validate_username(username)
            if not valid_username:
                flash(username_message, 'error')
                return render_template('register.html')
            
            # Register user
            success, message = self.register_user(username, password, full_name, email)
            
            if success:
                # Get user ID for face processing
                conn = self.get_db_connection()
                if self.DATABASE_URL:
                    cursor = conn.cursor()
                    cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
                    user = cursor.fetchone()
                    cursor.close()
                else:
                    user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
                
                user_id = user['id'] if user else None
                conn.close()
                
                face_success, face_message = True, ""
                
                # Process face image if provided
                if 'face_image' in request.files:
                    face_file = request.files['face_image']
                    if face_file and face_file.filename != '' and self.allowed_file(face_file.filename):
                        face_success, face_message = self.process_face_image(face_file, user_id, full_name)
                
                if face_success and face_message:
                    flash(f'{message} Face recognition aktif!', 'success')
                elif face_success:
                    flash(f'{message} Silakan setup face recognition di profil nanti.', 'success')
                else:
                    flash(f'{message} Namun gagal setup face recognition: {face_message}', 'warning')
                
                return redirect(url_for('login'))
            else:
                flash(message, 'error')
        
        return render_template('register.html')
    
    def verify_face(self, uploaded_image, user_id):
        """Verify face against stored encoding"""
        if not FACE_RECOGNITION_AVAILABLE:
            return False, "Face recognition not available in this environment"
        
        try:
            conn = self.get_db_connection()
            
            if self.DATABASE_URL:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT face_encoding FROM face_data WHERE user_id = %s AND active = TRUE',
                    (user_id,)
                )
                face_data = cursor.fetchone()
                cursor.close()
            else:
                face_data = conn.execute(
                    'SELECT face_encoding FROM face_data WHERE user_id = ? AND active = 1',
                    (user_id,)
                ).fetchone()
            
            conn.close()
            
            if not face_data or not face_data['face_encoding']:
                return False, "Face data not found for user"
            
            stored_encoding = np.array(json.loads(face_data['face_encoding']))
            
            image = face_recognition.load_image_file(uploaded_image)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                return False, "No face detected in uploaded image"
            
            matches = face_recognition.compare_faces([stored_encoding], face_encodings[0])
            face_distance = face_recognition.face_distance([stored_encoding], face_encodings[0])
            
            threshold = 0.4
            
            if matches[0] and face_distance[0] < threshold:
                return True, f"Face verified! Confidence: {(1-face_distance[0])*100:.1f}%"
            else:
                return False, f"Face not recognized. Distance: {face_distance[0]:.3f}"
                
        except Exception as e:
            return False, f"Error verifying face: {str(e)}"
    
    def get_user_stats(self):
        """Get user statistics"""
        try:
            conn = self.get_db_connection()
            
            if self.DATABASE_URL:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) as count FROM users WHERE active = TRUE')
                total_result = cursor.fetchone()
                
                cursor.execute('SELECT COUNT(DISTINCT user_id) as count FROM face_data WHERE active = TRUE')
                face_result = cursor.fetchone()
                cursor.close()
            else:
                total_result = conn.execute('SELECT COUNT(*) as count FROM users WHERE active = 1').fetchone()
                face_result = conn.execute('SELECT COUNT(DISTINCT user_id) as count FROM face_data WHERE active = 1').fetchone()
            
            conn.close()
            
            total_users = total_result['count']
            face_users = face_result['count']
            
            return {
                'total_users': total_users,
                'face_registered': face_users,
                'face_percentage': (face_users / total_users * 100) if total_users > 0 else 0
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'total_users': 0,
                'face_registered': 0,
                'face_percentage': 0
            }
    
    def setup_routes(self):
        """Setup Flask routes for registration"""
        
        @self.app.route('/api/check_username', methods=['POST'])
        def check_username():
            data = request.get_json() or {}
            username = data.get('username', '').strip()
            if not username:
                return jsonify({'available': False, 'message': 'Username tidak boleh kosong'})
            valid, message = self.validate_username(username)
            return jsonify({'available': valid, 'message': message})
        
        @self.app.route('/api/verify_face', methods=['POST'])
        def verify_face_api():
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': 'Not logged in'})
            if 'face_image' not in request.files:
                return jsonify({'success': False, 'message': 'No face image provided'})
            
            face_file = request.files['face_image']
            if face_file.filename == '':
                return jsonify({'success': False, 'message': 'No file selected'})
            if not self.allowed_file(face_file.filename):
                return jsonify({'success': False, 'message': 'Invalid file format'})
            
            temp_path = os.path.join(self.upload_folder, 'temp_verify.jpg')
            face_file.save(temp_path)
            
            try:
                success, message = self.verify_face(temp_path, session['user_id'])
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return jsonify({'success': success, 'message': message})
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return jsonify({'success': False, 'message': f'Verification error: {str(e)}'})
        
        @self.app.route('/admin/registration_stats')
        def registration_stats():
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if session.get('role') != 'admin':
                flash('Access denied', 'error')
                return redirect(url_for('index'))
            stats = self.get_user_stats()
            return render_template('admin/registration_stats.html', stats=stats)

# Function to initialize web registration
def init_web_registration(app):
    """Initialize web registration with the Flask app"""
    try:
        web_reg = WebRegistration(app)
        web_reg.setup_routes()
        print("✅ Web registration initialized successfully")
        return web_reg
    except Exception as e:
        print(f"❌ Error initializing web registration: {e}")
        return None

# Utility function for face recognition setup
def setup_face_recognition():
    return FACE_RECOGNITION_AVAILABLE

if __name__ == "__main__":
    if setup_face_recognition():
        print("✓ Face recognition dependencies are installed")
    else:
        print("✗ Face recognition dependencies missing")
