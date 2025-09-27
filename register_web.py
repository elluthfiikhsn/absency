"""
Web interface untuk registrasi pengguna
Handles web-based registration with face recognition setup
"""

from flask import request, render_template, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import face_recognition
import numpy as np
import pickle
import json
import sqlite3
from register import UserRegistration

class WebRegistration:
    def __init__(self, app, upload_folder='faces'):
        self.app = app
        self.upload_folder = upload_folder
        self.user_reg = UserRegistration()
        
        # Ensure upload folder exists
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
    
    def allowed_file(self, filename):
        """Check if uploaded file is allowed"""
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def process_face_image(self, image_file, user_id, full_name):
        """Process uploaded face image for recognition"""
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
                # No face detected
                os.remove(image_path)  # Clean up
                return False, "Tidak ada wajah terdeteksi dalam gambar"
            
            if len(face_encodings) > 1:
                # Multiple faces detected
                return False, "Terdeteksi lebih dari satu wajah. Gunakan foto dengan satu wajah saja"
            
            # Get the face encoding
            face_encoding = face_encodings[0]
            
            # Convert to list for JSON serialization
            encoding_list = face_encoding.tolist()
            
            # Save encoding to database
            import sqlite3
            conn = sqlite3.connect('database.db')
            conn.execute('''
                INSERT INTO face_data (user_id, face_encoding, photo_path, active)
                VALUES (?, ?, ?, 1)
            ''', (user_id, json.dumps(encoding_list), image_path))
            conn.commit()
            conn.close()
            
            # Also save as pickle for faster loading
            encoding_file = os.path.join(user_folder, f"{user_id}_encoding.pkl")
            with open(encoding_file, 'wb') as f:
                pickle.dump(face_encoding, f)
            
            return True, "Wajah berhasil didaftarkan"
            
        except Exception as e:
            return False, f"Error processing face: {str(e)}"
    
    def handle_registration(self):
        """Handle web registration form submission"""
        if request.method == 'POST':
            # Get form data
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip() or None
            
            # Validate password confirmation
            if password != confirm_password:
                flash('Password dan konfirmasi password tidak cocok!', 'error')
                return render_template('register.html')
            
            # Register user
            success, message = self.user_reg.register_user(username, password, full_name, email)
            
            if success:
                # Get the user ID for face processing
                conn = sqlite3.connect('database.db')
                user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
                user_id = user[0] if user else None
                conn.close()
                
                face_success = True
                face_message = ""
                
                # Process face image if uploaded
                if 'face_image' in request.files:
                    face_file = request.files['face_image']
                    if face_file and face_file.filename != '' and self.allowed_file(face_file.filename):
                        face_success, face_message = self.process_face_image(face_file, user_id, full_name)
                
                if face_success:
                    flash(f'{message} Face recognition aktif!', 'success')
                else:
                    flash(f'{message} Namun gagal setup face recognition: {face_message}', 'warning')
                
                return redirect(url_for('login'))
            else:
                flash(message, 'error')
        
        return render_template('register.html')
    
    def verify_face(self, uploaded_image, user_id):
        """Verify face against stored encoding"""
        try:
            # Get stored face encoding from database
            import sqlite3
            conn = sqlite3.connect('database.db')
            face_data = conn.execute(
                'SELECT face_encoding FROM face_data WHERE user_id = ? AND active = 1',
                (user_id,)
            ).fetchone()
            conn.close()
            
            if not face_data:
                return False, "Face data not found for user"
            
            # Load stored encoding
            stored_encoding = np.array(json.loads(face_data[0]))
            
            # Process uploaded image
            image = face_recognition.load_image_file(uploaded_image)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                return False, "No face detected in uploaded image"
            
            # Compare faces
            matches = face_recognition.compare_faces([stored_encoding], face_encodings[0])
            face_distance = face_recognition.face_distance([stored_encoding], face_encodings[0])
            
            # Threshold for face matching (lower = more strict)
            threshold = 0.4
            
            if matches[0] and face_distance[0] < threshold:
                return True, f"Face verified! Confidence: {(1-face_distance[0])*100:.1f}%"
            else:
                return False, f"Face not recognized. Distance: {face_distance[0]:.3f}"
                
        except Exception as e:
            return False, f"Error verifying face: {str(e)}"
        
    
    def get_registration_stats(self):
        """Get registration statistics for admin"""
        stats = self.user_reg.get_user_stats()
        if stats:
            # Add face registration stats
            try:
                import sqlite3
                conn = sqlite3.connect('database.db')
                
                face_users = conn.execute(
                    'SELECT COUNT(DISTINCT user_id) FROM face_data WHERE active = 1'
                ).fetchone()[0]
                
                stats['face_registered'] = face_users
                stats['face_percentage'] = (face_users / stats['total_users'] * 100) if stats['total_users'] > 0 else 0
                
                conn.close()
                
            except Exception:
                stats['face_registered'] = 0
                stats['face_percentage'] = 0
        
        return stats
    
    def setup_routes(self):
        """Setup Flask routes for registration"""
        
        @self.app.route('/register', methods=['GET', 'POST'])
        def register_user():
            return self.handle_registration()
        
        @self.app.route('/api/check_username', methods=['POST'])
        def check_username():
            """API endpoint to check username availability"""
            username = request.json.get('username', '').strip()
            
            if not username:
                return jsonify({'available': False, 'message': 'Username tidak boleh kosong'})
            
            valid, message = self.user_reg.validate_username(username)
            return jsonify({'available': valid, 'message': message})
        
        @self.app.route('/api/verify_face', methods=['POST'])
        def verify_face_api():
            """API endpoint for face verification during attendance"""
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': 'Not logged in'})
            
            if 'face_image' not in request.files:
                return jsonify({'success': False, 'message': 'No face image provided'})
            
            face_file = request.files['face_image']
            if face_file.filename == '':
                return jsonify({'success': False, 'message': 'No file selected'})
            
            if not self.allowed_file(face_file.filename):
                return jsonify({'success': False, 'message': 'Invalid file format'})
            
            # Save temporary file for processing
            temp_path = os.path.join(self.upload_folder, 'temp_verify.jpg')
            face_file.save(temp_path)
            
            try:
                success, message = self.verify_face(temp_path, session['user_id'])
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return jsonify({'success': success, 'message': message})
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return jsonify({'success': False, 'message': f'Verification error: {str(e)}'})
        
        @self.app.route('/admin/registration_stats')
        def registration_stats():
            """Admin endpoint for registration statistics"""
            if 'user_id' not in session:
                return redirect(url_for('login'))
            
            # Check if user is admin (implement your admin check logic)
            # For now, checking username
            if session.get('username') != 'admin':
                flash('Access denied', 'error')
                return redirect(url_for('index'))
            
            stats = self.get_registration_stats()
            return render_template('admin/registration_stats.html', stats=stats)

# Function to initialize web registration
def init_web_registration(app):
    """Initialize web registration module"""
    web_reg = WebRegistration(app)
    web_reg.setup_routes()
    return web_reg

# Utility function for face recognition setup
def setup_face_recognition():
    """Setup face recognition dependencies"""
    try:
        import cv2
        import face_recognition
        return True
    except ImportError as e:
        print(f"Face recognition dependencies not installed: {e}")
        print("Install with: pip install opencv-python face_recognition")
        return False

if __name__ == "__main__":
    # Test face recognition setup
    if setup_face_recognition():
        print("✓ Face recognition dependencies are installed")
    else:
        print("✗ Face recognition dependencies missing")   