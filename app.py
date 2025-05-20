# app.py - Main script for video processing with integrated crowd monitoring
import cv2
import threading
import time
import numpy as np
import imutils
import os
import datetime
import json
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.spatial.distance import euclidean

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crowdmonitoringsecretkey123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Video processing globals
outputFrame = None
lock = threading.Lock()
processing_stats = {
    "crowd_count": 0,
    "social_distance_violations": 0,
    "restricted_entry": False,
    "abnormal_activity": False
}

# Store threads and stop events
video_feed_threads = {}
stop_events = {}

# Define RGB colors for visualization
RGB_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0)
}

# Camera and Processing Parameters
class VideoConfig:
    def __init__(self, location):
        self.config = {
            "kapaleshwar_temple": {
                "VIDEO_CAP": "static/videos/temple.mp4",
                "IS_CAM": False,
                "HIGH_CAM": False,
                "START_TIME": datetime.datetime.now(),
                "CAM_APPROX_FPS": 30
            },
            "godavari_ghat": {
                "VIDEO_CAP": "static/videos/ghat.mp4",
                "IS_CAM": False,
                "HIGH_CAM": True,
                "START_TIME": datetime.datetime.now(),
                "CAM_APPROX_FPS": 30
            },
            "sita_gufa": {
                "VIDEO_CAP": "static/videos/cave.mp4",
                "IS_CAM": False,
                "HIGH_CAM": False,
                "START_TIME": datetime.datetime.now(),
                "CAM_APPROX_FPS": 30
            }
        }
        self.current = self.config.get(location, self.config["kapaleshwar_temple"])

# Processing constants
FRAME_SIZE = 640
SOCIAL_DISTANCE = 50
TRACK_MAX_AGE = 30
RE_START_TIME = datetime.time(22, 0)  # 10:00 PM
RE_END_TIME = datetime.time(5, 0)     # 5:00 AM
ABNORMAL_ENERGY = 1000
ABNORMAL_THRESH = 0.3
ABNORMAL_MIN_PEOPLE = 5

# Helper for tracking and detection
class Track:
    def __init__(self, track_id, bbox, position):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.positions = [position]  # [(x, y)]
        self.entry = 0
        self.exit = 0
    
    def to_tlbr(self):
        return np.array([self.bbox[0], self.bbox[1], 
                         self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]])

def rect_distance(rect1, rect2):
    """Calculate distance between two rectangles"""
    x1, y1, w1, h1 = rect1
    x3, y3, w2, h2 = rect2
    
    x2, y2 = x1 + w1, y1 + h1
    x4, y4 = x3 + w2, y3 + h2
    
    left = x4 < x1
    right = x2 < x3
    bottom = y4 < y1
    top = y2 < y3
    
    if top and left:
        return np.linalg.norm(np.array([x1, y2]) - np.array([x4, y3]))
    elif left and bottom:
        return np.linalg.norm(np.array([x1, y1]) - np.array([x4, y4]))
    elif bottom and right:
        return np.linalg.norm(np.array([x2, y1]) - np.array([x3, y4]))
    elif right and top:
        return np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))
    elif left:
        return x1 - x4
    elif right:
        return x3 - x2
    elif bottom:
        return y1 - y4
    elif top:
        return y3 - y2
    else:
        return 0

def kinetic_energy(pos1, pos2, time_step):
    """Calculate kinetic energy based on positions and time step"""
    if not pos1 or not pos2:
        return 0
    
    velocity = np.linalg.norm(np.array(pos1) - np.array(pos2)) / time_step
    return 0.5 * velocity * velocity

def detect_human(frame):
    """Detect humans in a frame (simulated for this example)"""
    height, width = frame.shape[:2]
    
    # Simulate detecting random number of people
    num_detections = np.random.randint(3, 10)
    humans_detected = []
    
    for i in range(num_detections):
        # Generate random bounding box
        x = np.random.randint(0, width - 100)
        y = np.random.randint(0, height - 200)
        w = np.random.randint(50, 100)
        h = np.random.randint(100, 200)
        
        # Create centroid position
        position = (x + w//2, y + h//2)
        
        # Create a track object
        track = Track(i, [x, y, w, h], position)
        # Add a second position for velocity calculation
        track.positions.append((position[0] + np.random.randint(-5, 5), 
                               position[1] + np.random.randint(-5, 5)))
        
        humans_detected.append(track)
    
    return humans_detected, []  # Returns detected humans and empty expired list

def video_process_thread(location, stop_event):
    """Process video for crowd monitoring"""
    global outputFrame, lock, processing_stats
    
    # Initialize configurations
    video_config = VideoConfig(location)
    
    # Open video capture
    video_path = video_config.current["VIDEO_CAP"]
    if not os.path.exists(video_path):
        # If video file doesn't exist, use a simulated video
        video_path = 0  # Use webcam if available, otherwise will fail gracefully
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # If video can't be opened, simulate frames
        class DummyCap:
            def read(self):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Simulated Video Feed - {location}", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                time.sleep(0.033)  # ~30 FPS
                return True, frame
            
            def release(self):
                pass
        
        cap = DummyCap()
    
    frame_count = 0
    frame_skip = 5  # Process every 5th frame
    time_step = 1.0 / 30.0  # Assuming 30 FPS
    
    # Tracking state
    re_warning_timeout = 0
    sd_warning_timeout = 0
    ab_warning_timeout = 0
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Loop video for demonstration
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # Resize frame
            frame = imutils.resize(frame, width=FRAME_SIZE)
            
            # Process every frame_skip frame for efficiency
            if frame_count % frame_skip != 0:
                with lock:
                    outputFrame = frame.copy()
                continue
            
            # Get current time
            current_datetime = datetime.datetime.now()
            
            # Run detection - implement actual detection if you have YOLO weights
            humans_detected, expired = detect_human(frame)
            
            # Check for restricted entry
            RE = False
            if (current_datetime.time() > RE_START_TIME) or (current_datetime.time() < RE_END_TIME):
                if len(humans_detected) > 0:
                    RE = True
                    re_warning_timeout = 10
            else:
                re_warning_timeout = max(0, re_warning_timeout - 1)
            
            # Initialize variables for visualization
            violate_set = set()
            violate_count = np.zeros(len(humans_detected))
            abnormal_individual = []
            ABNORMAL = False
            
            # Process each detected human
            for i, track in enumerate(humans_detected):
                # Get bounding box
                bbox = track.to_tlbr().astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                # Check for social distance violation
                if len(humans_detected) >= 2:
                    for j, track_2 in enumerate(humans_detected[i+1:], start=i+1):
                        bbox2 = track_2.to_tlbr().astype(int)
                        x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
                        
                        if video_config.current["HIGH_CAM"]:
                            cx, cy = track.positions[0]
                            cx_2, cy_2 = track_2.positions[0]
                            distance = np.linalg.norm(np.array([cx, cy]) - np.array([cx_2, cy_2]))
                        else:
                            distance = rect_distance((x, y, w, h), (x2, y2, w2, h2))
                        
                        if distance < SOCIAL_DISTANCE:
                            violate_set.add(i)
                            violate_count[i] += 1
                            violate_set.add(j)
                            violate_count[j] += 1
                
                # Check for abnormal movement
                ke = kinetic_energy(track.positions[0], track.positions[1], time_step)
                if ke > ABNORMAL_ENERGY:
                    abnormal_individual.append(track.track_id)
                
                # Draw bounding boxes based on status
                if RE:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), RGB_COLORS["red"], 2)  # Red for restricted entry
                elif i in violate_set:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), RGB_COLORS["yellow"], 2)  # Yellow for violation
                    cv2.putText(frame, str(int(violate_count[i])), (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["yellow"], 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), RGB_COLORS["green"], 2)  # Green for normal
                
                # Show tracking ID
                cv2.putText(frame, str(track.track_id), (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)
            
            # Check for abnormal activity
            if len(humans_detected) > ABNORMAL_MIN_PEOPLE:
                if len(abnormal_individual) / len(humans_detected) > ABNORMAL_THRESH:
                    ABNORMAL = True
                    ab_warning_timeout = 10
                else:
                    ab_warning_timeout = max(0, ab_warning_timeout - 1)
            
            # Display warnings on the frame
            if len(violate_set) > 0:
                sd_warning_timeout = 10
                text = f"Violation count: {len(violate_set)}"
                cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, RGB_COLORS["red"], 2)
            else:
                sd_warning_timeout = max(0, sd_warning_timeout - 1)
            
            if re_warning_timeout > 0 and frame_count % 3 != 0:
                cv2.putText(frame, "RESTRICTED ENTRY", 
                           (frame.shape[1]//4, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, RGB_COLORS["red"], 3)
            
            if ab_warning_timeout > 0 and frame_count % 3 != 0:
                cv2.putText(frame, "ABNORMAL ACTIVITY", 
                           (frame.shape[1]//4, frame.shape[0]//2 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, RGB_COLORS["blue"], 3)
            
            # Display crowd count - Large and prominent
            text = f"Crowd Count: {len(humans_detected)}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RGB_COLORS["black"], 6)  # Black shadow
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RGB_COLORS["white"], 2)  # White text
            
            # Update processing stats
            processing_stats["crowd_count"] = len(humans_detected)
            processing_stats["social_distance_violations"] = len(violate_set)
            processing_stats["restricted_entry"] = RE
            processing_stats["abnormal_activity"] = ABNORMAL
            
            # Update the output frame
            with lock:
                outputFrame = frame.copy()
            
            # Simulate real-time processing
            time.sleep(0.1)
    except Exception as e:
        print(f"Video processing error: {e}")
    finally:
        cap.release()

def generate_frames():
    """Generate video frames for streaming"""
    global outputFrame, lock
    
    while True:
        # Wait until the lock is acquired
        with lock:
            # Check if the output frame is available
            if outputFrame is None:
                continue
            
            # Encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", outputFrame)
            
            # Ensure the frame was successfully encoded
            if not flag:
                continue
        
        # Yield the output frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        
        # Delay to simulate real-time streaming
        time.sleep(0.033)  # ~30 FPS

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        
        if user_exists:
            flash('Username already exists')
        elif email_exists:
            flash('Email already exists')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            
            login_user(new_user)
            return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/monitor/<location>')
@login_required
def monitor(location):
    global video_feed_threads, stop_events
    
    # Stop any existing thread for this location
    if location in stop_events:
        stop_events[location].set()
        if location in video_feed_threads and video_feed_threads[location].is_alive():
            video_feed_threads[location].join(timeout=1.0)
    
    # Create a new stop event
    stop_events[location] = threading.Event()
    
    # Create and start thread
    video_feed_threads[location] = threading.Thread(
        target=video_process_thread, 
        args=(location, stop_events[location])
    )
    video_feed_threads[location].daemon = True
    video_feed_threads[location].start()
    
    return render_template('monitor.html', location=location)

@app.route('/stop_monitor/<location>')
@login_required
def stop_monitor(location):
    if location in stop_events:
        stop_events[location].set()  # Signal the thread to stop
        if location in video_feed_threads and video_feed_threads[location].is_alive():
            video_feed_threads[location].join(timeout=1.0)
    
    return redirect(url_for('dashboard'))

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
@login_required
def get_stats():
    global processing_stats
    return jsonify(processing_stats)

# Initialize the database
with app.app_context():
    db.create_all()
    # Create a default admin user if not exists
    if not User.query.filter_by(username='admin').first():
        hashed_password = generate_password_hash('admin123', method='pbkdf2:sha256')
        admin = User(username='admin', email='admin@example.com', password=hashed_password)
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    # Create directories if they don't exist
    if not os.path.exists('static/videos'):
        os.makedirs('static/videos')
    
    # Create empty placeholder videos if they don't exist
    for video_name in ['temple.mp4', 'ghat.mp4', 'cave.mp4']:
        if not os.path.exists(f'static/videos/{video_name}'):
            print(f"Warning: Video file {video_name} not found. Please add it to static/videos/ directory.")
    
    app.run(debug=True)