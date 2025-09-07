from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import logging
import threading
from datetime import datetime, timedelta
import os
import gc
import torch
from flask_socketio import SocketIO, emit
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
MODEL_PATH = '???????????????????' #fill in the model path
ALLOWED_JETSONS = {
    'jetson1': {'api_key': os.getenv('JETSON1_API_KEY', 'secretkey1')},
    'jetson2': {'api_key': os.getenv('JETSON2_API_KEY', 'secretkey2')}
}

# Detection state tracking
DETECTION_TIMEOUT = 2.0  # seconds - how long to consider a detection "active"

class WeaponDetectionServer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.inference_count = 0
        self.total_inference_time = 0
        self.last_detection_time = None
        self.lock = threading.Lock()
        self.stats_per_client = {}
        self.start_time = time.time()
        
        # Detection state tracking
        self.detection_states = {
            'jetson1': {
                'last_detection': None,
                'currently_detecting': False,
                'connected': False
            },
            'jetson2': {
                'last_detection': None,
                'currently_detecting': False,
                'connected': False
            }
        }
        
        self.initialize_model()

    def initialize_model(self):
        try:
            logger.info(f"Loading model from: {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = YOLO(self.model_path)

            if torch.cuda.is_available():
                logger.info(f"CUDA available - GPU: {torch.cuda.get_device_name(0)}")
                self.model.to('cuda')
            else:
                logger.info("CUDA not available - using CPU")

            dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            logger.info("Model initialized successfully")
            logger.info(f"Model classes: {list(self.model.names.values())}")
            return True

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False

    def perform_inference(self, image, client_id):
        start_time = time.time()
        detections = []
        weapon_detected = False

        try:
            results = self.model(image, verbose=False, conf=0.3)

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()

                        detection = {
                            'class': self.model.names[cls],
                            'confidence': round(conf, 3),
                            'box': [int(x) for x in xyxy]
                        }
                        detections.append(detection)
                        weapon_detected = True

            inference_time = time.time() - start_time

            # Update detection state
            with self.lock:
                current_time = datetime.now()
                
                if client_id not in self.stats_per_client:
                    self.stats_per_client[client_id] = {
                        'frames': 0,
                        'detections': 0,
                        'total_time': 0
                    }

                stats = self.stats_per_client[client_id]
                stats['frames'] += 1
                stats['total_time'] += inference_time
                
                # Update detection state
                if weapon_detected:
                    stats['detections'] += 1
                    self.last_detection_time = current_time
                    self.detection_states[client_id]['last_detection'] = current_time
                    self.detection_states[client_id]['currently_detecting'] = True
                else:
                    self.detection_states[client_id]['currently_detecting'] = False

            if weapon_detected:
                logger.warning(f"[{client_id}] WEAPON DETECTED! {len(detections)} objects in {inference_time:.3f}s")
                for det in detections:
                    logger.warning(f"  - {det['class']}: {det['confidence']:.3f}")

            return detections, inference_time

        except Exception as e:
            logger.error(f"Inference error from {client_id}: {e}")
            return [], time.time() - start_time

    def update_detection_states(self):
        """Update detection states based on timeout"""
        current_time = datetime.now()
        timeout_delta = timedelta(seconds=DETECTION_TIMEOUT)
        
        with self.lock:
            for client_id in self.detection_states:
                state = self.detection_states[client_id]
                if (state['last_detection'] and 
                    current_time - state['last_detection'] > timeout_delta):
                    state['currently_detecting'] = False

    def get_alarm_commands(self):
        """Determine alarm commands for both devices based on current detection states"""
        self.update_detection_states()
        
        with self.lock:
            jetson1_detecting = self.detection_states['jetson1']['currently_detecting']
            jetson2_detecting = self.detection_states['jetson2']['currently_detecting']
            
            commands = {}
            
            if jetson1_detecting and jetson2_detecting:
                # Both devices detecting - both play ALARM_BOTH
                commands['jetson1'] = 'BOTH'
                commands['jetson2'] = 'BOTH'
                logger.info("ALARM STATE: Both devices detecting weapons")
                
            elif jetson1_detecting and not jetson2_detecting:
                # Only jetson1 detecting
                commands['jetson1'] = 'ME'      # jetson1 plays ALARM_ME
                commands['jetson2'] = 'OTHER'   # jetson2 plays ALARM_OTHER
                logger.info("ALARM STATE: Jetson1 detecting weapon")
                
            elif jetson2_detecting and not jetson1_detecting:
                # Only jetson2 detecting
                commands['jetson1'] = 'OTHER'   # jetson1 plays ALARM_OTHER
                commands['jetson2'] = 'ME'      # jetson2 plays ALARM_ME
                logger.info("ALARM STATE: Jetson2 detecting weapon")
                
            else:
                # No detections
                commands['jetson1'] = 'NONE'
                commands['jetson2'] = 'NONE'
            
            return commands

    def set_client_connected(self, client_id, connected):
        with self.lock:
            if client_id in self.detection_states:
                self.detection_states[client_id]['connected'] = connected
                logger.info(f"Client {client_id} connection status: {connected}")

    def get_all_stats(self):
        with self.lock:
            all_stats = {}
            for cid, stats in self.stats_per_client.items():
                avg_time = stats['total_time'] / stats['frames'] if stats['frames'] else 0
                all_stats[cid] = {
                    'frames': stats['frames'],
                    'detections': stats['detections'],
                    'avg_inference_time': round(avg_time, 3),
                    'last_detection': self.last_detection_time.isoformat() if self.last_detection_time else 'None',
                    'currently_detecting': self.detection_states[cid]['currently_detecting'],
                    'connected': self.detection_states[cid]['connected']
                }
        return all_stats

# Initialize
detection_server = WeaponDetectionServer(MODEL_PATH)

def authorize(request):
    api_key = request.headers.get('X-API-Key')
    for client_id, info in ALLOWED_JETSONS.items():
        if info['api_key'] == api_key:
            return client_id
    return None

def broadcast_alarm_commands():
    """Broadcast current alarm commands to all connected clients"""
    commands = detection_server.get_alarm_commands()
    
    socketio.emit('alarm_command', {
        'jetson1': commands.get('jetson1', 'NONE'),
        'jetson2': commands.get('jetson2', 'NONE'),
        'timestamp': datetime.now().isoformat()
    })

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('register_client')
def handle_register_client(data):
    client_id = data.get('client_id')
    api_key = data.get('api_key')
    
    # Verify client
    if (client_id in ALLOWED_JETSONS and 
        ALLOWED_JETSONS[client_id]['api_key'] == api_key):
        detection_server.set_client_connected(client_id, True)
        emit('registration_success', {'client_id': client_id})
        logger.info(f"Client {client_id} registered successfully")
        
        # Send current alarm state
        commands = detection_server.get_alarm_commands()
        emit('alarm_command', {
            'jetson1': commands.get('jetson1', 'NONE'),
            'jetson2': commands.get('jetson2', 'NONE'),
            'timestamp': datetime.now().isoformat()
        })
    else:
        emit('registration_failed', {'error': 'Invalid credentials'})

@app.route('/infer', methods=['POST'])
def infer():
    client_id = authorize(request)
    if not client_id:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        if detection_server.model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        file = request.files['image']
        image_data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        detections, inference_time = detection_server.perform_inference(image, client_id)

        if detection_server.inference_count % 100 == 0:
            gc.collect()

        # Broadcast updated alarm commands to all clients
        broadcast_alarm_commands()

        return jsonify({
            'client_id': client_id,
            'detections': detections,
            'inference_time': round(inference_time, 3),
            'timestamp': datetime.now().isoformat(),
            'image_shape': list(image.shape),
            'weapons_found': len(detections) > 0
        })

    except Exception as e:
        logger.error(f"[{client_id}] Request error: {e}")
        return jsonify({'error': f"Failed: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    try:
        stats = detection_server.get_all_stats()
        commands = detection_server.get_alarm_commands()
        return jsonify({
            'status': 'running',
            'clients': stats,
            'current_alarms': commands,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': detection_server.model is not None,
        'timestamp': datetime.now().isoformat()
    })

# Background task for cleanup
def cleanup_task():
    while True:
        detection_server.update_detection_states()
        time.sleep(1)

if __name__ == '__main__':
    if detection_server.model is None:
        logger.error("Failed to load model.")
        exit(1)

    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()

    logger.info("Starting multi-client weapon detection server with WebSocket support on port 8000...")
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)