import cv2
import requests
import numpy as np
import threading
import time
import pygame
import json
from queue import Queue, Empty
import logging
import socketio

# Configuration
VM_IP = "?????????????????????"
VM_PORT = 8000
INFERENCE_URL = f"http://{VM_IP}:{VM_PORT}/infer"
WEBSOCKET_URL = f"http://{VM_IP}:{VM_PORT}"

DEVICE_ID = "jetson1"

API_KEYS = {
    'jetson1': 'secretkey1',
    'jetson2': 'secretkey2'
}

# paths need to be inserted seperately as well as the VM_IP above

ALARM_ME = "????????????" #alarm to play when you recognise a weapon
ALARM_OTHER = "????????????" #alarm to play when the other device recognises a weapon
ALARM_BOTH = "????????????" #alarm to play when both devices recognise a weapon

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeaponDetectionClient:
    def __init__(self):
        self.cap = None
        self.running = False
        self.current_alarm_code = None
        self.alarm_playing = False
        self.websocket_connected = False

        try:
            pygame.mixer.init()
            logger.info("Audio system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")

        self.frame_queue = Queue(maxsize=2)
        self.detection_lock = threading.Lock()

        self.last_inference_time = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()

        # Initialize WebSocket client
        self.sio = socketio.Client(logger=False, engineio_logger=False)
        self.setup_websocket_handlers()

    def setup_websocket_handlers(self):
        @self.sio.event
        def connect():
            logger.info("WebSocket connected to server")
            self.websocket_connected = True
            # Register this client with the server
            self.sio.emit('register_client', {
                'client_id': DEVICE_ID,
                'api_key': API_KEYS[DEVICE_ID]
            })

        @self.sio.event
        def disconnect():
            logger.info("WebSocket disconnected from server")
            self.websocket_connected = False

        @self.sio.event
        def registration_success(data):
            logger.info(f"Successfully registered as {data['client_id']}")

        @self.sio.event
        def registration_failed(data):
            logger.error(f"Registration failed: {data['error']}")

        @self.sio.event
        def alarm_command(data):
            """Handle alarm commands from server"""
            try:
                my_alarm_code = data.get(DEVICE_ID, 'NONE')
                timestamp = data.get('timestamp', '')
                
                logger.info(f"Received alarm command: {my_alarm_code} at {timestamp}")
                
                with self.detection_lock:
                    self.handle_alarm_code(my_alarm_code)
                    
            except Exception as e:
                logger.error(f"Error handling alarm command: {e}")

    def connect_websocket(self):
        """Connect to WebSocket server"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting WebSocket connection (attempt {attempt + 1}/{max_retries})")
                self.sio.connect(WEBSOCKET_URL, wait_timeout=10)
                return True
            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                
        logger.error("Failed to establish WebSocket connection after all retries")
        return False

    def initialize_camera(self, camera_id=0):
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Cannot read from camera")

            logger.info(f"Camera initialized: {frame.shape}")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def play_alarm(self, alarm_path):
        try:
            if not pygame.mixer.get_init():
                return

            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

            pygame.mixer.music.load(alarm_path)
            pygame.mixer.music.play(-1)  # Loop indefinitely
            self.alarm_playing = True
            logger.info(f"Playing alarm: {alarm_path}")

        except Exception as e:
            logger.error(f"Failed to play alarm: {e}")

    def stop_alarm(self):
        try:
            if self.alarm_playing:
                pygame.mixer.music.stop()
                self.alarm_playing = False
                logger.info("Alarm stopped")
        except Exception as e:
            logger.error(f"Failed to stop alarm: {e}")

    def handle_alarm_code(self, alarm_code):
        """Handle alarm code received from server"""
        if alarm_code == self.current_alarm_code:
            return

        logger.info(f"Alarm code changed: {self.current_alarm_code} -> {alarm_code}")
        self.current_alarm_code = alarm_code
        
        if alarm_code == 'ME':
            self.play_alarm(ALARM_ME)
        elif alarm_code == 'OTHER':
            self.play_alarm(ALARM_OTHER)
        elif alarm_code == 'BOTH':
            self.play_alarm(ALARM_BOTH)
        else:  # 'NONE'
            self.stop_alarm()

    def send_frame_for_inference(self, frame):
        try:
            if DEVICE_ID not in API_KEYS:
                logger.error(f"Unknown DEVICE_ID: {DEVICE_ID}")
                return None

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            files = {'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            headers = {'X-API-Key': API_KEYS[DEVICE_ID]}

            response = requests.post(INFERENCE_URL, files=files, headers=headers, timeout=2.0)

            if response.status_code == 200:
                result = response.json()
                return result
            elif response.status_code == 401:
                logger.error("Unauthorized: Invalid API key")
            else:
                logger.warning(f"Unexpected response: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("Inference request timed out")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Inference error: {e}")

        return None

    def inference_worker(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                result = self.send_frame_for_inference(frame)
                if result:
                    self.last_inference_time = time.time()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Inference worker error: {e}")
                time.sleep(0.1)

    def websocket_monitor(self):
        """Monitor WebSocket connection and reconnect if needed"""
        while self.running:
            if not self.websocket_connected:
                logger.info("WebSocket disconnected, attempting to reconnect...")
                self.connect_websocket()
            time.sleep(5)

    def run(self):
        if not self.initialize_camera():
            return

        # Connect to WebSocket server
        if not self.connect_websocket():
            logger.error("Failed to connect to WebSocket server. Exiting.")
            return

        self.running = True
        
        # Start worker threads
        inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        inference_thread.start()
        
        websocket_monitor_thread = threading.Thread(target=self.websocket_monitor, daemon=True)
        websocket_monitor_thread.start()

        logger.info("Starting video stream... Press 'q' to quit")

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                display_frame = frame.copy()

                # Display current alarm status
                status_color = (0, 0, 255) if self.alarm_playing else (0, 255, 0)
                status_text = f"ALARM: {self.current_alarm_code or 'NONE'}"
                cv2.putText(display_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                # Display connection status
                ws_color = (0, 255, 0) if self.websocket_connected else (0, 0, 255)
                ws_status = "WS: CONNECTED" if self.websocket_connected else "WS: DISCONNECTED"
                cv2.putText(display_frame, ws_status, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ws_color, 2)

                # Display device ID
                cv2.putText(display_frame, f"Device: {DEVICE_ID}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # FPS calculation
                self.fps_counter += 1
                current_time = time.time()
                if self.fps_counter % 30 == 0:
                    fps = 30.0 / (current_time - self.fps_start_time)
                    logger.info(f"FPS: {fps:.1f}")
                    self.fps_start_time = current_time

                # Add frame to inference queue
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame.copy())
                    except:
                        pass

                cv2.imshow(f'Weapon Detection - {DEVICE_ID}', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Shutting down...")
        self.running = False
        self.stop_alarm()

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        pygame.mixer.quit()
        
        if self.sio.connected:
            self.sio.disconnect()
        
        logger.info("Cleanup complete")

if __name__ == "__main__":
    client = WeaponDetectionClient()
    client.run() 