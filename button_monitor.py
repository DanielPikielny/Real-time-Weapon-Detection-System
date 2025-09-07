#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
import subprocess
import signal
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BUTTON_PIN = 2  # GPIO 2 (physical pin 3) - has hardware pull-up
VENV_PATH = "???????????????"  # Insert virtual environment path
SCRIPT_PATH = "???????????????"  # Path to your script, adjust as needed
DEBOUNCE_TIME = 0.2  # Seconds to wait between button presses

class ScriptToggler:
    def __init__(self):
        self.process = None
        self.setup_gpio()
        self.setup_signal_handlers()
        
    def setup_gpio(self):
        try:
            GPIO.setmode(GPIO.BCM)
            # Don't set pull-up since GPIO 2 has hardware pull-up
            GPIO.setup(BUTTON_PIN, GPIO.IN)  # No pull_up_down parameter
            GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, 
                                callback=self.button_pressed, 
                                bouncetime=int(DEBOUNCE_TIME * 1000))
            logging.info(f"GPIO setup complete on pin {BUTTON_PIN} (using hardware pull-up)")
        except Exception as e:
            logging.error(f"GPIO setup failed: {e}")
            raise
    
    def setup_signal_handlers(self):
        signal.signal(signal.SIGTERM, self.cleanup)
        signal.signal(signal.SIGINT, self.cleanup)
    
    def button_pressed(self, channel):
        logging.info("Button pressed!")
        if self.is_script_running():
            self.stop_script()
        else:
            self.start_script()
    
    def is_script_running(self):
        return self.process is not None and self.process.poll() is None
    
    def start_script(self):
        if self.is_script_running():
            logging.info("Script is already running")
            return
            
        try:
            # Check if venv exists
            python_path = f"{VENV_PATH}/bin/python"
            if not os.path.exists(python_path):
                logging.error(f"Virtual environment python not found at: {python_path}")
                return
                
            # Check if script exists
            if not os.path.exists(SCRIPT_PATH):
                logging.error(f"Script not found at: {SCRIPT_PATH}")
                return
            
            logging.info(f"Starting script: {SCRIPT_PATH}")
            logging.info(f"Using python: {python_path}")
            
            # Start the script with venv python
            self.process = subprocess.Popen(
                [python_path, SCRIPT_PATH],
                preexec_fn=os.setsid,  # Create new process group for clean termination
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logging.info(f"Started script with PID: {self.process.pid}")
            
        except Exception as e:
            logging.error(f"Error starting script: {e}")
    
    def stop_script(self):
        if not self.is_script_running():
            logging.info("Script is not running")
            return
            
        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for process to terminate
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                logging.warning("Process didn't terminate gracefully, force killing...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                
            logging.info("Script stopped")
            self.process = None
        except Exception as e:
            logging.error(f"Error stopping script: {e}")
    
    def cleanup(self, signum=None, frame=None):
        logging.info("Cleaning up...")
        if self.is_script_running():
            self.stop_script()
        GPIO.cleanup()
        sys.exit(0)
    
    def run(self):
        logging.info("Button monitor started. Press Ctrl+C to exit.")
        logging.info(f"Monitoring button on GPIO {BUTTON_PIN} (hardware pull-up)")
        logging.info(f"Virtual environment: {VENV_PATH}")
        logging.info(f"Script to run: {SCRIPT_PATH}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

if __name__ == "__main__":
    toggler = ScriptToggler()
    toggler.run()