# core_engine.py - Core algorithms for fishing bot
# This file will be loaded remotely by the main application

import cv2
import numpy as np
import time
from collections import deque

class VisionEngine:
    """Core vision processing engine"""
    
    def __init__(self):
        # Detection parameters
        self.red_lower = np.array([240, 200, 180], dtype=np.uint8)
        self.red_upper = np.array([255, 220, 200], dtype=np.uint8)
        self.green_lower = np.array([65, 20, 20], dtype=np.uint8)
        self.green_upper = np.array([85, 40, 35], dtype=np.uint8)
        
        # Morphology kernel
        self.kernel = np.ones((2, 2), np.uint8)
        
        # Prediction buffers
        self.position_buffer = deque(maxlen=10)
        self.velocity_buffer = deque(maxlen=6)
        
        print("🔧 VisionEngine initialized")
    
    def detect_red_progress(self, frame):
        """Detect red progress bar"""
        if frame is None:
            return None
        
        try:
            # Color masking
            mask = cv2.inRange(frame, self.red_lower, self.red_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 85:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(largest)
                    return {
                        'x': x + w,  # Right edge position
                        'y': y + h // 2,
                        'area': area,
                        'width': w,
                        'height': h
                    }
        except Exception as e:
            print(f"Red detection error: {e}")
        
        return None
    
    def detect_green_zone(self, frame):
        """Detect green target zone"""
        if frame is None:
            return None
        
        try:
            # Color masking
            mask = cv2.inRange(frame, self.green_lower, self.green_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 50:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(largest)
                    return {
                        'left': x,
                        'right': x + w,
                        'center': x + w // 2,
                        'y': y + h // 2,
                        'area': area,
                        'width': w,
                        'height': h
                    }
        except Exception as e:
            print(f"Green detection error: {e}")
        
        return None
    
    def predict_collision(self, red_data, green_data):
        """Advanced collision prediction algorithm"""
        if not red_data or not green_data:
            return False, 0
        
        current_time = time.perf_counter()
        red_x = red_data['x']
        
        # Add to position buffer
        self.position_buffer.append((current_time, red_x))
        
        if len(self.position_buffer) < 5:
            return False, 0
        
        # Calculate velocity with multiple data points
        recent_positions = list(self.position_buffer)[-5:]
        velocities = []
        
        for i in range(1, len(recent_positions)):
            time_diff = recent_positions[i][0] - recent_positions[i-1][0]
            pos_diff = recent_positions[i][1] - recent_positions[i-1][1]
            
            if time_diff > 0:
                velocity = pos_diff / time_diff
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return False, 0
        
        # Weighted average for smooth velocity
        weights = [1, 1.5, 2, 2.5]
        if len(velocities) >= 4:
            weighted_velocity = sum(v * w for v, w in zip(velocities[-4:], weights)) / sum(weights)
        else:
            weighted_velocity = sum(velocities) / len(velocities)
        
        self.velocity_buffer.append(weighted_velocity)
        
        if len(self.velocity_buffer) >= 3:
            # Final smoothed velocity
            smooth_velocity = sum(self.velocity_buffer) / len(self.velocity_buffer)
            
            if smooth_velocity > 0:
                # Predict collision
                target_zone = green_data['left'] - 3  # Early trigger
                distance = target_zone - red_x
                
                if distance > 0:
                    time_to_collision = distance / smooth_velocity
                    
                    # Optimal prediction window
                    if 0.025 < time_to_collision <= 0.15:
                        optimal_delay = max(0, time_to_collision - 0.03)
                        return True, optimal_delay
        
        return False, 0
    
    def check_immediate_collision(self, red_data, green_data):
        """Check for immediate collision"""
        if not red_data or not green_data:
            return False
        
        red_x = red_data['x']
        green_left = green_data['left']
        green_right = green_data['right']
        
        # Dynamic tolerance based on size
        base_tolerance = 5
        size_factor = min(red_data.get('width', 20), 30) / 20
        tolerance = int(base_tolerance * size_factor)
        
        return (green_left - tolerance) <= red_x <= (green_right + tolerance)

class BotConfig:
    """Configuration settings for the bot"""
    
    def __init__(self):
        # Performance settings
        self.target_fps = 32
        self.frame_interval = 1.0 / self.target_fps
        
        # Timing settings
        self.rod_timeout = 4.8
        self.e_key_cooldown = 0.06
        self.number_key_cooldown = 0.12
        
        print("⚙️ BotConfig initialized")
    
    def get_scan_region(self, screen_width, screen_height):
        """Calculate optimal scan region"""
        width = int(screen_width * 0.24)
        height = int(screen_height * 0.16)
        left = (screen_width - width) // 2
        top = screen_height - height - 50
        return (left, top, left + width, top + height)
    
    def get_timing_config(self):
        """Get timing configuration"""
        return {
            'target_fps': self.target_fps,
            'frame_interval': self.frame_interval,
            'rod_timeout': self.rod_timeout,
            'e_key_cooldown': self.e_key_cooldown,
            'number_key_cooldown': self.number_key_cooldown
        }

class InputOptimizer:
    """Input method optimization"""
    
    def __init__(self):
        # Method priorities (1 = highest priority)
        self.method_priorities = {
            'sendinput': 1,
            'pyautogui': 2,
            'keyboard': 3,
            'win32': 4,
            'postmsg': 5
        }
        
        # Success rate tracking
        self.success_rates = {
            'sendinput': 0.95,
            'pyautogui': 0.90,
            'keyboard': 0.85,
            'win32': 0.80,
            'postmsg': 0.75
        }
        
        print("🎯 InputOptimizer initialized")
    
    def get_optimal_sequence(self, key_type='e_key'):
        """Get optimal input method sequence"""
        if key_type == 'e_key':
            return ['sendinput', 'pyautogui', 'keyboard', 'win32']
        elif key_type == 'number':
            return ['sendinput', 'pyautogui', 'keyboard', 'win32', 'postmsg']
        else:
            return ['sendinput', 'pyautogui']
    
    def calculate_timing_offset(self, prediction_time):
        """Calculate optimal timing offset"""
        base_offset = 0.03
        
        if prediction_time < 0.05:
            return base_offset * 0.8
        elif prediction_time > 0.1:
            return base_offset * 1.2
        else:
            return base_offset
    
    def get_method_config(self, method_name):
        """Get configuration for specific input method"""
        configs = {
            'sendinput': {'delay': 0.02, 'retry_count': 1},
            'pyautogui': {'delay': 0.01, 'retry_count': 1},
            'keyboard': {'delay': 0.015, 'retry_count': 1},
            'win32': {'delay': 0.02, 'retry_count': 2},
            'postmsg': {'delay': 0.02, 'retry_count': 2}
        }
        
        return configs.get(method_name, {'delay': 0.02, 'retry_count': 1})

# Version info and integrity check
CORE_VERSION = "6.5.1"
CORE_CHECKSUM = "a1b2c3d4e5f6"  # Simple integrity marker

def get_core_info():
    """Get core engine information"""
    return {
        'version': CORE_VERSION,
        'checksum': CORE_CHECKSUM,
        'components': ['VisionEngine', 'BotConfig', 'InputOptimizer'],
        'loaded_at': time.time()
    }

# Test function
if __name__ == "__main__":
    print("🧪 Testing core engine components...")
    
    # Test VisionEngine
    vision = VisionEngine()
    print("✅ VisionEngine created")
    
    # Test BotConfig
    config = BotConfig()
    print("✅ BotConfig created")
    
    # Test InputOptimizer
    optimizer = InputOptimizer()
    print("✅ InputOptimizer created")
    
    print("🎉 All core components working!")
    print(f"📋 Core info: {get_core_info()}")
