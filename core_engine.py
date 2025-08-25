# core_engine.py - File này sẽ được upload lên GitHub Raw
import cv2
import numpy as np
import time
from collections import deque

class VisionEngine:
    def __init__(self):
        # Core detection algorithms
        self.red_lower = np.array([240, 200, 180], dtype=np.uint8)
        self.red_upper = np.array([255, 220, 200], dtype=np.uint8)
        self.green_lower = np.array([65, 20, 20], dtype=np.uint8)
        self.green_upper = np.array([85, 40, 35], dtype=np.uint8)
        self.kernel = np.ones((2, 2), np.uint8)
        self.position_buffer = deque(maxlen=10)
        self.velocity_buffer = deque(maxlen=6)

    def detect_red_progress(self, frame):
        """Core red detection algorithm"""
        if frame is None:
            return None
        
        try:
            mask = cv2.inRange(frame, self.red_lower, self.red_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 85:
                    x, y, w, h = cv2.boundingRect(largest)
                    return {
                        'x': x + w,
                        'y': y + h // 2,
                        'area': area,
                        'width': w,
                        'height': h
                    }
        except:
            pass
        return None

    def detect_green_zone(self, frame):
        """Core green detection algorithm"""
        if frame is None:
            return None
        
        try:
            mask = cv2.inRange(frame, self.green_lower, self.green_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 50:
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
        except:
            pass
        return None

    def predict_collision(self, red_data, green_data):
        """Advanced collision prediction"""
        if not red_data or not green_data:
            return False, 0
        
        current_time = time.perf_counter()
        red_x = red_data['x']
        
        self.position_buffer.append((current_time, red_x))
        
        if len(self.position_buffer) < 5:
            return False, 0
        
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
        
        weights = [1, 1.5, 2, 2.5]
        if len(velocities) >= 4:
            weighted_velocity = sum(v * w for v, w in zip(velocities[-4:], weights)) / sum(weights)
        else:
            weighted_velocity = sum(velocities) / len(velocities)
        
        self.velocity_buffer.append(weighted_velocity)
        
        if len(self.velocity_buffer) >= 3:
            smooth_velocity = sum(self.velocity_buffer) / len(self.velocity_buffer)
            
            if smooth_velocity > 0:
                target_zone = green_data['left'] - 3
                distance = target_zone - red_x
                
                if distance > 0:
                    time_to_collision = distance / smooth_velocity
                    
                    if 0.025 < time_to_collision <= 0.15:
                        optimal_delay = max(0, time_to_collision - 0.03)
                        return True, optimal_delay
        
        return False, 0

# Configuration class
class BotConfig:
    def __init__(self):
        self.target_fps = 32
        self.frame_interval = 1.0 / self.target_fps
        self.rod_timeout = 4.8
        self.e_key_cooldown = 0.06
        self.number_key_cooldown = 0.12
        
    def get_scan_region(self, screen_width, screen_height):
        width = int(screen_width * 0.24)
        height = int(screen_height * 0.16)
        left = (screen_width - width) // 2
        top = screen_height - height - 50
        return (left, top, left + width, top + height)

# Input optimization algorithms
class InputOptimizer:
    def __init__(self):
        self.method_priorities = {
            'sendinput': 1,
            'pyautogui': 2,
            'keyboard': 3,
            'win32': 4,
            'postmsg': 5
        }
    
    def optimize_input_sequence(self, key_type):
        """Optimize input method sequence based on success rates"""
        if key_type == 'e_key':
            return ['sendinput', 'pyautogui', 'keyboard', 'win32']
        elif key_type == 'number':
            return ['sendinput', 'pyautogui', 'keyboard', 'win32', 'postmsg']
        return ['sendinput']
    
    def calculate_timing_offset(self, prediction_time):
        """Calculate optimal timing offset for input"""
        base_offset = 0.03
        if prediction_time < 0.05:
            return base_offset * 0.8
        elif prediction_time > 0.1:
            return base_offset * 1.2
        return base_offset
