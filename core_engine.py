# ═══════════════════════════════════════════════════════════════════
# TURTLE AUTO FISHING - CORE SYSTEM
# Version: 6.5.2 - FIXED DOUBLE PRESS
# Author: FSERVICE808
# ═══════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import pyautogui
import time
from PIL import ImageGrab, Image, ImageTk
import threading
import keyboard
from datetime import datetime
import win32api
import win32con
import win32gui
import ctypes
from ctypes import wintypes
import os
from collections import deque

# Global lock để tránh double execution
_execution_lock = threading.Lock()
_is_executing = False

class UltraReliableInputManager:
    """Ultra-reliable input manager với multiple redundant methods"""
    
    def __init__(self):
        # Timing tối ưu để tránh miss - TĂNG COOLDOWN ĐỂ TRÁNH DOUBLE PRESS
        self.e_key_cooldown = 0.25  # Tăng từ 0.06 lên 0.25
        self.number_key_cooldown = 0.5  # Tăng từ 0.12 lên 0.5
        
        self.last_e_time = 0
        self.last_number_time = 0
        
        # Pre-compile structures
        self.setup_input_structures()
        
        # Success tracking
        self.method_stats = {
            'e_key': {'sendinput': 0, 'pyautogui': 0, 'keyboard': 0, 'win32': 0},
            'number': {'sendinput': 0, 'pyautogui': 0, 'keyboard': 0, 'win32': 0, 'postmsg': 0}
        }
        
        # Lock để tránh concurrent access
        self.input_lock = threading.Lock()
    
    def setup_input_structures(self):
        """Pre-compile input structures"""
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]
        
        class INPUT(ctypes.Structure):
            class _INPUT(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT)]
            _anonymous_ = ("_input",)
            _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT)]
        
        self.KEYBDINPUT = KEYBDINPUT
        self.INPUT = INPUT
        
        # Pre-compile E key
        VK_E = 0x45
        self.e_key_down = INPUT(type=1, ki=KEYBDINPUT(wVk=VK_E, dwFlags=0))
        self.e_key_up = INPUT(type=1, ki=KEYBDINPUT(wVk=VK_E, dwFlags=0x0002))
        self.e_inputs = (INPUT * 2)(self.e_key_down, self.e_key_up)
        
        # Pre-compile number keys
        self.number_inputs = {}
        VK_CODES = {1: 0x31, 2: 0x32, 3: 0x33, 4: 0x34, 5: 0x35}
        for num, vk in VK_CODES.items():
            key_down = INPUT(type=1, ki=KEYBDINPUT(wVk=vk, dwFlags=0))
            key_up = INPUT(type=1, ki=KEYBDINPUT(wVk=vk, dwFlags=0x0002))
            self.number_inputs[num] = (INPUT * 2)(key_down, key_up)
    
    def ultra_reliable_e_press(self):
        """Ultra-reliable E key press với multiple methods - FIXED DOUBLE PRESS"""
        with self.input_lock:  # Lock để tránh concurrent access
            current_time = time.perf_counter()
            if current_time - self.last_e_time < self.e_key_cooldown:
                return False
            
            self.last_e_time = current_time
            success_count = 0
            
            # CHỈ SỬ DỤNG 1 METHOD DUY NHẤT ĐỂ TRÁNH DOUBLE PRESS
            # Method 1: SendInput (fastest và reliable nhất)
            try:
                result = ctypes.windll.user32.SendInput(2, self.e_inputs, ctypes.sizeof(self.INPUT))
                if result == 2:
                    success_count += 1
                    self.method_stats['e_key']['sendinput'] += 1
                    return True  # Return ngay khi thành công
            except:
                pass
            
            # Fallback methods chỉ khi SendInput fail
            time.sleep(0.05)
            
            # Method 2: PyAutoGUI (fallback)
            try:
                pyautogui.press('e')
                success_count += 1
                self.method_stats['e_key']['pyautogui'] += 1
                return True
            except:
                pass
            
            return success_count > 0
    
    def ultra_reliable_number_press(self, number, game_window=None):
        """Ultra-reliable number press - FIXED DOUBLE PRESS"""
        with self.input_lock:  # Lock để tránh concurrent access
            current_time = time.perf_counter()
            if current_time - self.last_number_time < self.number_key_cooldown:
                return False
            
            self.last_number_time = current_time
            success_count = 0
            number_str = str(number)
            
            # CHỈ SỬ DỤNG 1 METHOD DUY NHẤT
            # Method 1: SendInput
            try:
                if number in self.number_inputs:
                    result = ctypes.windll.user32.SendInput(2, self.number_inputs[number], ctypes.sizeof(self.INPUT))
                    if result == 2:
                        success_count += 1
                        self.method_stats['number']['sendinput'] += 1
                        return True
            except:
                pass
            
            # Fallback
            time.sleep(0.1)
            
            # Method 2: PyAutoGUI
            try:
                pyautogui.press(number_str)
                success_count += 1
                self.method_stats['number']['pyautogui'] += 1
                return True
            except:
                pass
            
            return success_count > 0

class EnhancedVisionProcessor:
    """Enhanced vision processor với improved detection"""
    
    def __init__(self):
        # Optimized colors
        self.red_lower = np.array([240, 200, 180], dtype=np.uint8)
        self.red_upper = np.array([255, 220, 200], dtype=np.uint8)
        self.green_lower = np.array([65, 20, 20], dtype=np.uint8)
        self.green_upper = np.array([85, 40, 35], dtype=np.uint8)
        
        # Morphology kernel
        self.kernel = np.ones((2, 2), np.uint8)
        
        # Enhanced prediction system
        self.position_buffer = deque(maxlen=10)
        self.velocity_buffer = deque(maxlen=6)
        
        # Screen setup
        self.screen_width, self.screen_height = pyautogui.size()
        self.scan_region = self.calculate_scan_region()
        
        # Performance stats
        self.detection_stats = {
            'frames_processed': 0,
            'red_detections': 0,
            'green_detections': 0,
            'predictions': 0,
            'immediate_collisions': 0
        }
    
    def calculate_scan_region(self):
        """Calculate optimal scan region"""
        width = int(self.screen_width * 0.24)
        height = int(self.screen_height * 0.16)
        left = (self.screen_width - width) // 2
        top = self.screen_height - height - 50
        return (left, top, left + width, top + height)
    
    def capture_screen_fast(self):
        """Fast screen capture with error handling"""
        try:
            screenshot = ImageGrab.grab(bbox=self.scan_region)
            return np.array(screenshot, dtype=np.uint8)
        except Exception:
            return None
    
    def detect_red_progress_enhanced(self, frame):
        """Enhanced red progress detection"""
        if frame is None:
            return None
        
        try:
            # Color masking
            mask = cv2.inRange(frame, self.red_lower, self.red_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            
            # Contour detection
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 85:
                    x, y, w, h = cv2.boundingRect(largest)
                    self.detection_stats['red_detections'] += 1
                    return {
                        'x': x + w,  # Right edge
                        'y': y + h // 2,
                        'area': area,
                        'width': w,
                        'height': h
                    }
        except Exception:
            pass
        
        return None
    
    def detect_green_zone_enhanced(self, frame):
        """Enhanced green zone detection"""
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
                    self.detection_stats['green_detections'] += 1
                    return {
                        'left': x,
                        'right': x + w,
                        'center': x + w // 2,
                        'y': y + h // 2,
                        'area': area,
                        'width': w,
                        'height': h
                    }
        except Exception:
            pass
        
        return None
    
    def predict_collision_enhanced(self, red_data, green_data):
        """Enhanced collision prediction với better accuracy"""
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
        
        # Smooth velocity với weighted average
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
                # Predict collision với enhanced parameters
                target_zone = green_data['left'] - 3
                distance = target_zone - red_x
                
                if distance > 0:
                    time_to_collision = distance / smooth_velocity
                    
                    # Optimal prediction window
                    if 0.025 < time_to_collision <= 0.15:
                        self.detection_stats['predictions'] += 1
                        # Calculate optimal delay với compensation
                        optimal_delay = max(0, time_to_collision - 0.03)
                        return True, optimal_delay
        
        return False, 0
    
    def check_immediate_collision_enhanced(self, red_data, green_data):
        """Enhanced immediate collision detection"""
        if not red_data or not green_data:
            return False
        
        red_x = red_data['x']
        green_left = green_data['left']
        green_right = green_data['right']
        
        # Enhanced tolerance với size-based adjustment
        base_tolerance = 5
        size_factor = min(red_data.get('width', 20), 30) / 20
        tolerance = int(base_tolerance * size_factor)
        
        collision = (green_left - tolerance) <= red_x <= (green_right + tolerance)
        
        if collision:
            self.detection_stats['immediate_collisions'] += 1
        
        return collision

def find_game_window():
    """Find FiveM game window"""
    def enum_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if any(keyword in window_text.lower() for keyword in ['fivem', 'cfx.re', 'grand theft auto']):
                windows.append((hwnd, window_text))
        return True
    
    windows = []
    win32gui.EnumWindows(enum_callback, windows)
    
    if windows:
        hwnd, title = windows[0]
        return hwnd
    
    return None

def log_message(message):
    """Log message function"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    
    # Try to log to GUI if bot_instance exists
    try:
        if 'bot_instance' in globals():
            bot_instance.log(message)
    except:
        pass

def main_detection_loop():
    """Main detection loop với zero-lag focus - FIXED DOUBLE EXECUTION"""
    global _execution_lock, _is_executing
    
    # Check if already executing
    with _execution_lock:
        if _is_executing:
            log_message("⚠️ Detection loop already running! Skipping...")
            return
        _is_executing = True
    
    try:
        # Initialize components
        input_manager = UltraReliableInputManager()
        vision_processor = EnhancedVisionProcessor()
        game_window = find_game_window()
        
        # Get settings from context
        bot = bot_instance
        
        # Settings
        target_fps = 30  # Giảm xuống để ổn định hơn
        frame_interval = 1.0 / target_fps
        last_minigame_time = time.time()
        rod_timeout = 5.0  # Tăng timeout
        
        # Statistics
        stats = {
            'frames_processed': 0,
            'fish_caught': 0,
            'e_attempts': 0,
            'rod_deploys': 0,
            'predictions_used': 0,
            'immediate_hits': 0,
            'start_time': time.time()
        }
        
        collision_confirmed = False
        consecutive_misses = 0
        last_action_time = 0  # Track last action để tránh spam
        
        log_message(f"🚀 Zero-Lag Detection Started (Target: {target_fps} FPS)")
        
        while bot.is_running:
            loop_start = time.perf_counter()
            current_time = time.time()
            
            try:
                # Capture frame
                frame = vision_processor.capture_screen_fast()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                stats['frames_processed'] += 1
                
                # Detect elements
                red_data = vision_processor.detect_red_progress_enhanced(frame)
                green_data = vision_processor.detect_green_zone_enhanced(frame)
                
                minigame_active = bool(red_data and green_data)
                
                if minigame_active:
                    last_minigame_time = current_time
                    consecutive_misses = 0
                    
                    # Tránh action quá nhanh
                    if current_time - last_action_time < 0.5:
                        time.sleep(0.01)
                        continue
                    
                    # Try predictive collision first
                    should_predict, delay = vision_processor.predict_collision_enhanced(red_data, green_data)
                    
                    if should_predict and not collision_confirmed:
                        stats['predictions_used'] += 1
                        
                        # Apply optimal delay
                        if delay > 0:
                            time.sleep(min(delay, 0.1))
                        
                        # Execute ultra-reliable E-key press
                        if input_manager.ultra_reliable_e_press():
                            stats['e_attempts'] += 1
                            stats['fish_caught'] += 1
                            log_message("⚡ *** FISH CAUGHT - PREDICTIVE AI ***")
                            collision_confirmed = True
                            last_action_time = current_time
                            
                            # Clear buffers
                            vision_processor.position_buffer.clear()
                            vision_processor.velocity_buffer.clear()
                            
                            time.sleep(1.0)  # Longer pause
                            continue
                    
                    # Fallback to immediate collision
                    elif vision_processor.check_immediate_collision_enhanced(red_data, green_data) and not collision_confirmed:
                        if input_manager.ultra_reliable_e_press():
                            stats['e_attempts'] += 1
                            stats['fish_caught'] += 1
                            stats['immediate_hits'] += 1
                            log_message("⚡ *** FISH CAUGHT - IMMEDIATE DETECTION ***")
                            collision_confirmed = True
                            last_action_time = current_time
                            time.sleep(1.0)  # Longer pause
                else:
                    # Reset collision state
                    collision_confirmed = False
                    consecutive_misses += 1
                    
                    # Clear buffers if no minigame for a while
                    if consecutive_misses > 50:
                        vision_processor.position_buffer.clear()
                        vision_processor.velocity_buffer.clear()
                        consecutive_misses = 0
                    
                    # Check rod deployment
                    time_since_minigame = current_time - last_minigame_time
                    time_since_action = current_time - last_action_time
                    
                    if (time_since_minigame >= rod_timeout and 
                        time_since_action >= 2.0 and  # Tránh rod spam
                        bot.auto_rod_enabled):
                        
                        if input_manager.ultra_reliable_number_press(bot.selected_rod_key, game_window):
                            stats['rod_deploys'] += 1
                            log_message(f"🎣 *** ROD DEPLOYED (Key {bot.selected_rod_key}) - ULTRA-RELIABLE ***")
                            last_minigame_time = current_time
                            last_action_time = current_time
                
                # Frame rate control
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                log_message(f"Detection loop error: {e}")
                time.sleep(0.1)
    
    finally:
        # Reset execution flag
        with _execution_lock:
            _is_executing = False
        log_message("🛑 Detection loop stopped")

# Execute main loop
main_detection_loop()
