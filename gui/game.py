import random
import pygame
import os
from PIL import Image
import io

class EmotionGame:
    def __init__(self, emotions, app):
        self.emotions = emotions
        self.app = app  # Tham chiếu đến EmotionApp để truy cập window
        self.target_emotion = None
        self.sound_played = False

    def start_game(self, _):
        """Khởi động trò chơi: chọn cảm xúc mục tiêu và hiển thị icon"""
        self.target_emotion = random.choice(self.emotions)
        self.update_target_icon()
        self.play_instruction_sound(_)

    def next_level(self, _):
        """Chuyển sang cấp độ tiếp theo: chọn cảm xúc mới và cập nhật icon"""
        self.target_emotion = random.choice(self.emotions)
        self.update_target_icon()
        self.play_instruction_sound(_)

    def update_target_icon(self):
        """Cập nhật icon cảm xúc mục tiêu trên giao diện"""
        if '-TARGET_ICON-' not in self.app.window.AllKeysDict:
            print("Error: -TARGET_ICON- not found in window!")
            return
        icon_path = f'icons/{self.target_emotion}.png'
        if os.path.exists(icon_path):
            icon_data = self.app.resize_icon(icon_path, target_size=(64, 64))
            if icon_data:
                self.app.window['-TARGET_ICON-'].update(data=icon_data)
                print(f"Updated -TARGET_ICON- with {self.target_emotion}")
            else:
                print(f"Failed to resize icon for {self.target_emotion}")
                self.app.window['-TARGET_ICON-'].update(data=None)
        else:
            print(f"Icon file not found: {icon_path}")
            default_icon = 'icons/default.png'
            if os.path.exists(default_icon):
                icon_data = self.app.resize_icon(default_icon, target_size=(64, 64))
                if icon_data:
                    self.app.window['-TARGET_ICON-'].update(data=icon_data)
                    print("Loaded default icon")
                else:
                    print("Failed to resize default icon")
                    self.app.window['-TARGET_ICON-'].update(data=None)
            else:
                print("Default icon not found")
                self.app.window['-TARGET_ICON-'].update(data=None)

    def play_instruction_sound(self, _):
        """Phát âm thanh hướng dẫn cho cảm xúc mục tiêu"""
        if not self.sound_played:
            sound_path = f'sounds/{self.target_emotion}.mp3'
            if os.path.exists(sound_path):
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
                self.sound_played = True