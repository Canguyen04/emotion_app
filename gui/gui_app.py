import PySimpleGUI as sg
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import datetime
from PIL import Image
import io
import matplotlib.pyplot as plt
import pygame
import asyncio
import platform
import time
import threading

# Sử dụng nhập tương đối
from .history import EmotionHistory
from .email_sender import EmailSender
from .audio_analyzer import AudioAnalyzer
from .game import EmotionGame
from .settings import Settings


class EmotionApp:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_colors = {
            'angry': (0, 0, 255), 'disgust': (0, 255, 0), 'fear': (255, 0, 0),
            'happy': (0, 255, 255), 'sad': (255, 255, 0), 'surprise': (255, 0, 255),
            'neutral': (128, 128, 128)
        }
        self.current_frame = None
        self.selected_image = None
        self.captures_dir = 'captures'
        if not os.path.exists(self.captures_dir):
            os.makedirs(self.captures_dir)
        pygame.mixer.init()
        self.negative_emotion_timer = 0
        self.negative_emotion_threshold = 10  # seconds
        self.lang = 'en'  # Ngôn ngữ mặc định
        self.performance_mode = 'normal'
        self.latest_emotion = None

        # Biến trạng thái trò chơi
        self.game_running = False
        self.game_score = 0
        self.game_time_left = 10  # Thời gian mỗi vòng (10 giây)
        self.last_game_update = time.time()

        # Biến để quản lý webcam thread
        self.webcam_thread = None
        self.webcam_running = False
        self.webcam_frame = None
        self.webcam_lock = threading.Lock()

        # Placeholder image
        self.placeholder_image = np.zeros((360, 480, 3), dtype=np.uint8)
        self.placeholder_image_bytes = cv2.imencode('.png', self.placeholder_image)[1].tobytes()

        # Từ điển bản dịch
        self.translations = {
            'en': {
                'Emotion Recognition': 'Emotion Recognition',
                'Emotion:': 'Emotion:',
                'Probabilities:': 'Probabilities:',
                'Load Image': 'Load Image',
                'Load Video': 'Load Video',
                'Webcam': 'Webcam',
                'Stop Webcam': 'Stop Webcam',
                'Capture': 'Capture',
                'Gallery': 'Gallery',
                'History': 'History',
                'Share': 'Share',
                'Settings': 'Settings',
                'Game': 'Game',
                'Exit': 'Exit',
                'Choose an image': 'Choose an image',
                'Image saved as': 'Image saved as',
                'No images in gallery!': 'No images in gallery!',
                'Captured Images': 'Captured Images',
                'Delete': 'Delete',
                'Close': 'Close',
                'Are you sure you want to delete': 'Are you sure you want to delete',
                'Deleted': 'Deleted',
                'Email sent successfully!': 'Email sent successfully!',
                'Failed to send email:': 'Failed to send email:',
                'Theme': 'Theme',
                'Icon': 'Icon',
                'Language': 'Language',
                'Performance Mode': 'Performance Mode',
                'Save': 'Save',
                'Cancel': 'Cancel',
                'Please express the emotion:': 'Please express the emotion:',
                'Warning: Negative emotion detected for too long!': 'Warning: Negative emotion detected for too long!',
                'Warning': 'Warning',
                'Sender Email:': 'Sender Email:',
                'App Password:': 'App Password:',
                'Recipient Email:': 'Recipient Email:',
                'Send': 'Send',
                'Please fill in all fields!': 'Please fill in all fields!',
                'Invalid sender email! Must be a valid Gmail address.': 'Invalid sender email! Must be a valid Gmail address.',
                'Invalid recipient email!': 'Invalid recipient email!',
                'Invalid App Password! It should be 16 characters with no spaces.': 'Invalid App Password! It should be 16 characters with no spaces.',
                'Target Emotion Icon': 'Target Emotion Icon',
                'Game: Express the emotion': 'Game: Express the emotion',
                'Time left:': 'Time left:',
                'Score:': 'Score:',
                'Exit Game': 'Exit Game',
                'Game Over': 'Game Over',
                'Image not found': 'Image not found',
                'Level Up!': 'Level Up!',
                'Try Again!': 'Try Again!',
                'Current Emotion:': 'Current Emotion:'
            },
            'vi': {
                'Emotion Recognition': 'Nhận diện cảm xúc',
                'Emotion:': 'Cảm xúc:',
                'Probabilities:': 'Xác suất:',
                'Load Image': 'Tải ảnh',
                'Load Video': 'Tải video',
                'Webcam': 'Máy quay',
                'Stop Webcam': 'Tắt máy quay',
                'Capture': 'Chụp ảnh',
                'Gallery': 'Thư viện',
                'History': 'Lịch sử',
                'Share': 'Chia sẻ',
                'Settings': 'Cài đặt',
                'Game': 'Trò chơi',
                'Exit': 'Thoát',
                'Choose an image': 'Chọn một ảnh',
                'Image saved as': 'Ảnh đã được lưu dưới tên',
                'No images in gallery!': 'Không có ảnh trong thư viện!',
                'Captured Images': 'Ảnh đã chụp',
                'Delete': 'Xóa',
                'Close': 'Đóng',
                'Are you sure you want to delete': 'Bạn có chắc chắn muốn xóa',
                'Deleted': 'Đã xóa',
                'Email sent successfully!': 'Email đã được gửi thành công!',
                'Failed to send email:': 'Gửi email thất bại:',
                'Theme': 'Chủ đề',
                'Icon': 'Biểu tượng',
                'Language': 'Ngôn ngữ',
                'Performance Mode': 'Chế độ hiệu suất',
                'Save': 'Lưu',
                'Cancel': 'Hủy',
                'Please express the emotion:': 'Vui lòng thể hiện cảm xúc:',
                'Warning: Negative emotion detected for too long!': 'Cảnh báo: Cảm xúc tiêu cực được phát hiện quá lâu!',
                'Warning': 'Cảnh báo',
                'Sender Email:': 'Email người gửi:',
                'App Password:': 'Mật khẩu ứng dụng:',
                'Recipient Email:': 'Email người nhận:',
                'Send': 'Gửi',
                'Please fill in all fields!': 'Vui lòng điền đầy đủ tất cả các trường!',
                'Invalid sender email! Must be a valid Gmail address.': 'Email người gửi không hợp lệ! Phải là địa chỉ Gmail hợp lệ.',
                'Invalid recipient email!': 'Email người nhận không hợp lệ!',
                'Invalid App Password! It should be 16 characters with no spaces.': 'Mật khẩu ứng dụng không hợp lệ! Phải có 16 ký tự và không chứa khoảng trắng.',
                'Target Emotion Icon': 'Biểu tượng cảm xúc mục tiêu',
                'Game: Express the emotion': 'Trò chơi: Thể hiện cảm xúc',
                'Time left:': 'Thời gian còn lại:',
                'Score:': 'Điểm:',
                'Exit Game': 'Thoát trò chơi',
                'Game Over': 'Kết thúc trò chơi',
                'Image not found': 'Không tìm thấy ảnh',
                'Level Up!': 'Lên cấp!',
                'Try Again!': 'Thử lại!',
                'Current Emotion:': 'Cảm xúc hiện tại:'
            }
        }

        self._ = lambda s: self.translations[self.lang].get(s, s)

        sg.theme('DarkBlue3')
        self.layout = [
            [sg.Text(self._('Emotion Recognition'), font=('Helvetica', 20, 'bold'), justification='center',
                     key='-TITLE-', text_color='white', background_color='#2E3B4E', pad=(10, 10))],
            [
                sg.Column([
                    [sg.Image(key='-IMAGE-', size=(480, 360), background_color='black', pad=(10, 10))],
                ], key='-WEB_CAM_COLUMN-', element_justification='center', background_color='#2E3B4E', pad=(10, 10),
                    size=(500, 400)),
                sg.Column([
                    [sg.Image(key='-EMOTION_ICON-', size=(64, 64), pad=(10, 10))],
                    [sg.Image(key='-PIE_CHART-', size=(200, 200), pad=(10, 10))]
                ], element_justification='center', background_color='#2E3B4E', pad=(10, 10))
            ],
            [
                sg.Column([
                    [sg.Text(self._('Emotion:'), key='-EMOTION_TEXT-', font=('Helvetica', 12), text_color='white',
                             background_color='#2E3B4E', pad=(5, 5)),
                     sg.Text('', key='-EMOTION-', font=('Helvetica', 12, 'bold'), text_color='#FFD700',
                             background_color='#2E3B4E', pad=(5, 5))],
                    [sg.Text(self._('Probabilities:'), key='-PROBS_TEXT-', font=('Helvetica', 10), text_color='white',
                             background_color='#2E3B4E', pad=(5, 5)),
                     sg.Text('', key='-PROBS-', font=('Helvetica', 10), text_color='#FFD700',
                             background_color='#2E3B4E', pad=(5, 5))]
                ], element_justification='center', background_color='#2E3B4E', pad=(10, 10))
            ],
            [
                sg.Column([
                    [sg.Push(background_color='#2E3B4E'),
                     sg.Button(self._('Load Image'), key='-LOAD_IMAGE-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Load Video'), key='-LOAD_VIDEO-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Webcam'), key='-WEBCAM-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Stop Webcam'), key='-STOP_WEBCAM-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Capture'), key='-CAPTURE-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Gallery'), key='-GALLERY-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('History'), key='-HISTORY-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Share'), key='-SHARE-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Settings'), key='-SETTINGS-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Game'), key='-GAME-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Button(self._('Exit'), key='-EXIT-', font=('Helvetica', 10), pad=(5, 5)),
                     sg.Push(background_color='#2E3B4E')]
                ], key='-BUTTON_ROW-', background_color='#2E3B4E', element_justification='center', pad=(0, 0))
            ]
        ]
        self.window = sg.Window(self._('Emotion Recognition'), self.layout, finalize=True, resizable=True,
                                size=(1024, 768), background_color='#2E3B4E')
        self.video_capture = None
        self.last_window_size = (1024, 768)
        self.history_manager = EmotionHistory(self.captures_dir)
        self.email_sender = EmailSender()
        self.audio_analyzer = AudioAnalyzer()
        self.game = EmotionGame(self.emotions, self)
        self.settings = Settings(self.lang, self.performance_mode)

    def set_language(self, lang):
        self.lang = lang
        self.refresh_gui()

    def refresh_gui(self):
        self.window['-TITLE-'].update(self._('Emotion Recognition'))
        self.window['-EMOTION_TEXT-'].update(value=self._('Emotion:'))
        self.window['-PROBS_TEXT-'].update(value=self._('Probabilities:'))
        self.window.TKroot.title(self._('Emotion Recognition'))
        button_keys = {
            '-LOAD_IMAGE-': 'Load Image',
            '-LOAD_VIDEO-': 'Load Video',
            '-WEBCAM-': 'Webcam',
            '-STOP_WEBCAM-': 'Stop Webcam',
            '-CAPTURE-': 'Capture',
            '-GALLERY-': 'Gallery',
            '-HISTORY-': 'History',
            '-SHARE-': 'Share',
            '-SETTINGS-': 'Settings',
            '-GAME-': 'Game',
            '-EXIT-': 'Exit',
            '-EXIT_GAME-': 'Exit Game'
        }
        for key, text in button_keys.items():
            if key in self.window.AllKeysDict:
                self.window[key].update(text=self._(text))
        self.window.refresh()

    def reset_gui(self):
        self.current_frame = None
        self.window['-IMAGE-'].update(data=self.placeholder_image_bytes)
        self.window['-EMOTION_ICON-'].update(filename='')
        self.window['-EMOTION-'].update('')
        self.window['-PROBS-'].update('')
        if '-PIE_CHART-' in self.window.AllKeysDict:
            self.window['-PIE_CHART-'].update(data=None)
        self.negative_emotion_timer = 0
        self.latest_emotion = None

    def preprocess_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        img = img_to_array(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def resize_icon(self, icon_path, target_size):
        try:
            with Image.open(icon_path) as img:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                with io.BytesIO() as output:
                    img.save(output, format="PNG")
                    return output.getvalue()
        except Exception as e:
            print(f"Error resizing icon {icon_path}: {e}")
            return None

    def create_pie_chart(self, probs, target_size):
        labels = [k for k, v in probs.items() if float(v[:-1]) > 0]
        sizes = [float(v[:-1]) for k, v in probs.items() if float(v[:-1]) > 0]
        if not sizes:
            return None
        fig, ax = plt.subplots(figsize=(target_size[0] / 100, target_size[1] / 100))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.read()

    def play_sound(self, emotion):
        sound_path = f'sounds/{emotion}.mp3'
        if os.path.exists(sound_path):
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()

    def play_game_sound(self, sound_type):
        sound_path = f'sounds/{sound_type}.mp3'
        if os.path.exists(sound_path):
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()

    def process_frame(self, frame, image_key, window):
        if self.performance_mode == 'light':
            frame = cv2.resize(frame, (320, 240))
        self.current_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        emotions_detected = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_processed = self.preprocess_image(face)
                prediction = self.model.predict(face_processed)[0]
                emotion = self.emotions[np.argmax(prediction)]
                emotions_detected.append(emotion)
                color = self.emotion_colors.get(emotion, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                self.play_sound(emotion)
            if emotions_detected:
                dominant_emotion = max(set(emotions_detected), key=emotions_detected.count)
                self.latest_emotion = dominant_emotion
                if dominant_emotion in ['sad', 'angry', 'fear']:
                    self.negative_emotion_timer += 1
                    if self.negative_emotion_timer > self.negative_emotion_threshold:
                        sg.popup(self._('Warning: Negative emotion detected for too long!'), title=self._('Warning'))
                        self.negative_emotion_timer = 0
                else:
                    self.negative_emotion_timer = 0
                probs = {self.emotions[i]: f"{prediction[i] * 100:.1f}%" for i in range(len(self.emotions))}
                probs_str = ', '.join([f"{k}: {v}" for k, v in probs.items()])

                # Kiểm tra xem đây là cửa sổ chính hay cửa sổ game
                is_game_window = '-GAME_IMAGE-' in window.AllKeysDict
                if is_game_window:
                    # Cập nhật cảm xúc hiện tại trong cửa sổ game
                    if '-CURRENT_EMOTION-' in window.AllKeysDict:
                        window['-CURRENT_EMOTION-'].update(dominant_emotion)
                else:
                    # Cập nhật giao diện cửa sổ chính
                    icon_path = f'icons/{dominant_emotion}.png'
                    if os.path.exists(icon_path):
                        icon_data = self.resize_icon(icon_path, target_size=(64, 64))
                        if icon_data and '-EMOTION_ICON-' in window.AllKeysDict:
                            window['-EMOTION_ICON-'].update(data=icon_data)
                    if '-EMOTION-' in window.AllKeysDict:
                        window['-EMOTION-'].update(dominant_emotion)
                    if '-PROBS-' in window.AllKeysDict:
                        window['-PROBS-'].update(probs_str)
                    pie_chart_data = self.create_pie_chart(probs, (200, 200))
                    if pie_chart_data and '-PIE_CHART-' in window.AllKeysDict:
                        window['-PIE_CHART-'].update(data=pie_chart_data)
        else:
            self.latest_emotion = None
            # Đặt lại các giá trị trong cửa sổ tương ứng
            is_game_window = '-GAME_IMAGE-' in window.AllKeysDict
            if is_game_window:
                if '-CURRENT_EMOTION-' in window.AllKeysDict:
                    window['-CURRENT_EMOTION-'].update('')
            else:
                if '-EMOTION_ICON-' in window.AllKeysDict:
                    window['-EMOTION_ICON-'].update(filename='')
                if '-EMOTION-' in window.AllKeysDict:
                    window['-EMOTION-'].update('')
                if '-PROBS-' in window.AllKeysDict:
                    window['-PROBS-'].update('')
                if '-PIE_CHART-' in window.AllKeysDict:
                    window['-PIE_CHART-'].update(data=None)

        # Cập nhật hình ảnh webcam
        window_size = window[image_key].get_size()
        if window_size[0] > 0 and window_size[1] > 0:
            img_display = cv2.resize(frame, window_size)
            imgbytes = cv2.imencode('.png', img_display)[1].tobytes()
            window[image_key].update(data=imgbytes)

    def run_game(self):
        # Kiểm tra và khởi động webcam nếu chưa chạy
        if not self.webcam_running:
            if not self.video_capture:
                self.video_capture = cv2.VideoCapture(0)
            self.webcam_running = True
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()

        self.game_running = True
        self.game_score = 0
        self.game_time_left = 10
        self.last_game_update = time.time()

        # Khởi động trò chơi
        self.game.start_game(self._)

        # Tải icon cảm xúc mục tiêu
        target_icon_path = f'icons/{self.game.target_emotion}.png'
        target_icon_data = None
        if os.path.exists(target_icon_path):
            target_icon_data = self.resize_icon(target_icon_path, target_size=(150, 150))
        if not target_icon_data:
            target_icon_data = self.placeholder_image_bytes

        # Tạo giao diện game trong cửa sổ mới
        game_layout = [
            [sg.Text(self._('Game: Express the emotion'), font=('Helvetica', 20, 'bold'), text_color='#FFD700',
                     background_color='#1E2A44', justification='center', pad=(10, 10), relief=sg.RELIEF_RIDGE)],
            [sg.Image(key='-GAME_IMAGE-', size=(480, 360), background_color='black', pad=(10, 10))],
            [sg.Text(self._('Current Emotion:'), font=('Helvetica', 16), text_color='white',
                     background_color='#1E2A44', pad=(10, 5)),
             sg.Text('', key='-CURRENT_EMOTION-', font=('Helvetica', 16, 'bold'), text_color='#FFD700',
                     background_color='#1E2A44', pad=(5, 5))],
            [sg.Image(data=target_icon_data, key='-TARGET_ICON-', size=(150, 150), pad=(10, 10),
                      background_color='#1E2A44')],
            [sg.Text(self._('Please express the emotion:'), font=('Helvetica', 16), text_color='white',
                     background_color='#1E2A44', pad=(10, 5)),
             sg.Text(self.game.target_emotion, key='-TARGET_EMOTION-', font=('Helvetica', 16, 'bold'),
                     text_color='#00FFFF',
                     background_color='#1E2A44', pad=(5, 5))],
            [sg.Text(self._('Time left:'), font=('Helvetica', 16), text_color='white', background_color='#1E2A44',
                     pad=(10, 5)),
             sg.Text(str(self.game_time_left), key='-TIME_LEFT-', font=('Helvetica', 16, 'bold'), text_color='#FF4500',
                     background_color='#1E2A44', pad=(5, 5))],
            [sg.Text(self._('Score:'), font=('Helvetica', 16), text_color='white', background_color='#1E2A44',
                     pad=(10, 5)),
             sg.Text(str(self.game_score), key='-SCORE-', font=('Helvetica', 16, 'bold'), text_color='#32CD32',
                     background_color='#1E2A44', pad=(5, 5))],
            [sg.Button(self._('Exit Game'), key='-EXIT_GAME-', font=('Helvetica', 14),
                       button_color=('white', '#FF6347'),
                       pad=(10, 10), border_width=2)]
        ]

        game_window = sg.Window(self._('Game: Express the emotion'), game_layout, finalize=True, resizable=False,
                                background_color='#1E2A44', element_justification='center', size=(600, 800))

        # Ẩn nút game trên cửa sổ chính
        self.window['-GAME-'].update(visible=False)
        self.window.refresh()

        while self.game_running:
            event, values = game_window.read(timeout=100)
            if event in (sg.WIN_CLOSED, '-EXIT_GAME-'):
                self.game_running = False
                break

            # Cập nhật thời gian
            current_time = time.time()
            if current_time - self.last_game_update >= 1:
                self.game_time_left -= 1
                self.last_game_update = current_time
                time_color = '#FF4500' if self.game_time_left > 3 else '#FF0000'
                game_window['-TIME_LEFT-'].update(value=str(self.game_time_left), text_color=time_color)

                if self.game_time_left <= 0:
                    self.play_game_sound('game_over')
                    sg.popup(f"{self._('Game Over')}\n{self._('Score:')} {self.game_score}", title=self._('Game Over'),
                             font=('Helvetica', 12), text_color='white', background_color='#1E2A44')
                    self.game_running = False
                    break

            # Xử lý frame và nhận diện cảm xúc
            if self.webcam_running and self.webcam_frame is not None:
                with self.webcam_lock:
                    frame = self.webcam_frame.copy()
                self.process_frame(frame, '-GAME_IMAGE-', game_window)
                if self.latest_emotion and self.latest_emotion == self.game.target_emotion:
                    self.game_score += 10
                    self.game_time_left = 10
                    self.play_game_sound('level_up')
                    sg.popup(self._('Level Up!'), title=self._('Level Up!'), font=('Helvetica', 12),
                             text_color='white', background_color='#32CD32')
                    self.game.next_level(self._)
                    # Cập nhật icon cảm xúc mới
                    target_icon_path = f'icons/{self.game.target_emotion}.png'
                    target_icon_data = None
                    if os.path.exists(target_icon_path):
                        target_icon_data = self.resize_icon(target_icon_path, target_size=(150, 150))
                    if not target_icon_data:
                        target_icon_data = self.placeholder_image_bytes
                    game_window['-TARGET_ICON-'].update(data=target_icon_data)
                    game_window['-TARGET_EMOTION-'].update(value=self.game.target_emotion)
                    game_window['-SCORE-'].update(value=str(self.game_score))
                    game_window['-TIME_LEFT-'].update(value=str(self.game_time_left), text_color='#FF4500')

        # Đóng cửa sổ game và khôi phục giao diện chính
        game_window.close()
        self.window['-GAME-'].update(visible=True)
        self.window.refresh()

    async def run(self):
        while True:
            event, values = self.window.read(timeout=20)
            if event == sg.WIN_CLOSED or event == '-EXIT-':
                self.webcam_running = False
                if self.webcam_thread:
                    self.webcam_thread.join()
                if self.video_capture:
                    self.video_capture.release()
                pygame.mixer.music.stop()
                break
            if event == '-LOAD_IMAGE-':
                file_path = sg.popup_get_file(self._('Choose an image'), file_types=(('Image Files', '*.jpg *.png'),))
                if file_path:
                    img = cv2.imread(file_path)
                    self.process_frame(img, '-IMAGE-', self.window)
            if event == '-LOAD_VIDEO-':
                video_path = sg.popup_get_file(self._('Choose a video'), file_types=(('Video Files', '*.mp4 *.avi'),))
                if video_path:
                    self.webcam_running = False
                    if self.webcam_thread:
                        self.webcam_thread.join()
                    self.video_capture = cv2.VideoCapture(video_path)
                    self.webcam_running = True
                    self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
                    self.webcam_thread.start()
            if event == '-WEBCAM-':
                if not self.video_capture:
                    self.video_capture = cv2.VideoCapture(0)
                    self.webcam_running = True
                    self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
                    self.webcam_thread.start()
            if event == '-STOP_WEBCAM-':
                self.webcam_running = False
                if self.webcam_thread:
                    self.webcam_thread.join()
                if self.video_capture:
                    self.video_capture.release()
                    self.video_capture = None
                    self.webcam_frame = None
                    self.reset_gui()
            if event == '-CAPTURE-' and self.current_frame is not None:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'capture_{timestamp}.png'
                image_path = os.path.join(self.captures_dir, filename)
                cv2.imwrite(image_path, self.current_frame)
                if self.latest_emotion:
                    self.history_manager.save_emotion(self.latest_emotion, image_path)
                sg.popup(f"{self._('Image saved as')} {filename}")
            if event == '-GALLERY-':
                self.open_gallery()
            if event == '-HISTORY-':
                self.history_manager.show_history(self._)
            if event == '-SHARE-' and self.current_frame is not None:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'capture_{timestamp}.png'
                image_path = os.path.join(self.captures_dir, filename)
                cv2.imwrite(image_path, self.current_frame)
                self.email_sender.send_email(image_path, self._)
            if event == '-SETTINGS-':
                self.performance_mode = self.settings.show_settings(self._, self)
            if event == '-GAME-' and not self.game_running:
                self.run_game()

            if self.webcam_running and self.webcam_frame is not None and not self.game_running:
                with self.webcam_lock:
                    frame = self.webcam_frame.copy()
                self.process_frame(frame, '-IMAGE-', self.window)

        self.webcam_running = False
        if self.webcam_thread:
            self.webcam_thread.join()
        pygame.mixer.quit()
        self.window.close()

    def webcam_loop(self):
        while self.webcam_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                with self.webcam_lock:
                    self.webcam_frame = frame
            time.sleep(0.03)

    def open_gallery(self):
        captures = [f for f in os.listdir(self.captures_dir) if f.endswith('.png')]
        if not captures:
            sg.popup(self._('No images in gallery!'))
            return

        gallery_layout = []
        target_size = (300, 300)
        for f in captures:
            image_path = os.path.join(self.captures_dir, f)
            if os.path.exists(image_path):
                image_data = self.resize_icon(image_path, target_size)
                if image_data:
                    gallery_layout.append([
                        sg.Image(data=image_data, key=f'-PREVIEW-{f}', enable_events=True, pad=(5, 5)),
                        sg.Button(self._('Delete'), key=f'-DELETE-{f}', pad=(5, 5))
                    ])
                else:
                    gallery_layout.append([
                        sg.Text(self._('Image not found'), key=f'-PREVIEW-{f}', pad=(5, 5)),
                        sg.Button(self._('Delete'), key=f'-DELETE-{f}', pad=(5, 5))
                    ])
            else:
                gallery_layout.append([
                    sg.Text(self._('Image not found'), key=f'-PREVIEW-{f}', pad=(5, 5)),
                    sg.Button(self._('Delete'), key=f'-DELETE-{f}', pad=(5, 5))
                ])

        layout = [
            [sg.Text(self._('Captured Images'), font=('Helvetica', 14), key='-GALLERY_TITLE-'),
             sg.Push(background_color='#2E3B4E'),
             sg.Button(self._('Close'), key='-CLOSE-')],
            [sg.Column(gallery_layout, scrollable=True, vertical_scroll_only=True, size=(800, 600))]
        ]
        gallery_window = sg.Window(self._('Gallery'), layout, modal=True, resizable=True, size=(850, 700))
        while True:
            event, values = gallery_window.read()
            if event == sg.WIN_CLOSED or event == '-CLOSE-':
                break
            if event.startswith('-PREVIEW-'):
                self.selected_image = os.path.join(self.captures_dir, event.split('-PREVIEW-')[1])
                gallery_window.close()
                break
            if event.startswith('-DELETE-'):
                file_to_delete = event.split('-DELETE-')[1]
                confirm = sg.popup_yes_no(f"{self._('Are you sure you want to delete')} {file_to_delete}?")
                if confirm == 'Yes':
                    os.remove(os.path.join(self.captures_dir, file_to_delete))
                    sg.popup(f"{self._('Deleted')} {file_to_delete}")
                    gallery_window.close()
                    self.open_gallery()
                break
        gallery_window.close()