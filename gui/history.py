# gui/history.py
import sqlite3
import PySimpleGUI as sg
import os
import datetime

class EmotionHistory:
    def __init__(self, captures_dir):
        self.captures_dir = captures_dir
        self.db_path = 'emotion_history.db'
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history
                     (timestamp TEXT, emotion TEXT, image_path TEXT)''')
        conn.commit()
        conn.close()

    def save_emotion(self, emotion, image_path):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO history (timestamp, emotion, image_path) VALUES (?, ?, ?)",
                  (timestamp, emotion, image_path))
        conn.commit()
        conn.close()

    def show_history(self, _):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT timestamp, emotion, image_path FROM history")
        rows = c.fetchall()
        conn.close()
        history_layout = [
            [sg.Text(f"{row[0]} - {row[1]}"),
             sg.Image(filename=row[2], size=(100, 100)) if os.path.exists(row[2]) else sg.Text("Image not found")]
            for row in rows
        ]
        history_layout.append([sg.Button(_('Close'), key='-CLOSE-')])
        history_window = sg.Window(_('Emotion History'), history_layout, modal=True)
        while True:
            event, _values = history_window.read()
            if event in (sg.WIN_CLOSED, '-CLOSE-'):
                break
        history_window.close()