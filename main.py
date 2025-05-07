# main.py
import asyncio
import platform
from gui.gui_app import EmotionApp

if __name__ == '__main__':
    app = EmotionApp('emotion_model.h5')
    if platform.system() == "Emscripten":
        asyncio.ensure_future(app.run())
    else:
        asyncio.run(app.run())