# gui/settings.py
import PySimpleGUI as sg
import os

class Settings:
    def __init__(self, lang, performance_mode):
        self.lang = lang
        self.performance_mode = performance_mode

    def show_settings(self, _, gui_app):
        layout = [
            [sg.Text(_('Theme'), key='-THEME_TEXT-'), sg.Combo(['Light', 'Dark'], key='-THEME-', default_value='Dark')],
            [sg.Text(_('Icon'), key='-ICON_TEXT-'), sg.Input(key='-ICON_PATH-'), sg.FileBrowse()],
            [sg.Text(_('Language'), key='-LANG_TEXT-'), sg.Combo(['en', 'vi'], key='-LANG-', default_value=self.lang)],
            [sg.Text(_('Performance Mode'), key='-PERF_TEXT-'), sg.Combo(['normal', 'light'], key='-PERF-', default_value=self.performance_mode)],
            [sg.Button(_('Save'), key='-SAVE-'), sg.Button(_('Cancel'), key='-CANCEL-')]
        ]
        window = sg.Window(_('Settings'), layout, modal=True)
        while True:
            event, values = window.read()
            if event == '-SAVE-':
                sg.theme(values['-THEME-'])
                icon_path = values['-ICON_PATH-']
                if icon_path and os.path.exists(icon_path):
                    gui_app.window['-EMOTION_ICON-'].update(filename=icon_path)
                self.performance_mode = values['-PERF-']
                new_lang = values['-LANG-']
                if new_lang != self.lang:
                    self.lang = new_lang
                    gui_app.set_language(new_lang)
                    window.close()
                    return self.show_settings(_, gui_app)  # Mở lại để làm mới
                break
            elif event in (sg.WIN_CLOSED, '-CANCEL-'):
                break
        window.close()
        return self.performance_mode