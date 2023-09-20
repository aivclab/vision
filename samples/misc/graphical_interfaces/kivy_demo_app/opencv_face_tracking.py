#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import imageio
from functools import partial
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from neodroidvision import PROJECT_APP_PATH

__author__ = "Christian Heider Nielsen"
__doc__ = """ description """

Config.set("graphics", "resizable", 0)

Window.size = (600, 600)
Window.clearcolor = (0.9, 0.9, 0.9, 1)


class MainLayout(BoxLayout):
    """description"""

    _video_capture = None
    _face_cascade = None
    _frame_name = (
        rf'{PROJECT_APP_PATH.user_cache / "face_detection_frame.jpg"}'.replace(
            "\\", "/"
        )
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()

    def build_dropdown(self):
        """

        Returns:

        """
        dropdown_layout = DropDown()

        for index in range(10):
            # When adding widgets, we need to specify the height manually
            # (disabling the size_hint_y) so the dropdown can calculate
            # the area it needs.

            btn2 = Button(text=f"Model {index:d}", size_hint_y=None, height=20)

            # for each button, attach a callback that will call the select() method
            # on the dropdown. We'll pass the text of the button as the data of the
            # selection.
            btn2.bind(on_release=lambda btn: dropdown_layout.select(btn.text))

            # then add the button inside the dropdown
            dropdown_layout.add_widget(btn2)

        # create a big main button
        self._dropdown_btn = Button(text="Model", size_hint=(0.5, 0.1))

        # show the dropdown menu when the main button is released
        # note: all the bind() calls pass the instance of the caller (here, the
        # mainbutton instance) as the first argument of the callback (here,
        # dropdown.open.).
        self._dropdown_btn.bind(on_release=dropdown_layout.open)

        # one last thing, listen for the selection in the dropdown list and
        # assign the data to the button text.
        dropdown_layout.bind(on_select=self.on_select_model)

        return self._dropdown_btn

    def on_select_model(self, ins, model):
        """

        Args:
          ins:
          model:
        """
        self._selected_model = model
        self._dropdown_btn.text = model
        # setattr(self.dropdown_btn, 'text', model)

    def build(self):
        """description"""
        apply_btn = Button(text="Apply", bold=True)
        apply_btn.bind(on_press=self.settings_process)

        dropdown_btn = self.build_dropdown()

        kv_layout = GridLayout(cols=2)
        kv_layout.add_widget(Label(text="Model: ", bold=True))
        kv_layout.add_widget(dropdown_btn)

        settings_layout = BoxLayout(orientation="vertical")
        settings_layout.add_widget(kv_layout)
        settings_layout.add_widget(apply_btn)

        self._popup = Popup(
            title="Settings", content=settings_layout, size_hint=(0.6, 0.2)
        )

    def start(self):
        """description"""
        if self.ids.status.text == "Stop":
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        """description"""
        self.ids.status.text = "Stop"
        self._video_capture = cv2.VideoCapture(0)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        Clock.schedule_once(self.update)

    def stop_stream(self):
        """description"""
        self.ids.status.text = "Start"
        Clock.unschedule(self.update)
        self._video_capture.release()
        cv2.destroyAllWindows()

    def update(self, dt):
        """

        Args:
          dt:
        """
        ret, frame = self._video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for x, y, w, h in faces:
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except Exception as e:
            print(e)

        imageio.imsave(self._frame_name, rgb)
        self.ids.image_source.reload()
        Clock.schedule_once(self.update)

    def close(self):
        """description"""
        self.stop_stream()
        # self.stop()
        App.get_running_app().stop()
        # exit(0)

    def settings(self):
        """description"""
        self._popup.open()

    def settings_process(self, btn):
        """

        Args:
          btn:
        """
        try:
            self._current_model = self._selected_model
        except:
            pass
        self._popup.dismiss()


class VideoStreamApp(App):
    """
    VideoStreamApp"""

    layout_kv = f"""
MainLayout:
  BoxLayout:
    orientation: 'vertical'
    padding: root.width * 0.05, root.height * .05
    spacing: '5dp'
    BoxLayout:
      size_hint: [1, .85]
      Image:
        id: image_source
        source: '{MainLayout._frame_name}'
    BoxLayout:
      size_hint: [1, .15]
      GridLayout:
        cols: 3
        spacing: '10dp'
        Button:
          id: status
          text:'Start'
          bold: True
          background_normal: ''
          background_color: (0.82, 0.82, 0.82, 1.0)
          on_press: root.start()
        Button:
          text: 'Setting'
          bold: True
          background_normal: ''
          background_color: (0.82, 0.82, 0.82, 1.0)
          on_press: root.settings()
        Button:
          text: 'Close'
          bold: True
          background_normal: ''
          background_color: (0.82, 0.82, 0.82, 1.0)
          on_press: root.close()
"""

    def build(self):
        """

        Args:
          self:

        Returns:

        """
        a = Builder.load_string(VideoStreamApp.layout_kv, filename="my_rule.kv")
        # a.bind(on_request_close=self.on_request_close)
        a.start_stream()
        return a

    '''
def on_request_close(self, *args):
self.textpopup(title='Exit', text='Are you sure?')
return True
def textpopup(self, title='', text=''):
"""Open the pop-up with the name.

:param title: title of the pop-up to open
:type title: str
:param text: main text of the pop-up to open
:type text: str
:rtype: None
"""
box = BoxLayout(orientation='vertical')
box.add_widget(Label(text=text))
mybutton = Button(text='OK', size_hint=(1, 0.25))
box.add_widget(mybutton)
popup = Popup(title=title, content=box, size_hint=(None, None), size=(600, 300))
mybutton.bind(on_release=self.stop)
popup.open()
'''

    def stop(self, *largs):
        """

        Args:
          self:
          *largs:
        """
        if False:
            # Open the popup you want to open and declare callback if user pressed `Yes`
            popup = ExitPopup(title="Are you sure?")
            popup.bind(on_confirm=partial(self.close_app, *largs))
            popup.open()
        else:
            self.close_app()

    def close_app(self, *largs):
        """

        Args:
          self:
          *largs:
        """
        super().stop(*largs)


class ExitPopup(Popup):
    """description"""

    def __init__(self, **kwargs):
        super(ExitPopup, self).__init__(**kwargs)
        self.register_event_type("on_confirm")

    def on_confirm(self):
        """description"""
        pass

    def on_button_yes(self):
        """description"""
        self.dispatch("on_confirm")


def main():
    """description"""
    VideoStreamApp().run()


if __name__ == "__main__":
    main()
