#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import imageio
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

__author__ = 'cnheider'
__doc__ = ''

Config.set('graphics', 'resizable', 0)

Window.size = (600, 600)
Window.clearcolor = (.9, .9, .9, 1)


class MainLayout(BoxLayout):
  _video_capture = None
  _face_cascade = None
  _frame_name = str(PROJECT_APP_PATH.user_cache / 'face_detection_frame.jpg')

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.build()

  def build_dropdown(self):
    dropdown_layout = DropDown()

    for index in range(10):
      # When adding widgets, we need to specify the height manually
      # (disabling the size_hint_y) so the dropdown can calculate
      # the area it needs.

      btn2 = Button(text=f'Model {index:d}', size_hint_y=None, height=20)

      # for each button, attach a callback that will call the select() method
      # on the dropdown. We'll pass the text of the button as the data of the
      # selection.
      btn2.bind(on_release=lambda btn:dropdown_layout.select(btn.text))

      # then add the button inside the dropdown
      dropdown_layout.add_widget(btn2)

    # create a big main button
    self._dropdown_btn = Button(text='Model', size_hint=(.5, .1))

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
    self._selected_model = model
    self._dropdown_btn.text = model
    # setattr(self.dropdown_btn, 'text', model)

  def build(self):

    apply_btn = Button(text="Apply", bold=True)
    apply_btn.bind(on_press=self.settings_process)

    dropdown_btn = self.build_dropdown()

    kv_layout = GridLayout(cols=2)
    kv_layout.add_widget(Label(text="Model: ", bold=True))
    kv_layout.add_widget(dropdown_btn)

    settings_layout = BoxLayout(orientation="vertical")
    settings_layout.add_widget(kv_layout)
    settings_layout.add_widget(apply_btn)

    self._popup = Popup(title='Settings', content=settings_layout, size_hint=(.6, .2))

  def start(self):
    if self.ids.status.text == "Stop":
      self.stop()
    else:
      self.start_cam()

  def start_cam(self):
    self.ids.status.text = "Stop"
    self._video_capture = cv2.VideoCapture(0)
    self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    Clock.schedule_once(self.update)

  def stop(self):
    self.ids.status.text = "Start"
    Clock.unschedule(self.update)
    self._video_capture.release()
    cv2.destroyAllWindows()

  def update(self, dt):
    ret, frame = self._video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
      faces = self._face_cascade.detectMultiScale(gray,
                                                  scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  minSize=(30, 30)
                                                  )

      for (x, y, w, h) in faces:
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    except Exception as e:
      print(e)

    imageio.imsave(self._frame_name, rgb)
    self.ids.image_source.reload()
    Clock.schedule_once(self.update)

  @staticmethod
  def close():
    App.get_running_app().stop()

  def settings(self):
    self._popup.open()

  def settings_process(self, btn):
    try:
      self._current_model = self._selected_model
    except:
      pass
    self._popup.dismiss()


class VideoStreamApp(App):
  '''
  VideoStreamApp
  '''
  layout_kv = f'''
MainLayout:
  BoxLayout:
    orientation: 'vertical'
    padding: root.width * 0.05, root.height * .05
    spacing: '5dp'
    BoxLayout:
      size_hint: [1,.85]
      Image:
        id: image_source
        source: '{MainLayout._frame_name}'
    BoxLayout:
      size_hint: [1,.15]
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
  '''

  def build(self):
    a = Builder.load_string(VideoStreamApp.layout_kv)
    a.start_cam()
    return a


def main():
  VideoStreamApp().run()


if __name__ == '__main__':

  main()
