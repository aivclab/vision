#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imageio
import torch
from kivy.app import App
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
from neodroidvision.reconstruction.cvae.archs.flat import FlatNormalVAE

__author__ = 'cnheider'
__doc__ = ''

Config.set('graphics', 'resizable', 0)

Window.size = (600, 600)
Window.clearcolor = (.9, .9, .9, 1)

size_x, size_y = (28, 28)

DEVICE = torch.device('cpu')
ENCODING_SIZE = 3

model = FlatNormalVAE(encoding_size=ENCODING_SIZE).to(DEVICE)
checkpoint = torch.load(PROJECT_APP_PATH.user_data / 'results' / 'best_state_dict',
                        map_location=DEVICE)
model.load_state_dict(checkpoint)


class MainLayout(BoxLayout):
  _video_capture = None
  _face_cascade = None
  _frame_name = str(PROJECT_APP_PATH.user_cache / 'face_detection_frame.jpg')

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.build()

  def post_build(self):
    self.ids.slider1.bind(value=self.sample)
    self.ids.slider2.bind(value=self.sample)
    self.ids.slider3.bind(value=self.sample)

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

      dropdown_layout.add_widget(btn2)  # then add the button inside the dropdown

    self._dropdown_btn = Button(text='Model', size_hint=(.5, .1))  # create a big main button

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

  def sample(self, *args):
    rgb = model.sample_from([self.ids.slider1.value,
                             self.ids.slider2.value,
                             self.ids.slider3.value
                             ],
                            device=DEVICE).view(size_x, size_y).detach().numpy()

    imageio.imsave(self._frame_name, rgb)
    self.ids.image_source.reload()

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


class SamplerApp(App):
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
      Image:
        id: image_source
        source: '{MainLayout._frame_name}'
    GridLayout:
      cols: 1
      rows: 3
      size_hint: [1,.20]
      BoxLayout:
        Label:
          size_hint: [.2,1]
          text: '#1'
        Slider:
          id: slider1
          value: 0
          min: -1
          max: 1
          step: 0.01
          orientation: 'horizontal'
        Label:
          size_hint: [.2,1]
          text: str(slider1.value)
      BoxLayout:
        Label:
          size_hint: [.2,1]
          text: '#2'
        Slider:
          id: slider2
          value: 0
          min: -1
          max: 1
          step: 0.01
          orientation: 'horizontal'
        Label:
          size_hint: [.2,1]
          text: str(slider2.value)
      BoxLayout:
        Label:
          size_hint: [.2,1]
          text: '#3'
        Slider:
          id: slider3
          value: 0
          min: -1
          max: 1
          step: 0.01
          orientation: 'horizontal'
        Label:
          size_hint: [.2,1]
          text: str(slider3.value)
    BoxLayout:
      size_hint: [1,.10]
      GridLayout:
        cols: 3
        spacing: '10dp'
        Button:
          id: status
          text:'Resample'
          bold: True
          background_normal: ''
          background_color: (0.82, 0.82, 0.82, 1.0)
          on_press: root.sample()
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
    a = Builder.load_string(SamplerApp.layout_kv)
    a.post_build()
    return a


def main():
  SamplerApp().run()


if __name__ == '__main__':

  main()
