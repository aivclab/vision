#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from io import BytesIO, StringIO

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (4,4)
import numpy

from warg import NOD

__author__ = 'cnheider'
__doc__ = ''

def generate_qr():
  import pyqrcode
  import io
  import base64
  code = pyqrcode.create("hello")
  stream = io.BytesIO()
  code.png(stream, scale=6)
  png_encoded = base64.b64encode(stream.getvalue()).decode("ascii")
  return png_encoded

def generate_embedded_svg_plot():
  fig_file = StringIO()
  plt.savefig(fig_file, format='svg', dpi=100)
  fig_svg = f'<svg{fig_file.getvalue().split("<svg")[1]}'
  return fig_svg


def generate_embedded_png_plot():
  import base64
  fig_file = BytesIO()
  plt.savefig(fig_file, format='png', dpi=100)
  fig_file.seek(0)  # rewind to beginning of file
  fig_png = base64.b64encode(fig_file.getvalue()).decode("ascii")
  return fig_png


def generate_html(entries1):
  from jinja2 import Environment, select_autoescape, FileSystemLoader

  env = Environment(loader=FileSystemLoader(template_path),
                    autoescape=select_autoescape(['html', 'xml'])
                    )

  template = env.get_template('classification_report_template.html')
  with open(f'{title}.html', 'w') as f:
    f.writelines(template.render(title=title, entries=entries1))


def generate_pdf():
  import pdfkit
  pdfkit.from_file(f'{title}.html', f'{title}.pdf')


if __name__ == '__main__':
  title = 'report'
  data_path = pathlib.Path.home()
  template_path = 'templates'

  plt.plot(numpy.random((3,3)))

  a = NOD(name=1,
          image=None,
          figure=generate_embedded_svg_plot(),
          prediction="a",
          truth="b")

  plt.plot(numpy.ones((9, 3)))

  b = NOD(name=2,
          image=None,
          figure=generate_embedded_svg_plot(),
          prediction="b",
          truth="c")

  plt.plot(numpy.ones((5, 6)))

  c = NOD(name=3,
          image=generate_embedded_png_plot(),
          figure=None,
          prediction="a",
          truth="a")

  entries = [[a, b], [c], [a, b], [c, b]]

  generate_html(entries)
