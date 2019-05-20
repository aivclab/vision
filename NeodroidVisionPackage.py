from setuptools import find_packages

import os
import re

with open(pathlib.Path.joinpath(os.path.dirname(__file__), "vision/version.py"), "r") as f:
  # get version string from module
  version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)


class NeodroidVisionPackage:

  @property
  def test_dependencies(self) -> list:
    return [
      'pytest',
      'mock'
      ]

  @property
  def setup_dependencies(self) -> list:
    return [
      'pytest-runner'
      ]

  @property
  def package_name(self) -> str:
    return 'NeodroidVision'

  @property
  def url(self) -> str:
    return 'https://github.com/sintefneodroid/Vision'

  @property
  def download_url(self):
    return self.url + '/releases'

  @property
  def readme_type(self):
    return 'text/markdown'

  @property
  def packages(self):
    return find_packages(
        exclude=[
          # 'neodroid/environment_utilities'
          ]
        )

  @property
  def author_name(self):
    return 'Christian Heider Nielsen'

  @property
  def author_email(self):
    return 'cnheider@yandex.com'

  @property
  def maintainer_name(self):
    return self.author_name

  @property
  def maintainer_email(self):
    return self.author_email

  @property
  def package_data(self):
    # data = glob.glob('environment_utilities/mab/**', recursive=True)
    return {
      # 'neodroid':[
      # *data
      # 'environment_utilities/mab/**',
      # 'environment_utilities/mab/**_Data/*',
      # 'environment_utilities/mab/windows/*'
      # 'environment_utilities/mab/windows/*_Data/*'
      #  ]
      }

  @property
  def entry_points(self):
    return {
      'console_scripts':[
        # "name_of_executable = module.with:function_to_execute"
        'neodroid-seg = segmentation.run:main',
        'neodroid-cls = classification.run:main',
        'neodroid-rec = reconstruction.run:main'
        ]
      }

  @property
  def extras(self):
    these_extras = {
      # 'GUI':['kivy'],
      # 'mab':['neodroid-linux-mab; platform_system == "Linux"',
      #       'neodroid-win-mab platform_system == "Windows"']

      }

    all_dependencies = []

    for group_name in these_extras:
      all_dependencies += these_extras[group_name]
    these_extras['all'] = all_dependencies

    return these_extras

  @property
  def requirements(self) -> list:
    requirements_out = []
    with open('requirements.txt') as f:
      requirements = f.readlines()

      for requirement in requirements:
        requirements_out.append(requirement.strip())

    return requirements_out

  @property
  def description(self):
    return 'Computer Vision algorithm implementations, intended for use with the Neodroid platform'

  @property
  def readme(self):
    with open('README.md') as f:
      return f.read()

  @property
  def keyword(self):
    with open('KEYWORDS.md') as f:
      return f.read()

  @property
  def license(self):
    return 'Apache License, Version 2.0'

  @property
  def classifiers(self):
    return [
      'Development Status :: 4 - Beta',
      'Environment :: Console',
      'Intended Audience :: End Users/Desktop',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: MacOS :: MacOS X',
      'Operating System :: Microsoft :: Windows',
      'Operating System :: POSIX',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'Natural Language :: English',
      # 'Topic :: Scientific/Engineering :: Artificial Intelligence'
      # 'Topic :: Software Development :: Bug Tracking',
      ]

  @property
  def version(self):
    return version
