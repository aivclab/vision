#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 11-11-2020
           '''


import streamlit

x = streamlit.slider('Select a value')
streamlit.write(x, 'squared is', x * x)