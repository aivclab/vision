#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import h5py
import plotly.offline
from neodroidvision import PROJECT_APP_PATH
from plotly.graph_objs import Figure, Layout, Scatter3d
from warg import ensure_existence

with h5py.File(
    PROJECT_APP_PATH.user_data / "mnist3d" / "testing.h5", "r"
) as points_dataset:
    digits = []
    for i in range(10):
        digit = (
            points_dataset[str(i)]["img"][:],
            points_dataset[str(i)]["points"][:],
            points_dataset[str(i)].attrs["label"],
        )
        digits.append(digit)

for i in range(3):
    x_c = [r[0] for r in digits[i][1]]
    y_c = [r[1] for r in digits[i][1]]
    z_c = [r[2] for r in digits[i][1]]

    layout = Layout(
        height=500, width=600, title=f"Digit: {str(digits[i][2])} in 3D space"
    )
    fig = Figure(
        data=[
            Scatter3d(
                x=x_c,
                y=y_c,
                z=z_c,
                mode="markers",
                marker={
                    "size": 12,
                    "color": z_c,
                    "colorscale": "Viridis",
                    "opacity": 0.7,
                },
            )
        ],
        layout=layout,
    )
    plotly.offline.plot(
        fig,
        filename=str(
            ensure_existence(PROJECT_APP_PATH.user_cache / "mnist3d") / "temp-plot.html"
        ),
    )
