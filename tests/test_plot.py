#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.artist import TheArtist
import numpy as np

x = np.arange(1,10)
z = np.sin(x)
fig = TheArtist(latex = False)

def test_scatter():
    """
    Test the scatter method
    """
    fig.generate_figure_environment(cols = 1, rows =1)
    fig.plot_scatter(x,z,0,0)


def test_plot():
    """
    Test the plot lines method
    """
    fig.generate_figure_environment(cols = 1, rows =1)
    fig.plot_lines(x,z,0,0)


