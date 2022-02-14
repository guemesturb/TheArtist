#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.artist import TheArtist
import numpy as np

x = np.arange(1,10)
z = np.sin(x)
fig = TheArtist(latex = False,n_rows = 1,n_cols = 1)

def test_scatter():
    """
    Test the scatter method
    """
    fig.scatter(x,z,0,0)


def test_plot():
    """
    Test the plot lines method
    """
    fig.plot(x,z,0,0)


