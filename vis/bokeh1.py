#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:42:03 2017

@author: txt5999
"""
import numpy as np
from bokeh.plotting import figure, output_file, show
import csv

dates = []
values = []
with open('feeds.csv', encoding="utf-8", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        dates.append(row[0])
        values.append(row[1])

npdates = np.array(dates, dtype=np.datetime64)
npvalues = np.array(values)

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# output to static HTML file
output_file("happy.html")

# create a new plot with a title and axis labels
p = figure(title="Pursuit of Happyness", x_axis_type="datetime", x_axis_label='Date', y_axis_label='Happy level')

# add a line renderer with legend and line thickness
p.line(npdates, npvalues, legend="Temp.", line_width=2)

# show the results
show(p)