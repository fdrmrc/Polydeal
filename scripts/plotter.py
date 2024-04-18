import numpy as np
from numpy import genfromtxt
import pylab as pl
from matplotlib import collections  as mc

# Square
square_rtree16 = "../meshes/csvs/polygonrtree_16.csv"
square_metis16 = "../meshes/csvs/polygonmetis_16.csv"
square_rtree64 = "../meshes/csvs/polygonrtree_64.csv"
square_metis64 = "../meshes/csvs/polygonmetis_64.csv"

# Ball
ball_rtree80 = "../meshes/csvs/polygonrtree_80.csv"
ball_metis80 = "../meshes/csvs/polygonmetis_80.csv"
ball_rtree20 = "../meshes/csvs/polygonrtree_20.csv"
ball_metis20 = "../meshes/csvs/polygonmetis_20.csv"

# Unstructured
unstructured_rtree91 = "../meshes/csvs/polygonrtree_91.csv"
unstructured_metis91 = "../meshes/csvs/polygonmetis_91.csv"
unstructured_rtree364 = "../meshes/csvs/polygonrtree_364.csv"
unstructured_metis364 = "../meshes/csvs/polygonmetis_364.csv"

files = [
square_rtree16
,square_metis16
,square_rtree64
,square_metis64
, ball_rtree80
, ball_metis80
, ball_rtree20
, ball_metis20
, unstructured_rtree91
, unstructured_metis91
, unstructured_rtree364
, unstructured_metis364
]

grids = []
for file in files:
    grids.append(genfromtxt(file, delimiter=','))


counter = 0
len_prefix = 15
len_suffix = 4


for grid in grids:
    lines = []
    for x1,y1,x2,y2 in grid:
        lines.append([(x1,y1),(x2,y2)])

    lc = mc.LineCollection(lines,linewidths=.5,colors='k')
    fig, ax = pl.subplots(figsize=(10,10))
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_axis_off()
    fig.savefig(files[counter][len_prefix:-len_suffix]+".pdf")
    counter+=1
