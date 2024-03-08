#generieren von allen inp files für die simulation von modellregen
# + normale Niederschlagsereignisse
# hinterlegen an alle flächen

from io import StringIO

from pandas import read_csv

import swmm_api
from swmm_api import SwmmInput
from swmm_api.input_file import section_labels as sections
from swmm_api.input_file.section_types import SECTION_TYPES

from swmm_api.input_file.sections import Timeseries
from swmm_api.input_file.sections.others import TimeseriesData, TimeseriesFile

import os



working_dir = os.getcwd()

csv_file = StringIO(""" ,TS1
0:05,1.7
0:10,2.3
0:15,3.4
0:20,8
0:25,1.15 """)


series = read_csv(csv_file, index_col=0).squeeze("columns")
series

inp = swmm_api.read_inp_file('pythonProject\\swmm_Gievenbeck.inp')
inp[sections.TIMESERIES].add_obj(TimeseriesData('TS1', data=list(zip(series.index,series))))
print(inp[sections.TIMESERIES].to_inp_lines())




gievenbeck = swmm_api.read_inp_file('pythonProject\\swmm_Gievenbeck.inp')

swmm_api.input_file.
