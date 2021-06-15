#!/usr/bin/python3
from __future__ import division
import sys
import os
import subprocess
import csv
import operator
import time
import random
import argparse
import re
import logging
import os.path as osp

import random
import numpy as np
import pickle
from Bio import SeqIO
from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap


read_file = sys.argv[1]
ground_truth_file = sys.argv[2]
filtered_fastq = sys.argv[3]

# CAMI low taxons
# taxons = {10239, 1122187, 173053, 1385}
# CAMI medium taxons
# taxons = {1707, 915, 1817, 2242, 28136, 53458, 68170, 1121478, 1121266, 1121466}
# CAMI high taxons
# taxons = {68, 237, 379, 2242, 103729, 46913, 188872, 270918, 1122185, 225004, 468058, 765201, 1121119, 1189325, 1456769}
# CAMI airways taxons
taxons = {20, 475, 624, 1308, 1378, 32033, 84567, 165695, 272942, 398041}
# CAMI oral taxons
# taxons = {294, 482, 730, 1295, 1743, 28035, 89966, 176280, 698964, 1134782}
gt_map = defaultdict(int)

with open(ground_truth_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        strings[1] = strings[1].rstrip('\n')
        if not strings[1].isnumeric():
            print(strings[1])
        gt_map[strings[0]] = int(strings[1])
        line = file.readline()

filtered_records = []
for record in SeqIO.parse(read_file, 'fastq'):
    # name = record.name[:-2]
    name = record.name
    if name in gt_map:
        taxon = gt_map[name]
        if taxon in taxons:
            filtered_records.append(record)
    if len(filtered_records) == int(sys.argv[4]):
        break
        
SeqIO.write(filtered_records, filtered_fastq, 'fastq')
