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

taxons = {10239, 1122187, 173053, 1385}
gt_map = defaultdict(int)

with open(ground_truth_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        strings[1] = strings[1].rstrip('\n')
        gt_map[strings[0]] = int(strings[1])
        line = file.readline()

filtered_records = []
for record in SeqIO.parse(read_file, 'fastq'):
    name = record.name[:-2]
    if name in gt_map:
        taxon = gt_map[name]
        if taxon in taxons:
            filtered_records.append(record)
    if len(filtered_records) == int(sys.argv[4]):
        break
        
SeqIO.write(filtered_records, filtered_fastq, 'fastq')
