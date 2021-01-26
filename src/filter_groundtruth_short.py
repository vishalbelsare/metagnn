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

import numpy as np
import pickle
from Bio import SeqIO
from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap


ground_truth = sys.argv[1]
read_file = sys.argv[2]

read_bin_map = defaultdict(str)
with open(ground_truth) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        read_bin_map[strings[0]] = strings[1].rstrip('\n')
        line = file.readline()

filter_file = sys.argv[3]
fgt = open(filter_file, 'w+')
check_reads = {}
for record in SeqIO.parse(read_file, 'fastq'):
    name = record.name[:-2]
    if name in read_bin_map:
        if name not in check_reads:
            fgt.write(name + ' ' + read_bin_map[name] + '\n')
            check_reads[name] = 1

fgt.close()




