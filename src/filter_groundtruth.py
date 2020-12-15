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
read_list = sys.argv[2]

read_bin_map = defaultdict(str)
with open(ground_truth) as file:
    line = file.readline()
    while line != "":
        strings = line.split('\t')
        read_bin_map[strings[0]] = strings[1].rstrip('\n')
        line = file.readline()

filter_file = 'filterd_gt.txt'
fgt = open(filter_file, 'w')
with open(read_list) as file:
    line = file.readline()
    while line != "":
        name = line.rstrip('\n')
        name = name.lstrip('@')
        fgt.write(read_bin_map[name] + '\n')
        line = file.readline()

fgt.close()




