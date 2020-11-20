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


read_file = sys.argv[1]
ground_truth = sys.argv[2]

read_species_map = defaultdict(str)
ground_truth_file = sys.argv[2]
with open(ground_truth_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split('\t')
        read_species_map[strings[0]] = strings[1].rstrip('\n')
        line = file.readline()

filtered_fastq = sys.argv[3]
filtered_fastq_file = open(filtered_fastq, 'w')
for record in SeqIO.parse(read_file, 'fastq'):
    if record.name in read_species_map:
        filtered_fastq_file.write('@' + record.name + '\n')
        filtered_fastq_file.write(str(record.seq) + '\n')
        filtered_fastq_file.write('+' + '\n')
        filtered_fastq_file.write('+' + '\n')

