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

kraken_file = sys.argv[1]
ground_truth_file = sys.argv[2]

k_map = defaultdict(int)
gt_map = defaultdict(int)
with open(kraken_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        strings[1] = strings[1].rstrip('\n')
        if strings[1].isnumeric():
            k_map[strings[0]] = int(strings[1])
        line = file.readline()

with open(ground_truth_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        if len(strings) < 2:
            print(strings)
        strings[1] = strings[1].rstrip('\n')
        gt_map[strings[0]] = int(strings[1])
        line = file.readline()

true = 0
false = 0
total = 0

for read in k_map:
    if read in gt_map:
        ktaxon = k_map[read]
        if ktaxon > 0:
            taxon = gt_map[read]
            if ktaxon == taxon:
                true += 1
            else:
                false += 1
        total += 1
    else:
        print(read)

print("True: " + str(true))
print("False: " + str(false))
print("Total: " + str(total))
precision = true/(true+false)
recall = true/total

print("Precision: " + str(precision))
print("Recall: " + str(recall))
