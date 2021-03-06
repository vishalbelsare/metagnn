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
species_file = sys.argv[3]

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
        strings = line.split('\t')
        if len(strings) < 2:
            print(strings)
        strings[1] = strings[1].rstrip('\n')
        gt_map[strings[0]] = strings[1]
        line = file.readline()

species_map = {}
species_t_map = {}
with open(species_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        name = strings[0]
        idx = 1
        taxon = strings[1]
        t_type = strings[2].rstrip('\n')
        species_map[name] = int(taxon)
        species_t_map[name] = t_type
        line = file.readline()

true = 0
false = 0
total = 0

for read in k_map:
    if read in gt_map:
        ktaxon = k_map[read]
        sname = gt_map[read]
        if sname in species_map:
            taxon = species_map[sname]
            t_type = species_t_map[sname]
            # if (t_type == sys.argv[4]):
            if ktaxon == taxon:
                true += 1
            else:
                false += 1
            total += 1

print("True: " + str(true))
print("False: " + str(false))
print("Total: " + str(total))
precision = true/(true+false)
recall = true/total

print("Precision: " + str(precision))
print("Recall: " + str(recall))
