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
species_taxon_file = sys.argv[3]

k_map  = defaultdict(int)
gt_map = defaultdict(int)
st_map = defaultdict(str)
with open(kraken_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        strings[1] = strings[1].rstrip('\n')
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

with open(species_taxon_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        strings[1] = strings[1].rstrip('\n')
        st_map[int(strings[0])] = strings[1]
        line = file.readline()

true_pos = 0
false_pos = 0
not_found = 0

for read in k_map:
    if read in gt_map:
        taxon = k_map[read]
        if taxon in st_map:
            species = st_map[taxon]
            gt_species = gt_map[read]
            if species in gt_species:
                true_pos += 1
            else:
                false_pos += 1
                #print(species + ' ' + gt_species)
    else:
        not_found += 1
        # print("Node " + node + " not found in kraken")

print(str(not_found) + " nodes not found")
print("TP: " + str(true_pos))
print("FP: " + str(false_pos))
precision=true_pos/(true_pos+false_pos)
recall=true_pos/(len(k_map)-not_found)

print("Precision: " + str(precision))
print("Recall: " + str(recall))

