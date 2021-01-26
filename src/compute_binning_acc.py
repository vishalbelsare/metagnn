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

import json
import numpy as np
import pickle
from Bio import SeqIO
from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap


b_file = sys.argv[1]
gt_file = sys.argv[2]

with open(b_file) as f:
    bin_map = json.load(f)

read_bin_map = {}
for b in bin_map:
    reads = bin_map[b]
    for r in reads:
        read_bin_map[r] = b
# print(len(read_bin_map))

read_map = {}
species_map = {}
with open(gt_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        if len(strings) < 2:
            print(strings)
        read = strings[0]
        taxon = strings[1].rstrip('\n')
        read_map[read] = taxon
        if taxon in species_map:
            species_map[taxon].append(read)
        else:
            species_map[taxon] = [read]

        line = file.readline()

total_binned = 0
for t in bin_map:
    total_binned += len(bin_map[t])
# print(total_binned)

total_reads = 0
for t in species_map:
    total_reads += len(species_map[t])
# print(total_reads)

# print(len(species_map))
# print(len(bin_map))

total_max_count = 0
for b in bin_map:
    s_count = {}
    reads = bin_map[b]
    for r in reads:
        t = read_map[r]
        if t in s_count:
            s_count[t] += 1
        else:
            s_count[t] = 1
    max_count = 0
    for s in s_count:
        if max_count < s_count[s]:
            max_count = s_count[s]

    total_max_count += max_count

total_species_max_count = 0
for s in species_map:
    bin_count = {}
    reads = species_map[s]
    for r in reads:
        if r in read_bin_map:
            b = read_bin_map[r]
            if b in bin_count:
                bin_count[b] += 1
            else:
                bin_count[b] = 1
    max_count = 0
    for b in bin_count:
        if max_count < bin_count[b]:
            max_count = bin_count[b]

    total_species_max_count += max_count

print("Total max bin count: " + str(total_max_count))
print("Total max species count: " + str(total_species_max_count))
print("Total binned: " + str(total_binned))
print("Total reads: " + str(total_reads))

precision = total_max_count/total_binned
recall = total_species_max_count/total_reads

print("Precision: " + str(precision))
print("Recall: " + str(recall))
    
