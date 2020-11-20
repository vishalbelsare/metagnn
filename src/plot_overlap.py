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


#read labels
bin_file = sys.argv[1]

bin_map  = defaultdict(list)
total_reads = 0
with open(bin_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split("\t")
        bin_map[strings[1].rstrip('\n')].append(strings[0])
        total_reads += 1
        line = file.readline()
print(len(bin_map))

read_species_map = defaultdict(str)
ground_truth_file = sys.argv[2]
with open(ground_truth_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split('\t')
        read_species_map[strings[0]] = strings[1].rstrip('\n')
        line = file.readline()
print(len(read_species_map))

total_max = 0
for b in bin_map:
    # print(len(bin_map[b]))
    species_cnt = defaultdict(int)
    for r in bin_map[b]:
        if r not in read_species_map:
            print(r)
        s = read_species_map[r]
        if read_species_map[s] in species_cnt:
            species_cnt[s] =+ 1
        else:
            species_cnt[s] = 1
    # print(species_cnt)
    m = 0
    for s in species_cnt:
        if m < species_cnt[s]:
            m = species_cnt[s]
    total_max += m

precision = total_max/total_reads
recall = total_max/len(read_species_map)

print("Precision: " + str(precision))
print("Recall: " + str(recall))

# overlap_graph_file = sys.argv[2]
# overlap_graph = Graph()
# overlap_graph = overlap_graph.Read_GraphMLz(overlap_graph_file)
# overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)

# vertex_set = overlap_graph.vs
# for idx in range(len(vertex_set)):
    # if vertex_set[idx]['readname'] in read_map:
        # vertex_set[idx]['bin'] = read_map[vertex_set[idx]['readname']]
    # else:
        # vertex_set[idx]['bin'] = 'empty'
# bingraph_file = './metabcc_bin_graph.graphml'
# overlap_graph.write_graphml(bingraph_file)

# species_map = BidirectionalMap()
# species = []
# for v in overlap_graph.vs:
    # species.append(v['species'])

# # prepare vertex labels
# species_set = set(species)
# idx = 0
# for s in species_set:
    # species_map[s] = 'Bin-' + str(idx)
    # idx += 1

# print(species_map)

# ground_truth = open('ground_truth.txt', 'w')
# for v in overlap_graph.vs:
    # ground_truth.write(v['readname'] + '\t' + species_map[v['species']] + '\n')
# ground_truth.close()






