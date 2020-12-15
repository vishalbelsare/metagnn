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
bin_map_reverse = defaultdict(str)
total_reads = 0
with open(bin_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split("\t")
        b = strings[1].rstrip('\n')
        r = strings[0]
        bin_map[b].append(r)
        bin_map_reverse[r] = b
        total_reads += 1
        line = file.readline()
#print(len(bin_map))

species_map = defaultdict(list) 
species_map_reverse = defaultdict(str) 
all_reads = 0
ground_truth_file = sys.argv[2]
with open(ground_truth_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split('\t')
        s = strings[1].rstrip('\n')
        r = strings[0]
        species_map[s].append(r)
        species_map_reverse[r] = s
        all_reads += 1
        line = file.readline()

#for s in species_map:
    #print(s)
    #print(len(species_map[s]))

#print(len(species_map))

total_max_species = 0
for b in bin_map: # each bin
    # print(len(bin_map[b]))
    species_cnt = defaultdict(int)
    for r in bin_map[b]: # each read in bin b
        if r not in species_map_reverse:
            print(r)
        s = species_map_reverse[r] # species of read r
        if s in species_cnt: #increment species count
            species_cnt[s] += 1
        else:
            species_cnt[s] = 1
    # print(species_cnt)
    m = 0
    for s in species_cnt: # find species with max count
        if m < species_cnt[s]:
            m = species_cnt[s]
    total_max_species += m

precision = total_max_species/total_reads
print("Precision: " + str(precision))


total_max_bins = 0
for s in species_map: # each species
    # print(len(bin_map[b]))
    bin_cnt = defaultdict(int)
    for r in species_map[s]: # each read in species s
        if r not in bin_map_reverse:
            print(r)
        b = bin_map_reverse[r] # bin of read r
        if b in bin_cnt: #increment bin count
            bin_cnt[b] += 1
        else:
            bin_cnt[b] = 1
    # print(bin_cnt)
    m = 0
    for b in bin_cnt: # find bin with max count
        if m < bin_cnt[b]:
            m = bin_cnt[b]
    total_max_bins += m

recall = total_max_bins/all_reads
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






