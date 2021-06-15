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


print('Loading overlap graph')
overlap_graph = Graph()
overlap_graph = overlap_graph.Read_GraphMLz(sys.argv[1])
print('Graph loaded!')
print("Nodes: " + str(overlap_graph.vcount()))
print("Edges: " + str(overlap_graph.ecount()))
overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)

ground_truth_file = sys.argv[2]

gt_map = defaultdict(int)

with open(ground_truth_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split(' ')
        strings[1] = strings[1].rstrip('\n')
        gt_map[strings[0]] = int(strings[1])
        line = file.readline()

for v in overlap_graph.vs:
    # name = v['readname'][:-2]
    name = v['readname']
    if name in gt_map:
        v['species'] = gt_map[name]
    else:
        print(name)

# prepare vertex labels
# species_set = set(species)

# print(species_set)

#3 species set
#species_list = {'Candidatus_Nitrosopumilus_sp_AR2', 'Colwellia_psychrerythraea_34H', 'Marinobacter_sp_BSs20148'}
#6 species set
# species_list = {'Candidatus_Nitrosopumilus_sp_AR2', 'Colwellia_psychrerythraea_34H', 'Marinobacter_sp_BSs20148', 'Flavobacterium_arcticum', 'Salegentibacter_sp_T436', 'Rhodobacteraceae_bacterium_BAR1'}
#all species
#species_list = list(species_set)
# subgraph = overlap_graph.subgraph(overlap_graph.vs.select(species_in=species_list))
subgraph = overlap_graph
print("Nodes: " + str(subgraph.vcount()))
print("Edges: " + str(subgraph.ecount()))
overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)

# Add edges to the graph
subgraph.simplify(multiple=True, loops=True, combine_edges=None)
subgraph.write_graphml(sys.argv[3])

