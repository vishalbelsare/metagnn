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

overlap_graph_file = sys.argv[1]
overlap_graph = Graph()
overlap_graph = overlap_graph.Read_GraphML(overlap_graph_file)

'''
species_map = BidirectionalMap()
species = []
for v in overlap_graph.vs:
    species.append(v['species'])

# prepare vertex labels
species_set = set(species)
idx = 0
for s in species_set:
    species_map[s] = 'Bin-' + str(idx)
    idx += 1
'''

ground_truth = open(sys.argv[2], 'w')
for v in overlap_graph.vs:
    #ground_truth.write(v['readname'] + '\t' + species_map[v['species']] + '\n')
    ground_truth.write(v['readname'] + '\t' + v['species'] + '\n')
ground_truth.close()

