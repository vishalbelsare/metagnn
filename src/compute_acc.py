from __future__ import division
import sys
import os
import csv
import operator
import time
import random
import argparse
import re
import logging

from collections import defaultdict

kraken_file = sys.argv[1]
taxa_align_file = sys.argv[2]

kraken_map  = defaultdict(int)
taxa_align_map = defaultdict(int)
with open(kraken_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split("\t")
        if strings[1] != "\n":
            kraken_map[strings[0]] = int(strings[1])
        line = file.readline()

with open(taxa_align_file) as file:
    line = file.readline()
    while line != "":
        strings = line.split("\t")
        if strings[1] == "__Unclassified__":
            taxa_align_map[strings[0]] = 0
        else:
            taxa_align_map[strings[0]] = int(strings[2])
        line = file.readline()

true_pos = 0
false_pos = 0
false_neg = 0
true_neg = 0
not_found = 0
for node in taxa_align_map:
    if node not in kraken_map:
        not_found += 1
        # print("Node " + node + " not found in kraken")
    if  taxa_align_map[node] != 0:
        if taxa_align_map[node] == kraken_map[node]:
            true_pos += 1
        else:
            false_neg += 1
    else:
        if kraken_map[node] == 0:
            true_neg += 1
        else:
            false_pos += 1

print(str(not_found) + " nodes not found")
print("TP: " + str(true_pos))
print("FP: " + str(false_pos))
print("TN: " + str(true_neg))
print("FN: " + str(false_neg))
precision=true_pos/(true_pos+false_pos)
recall=true_pos/(true_pos+false_neg)

print("Precision: " + str(precision))
print("Recall: " + str(recall))

