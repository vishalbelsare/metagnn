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

import random
import numpy as np
import pickle
from Bio import SeqIO
from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap


read_file = sys.argv[1]

filtered_fastq = sys.argv[2]
filtered_records = []
for record in SeqIO.parse(read_file, 'fastq'):
    if random.randint(1,10000) % 250 == 0:
        filtered_records.append(record)

SeqIO.write(filtered_records, filtered_fastq, 'fastq')

