#!/usr/bin/python3

import sys
import os
import subprocess
import csv
import operator
import time
import argparse
import re
import logging

from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap

# Sample command
# -------------------------------------------------------------------
# python meta_gnn.py            --graph /path/to/graph_file.gfa
#                               --paths /path/to/paths_file.paths
#                               --fasta /path/to/fasta_file.paths
#                               --taxon /path/to/kraken_file.out
#                               --output /path/to/output_folder
# -------------------------------------------------------------------

# Setup logger
#-----------------------

logger = logging.getLogger('MetaGNN 1.0')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHeader = logging.StreamHandler()
consoleHeader.setFormatter(formatter)
logger.addHandler(consoleHeader)

start_time = time.time()

# Setup argument parser
#---------------------------------------------------

ap = argparse.ArgumentParser()

ap.add_argument("--graph", required=True, help="path to the assembly graph file")
ap.add_argument("--paths", required=True, help="path to the contigs.paths file")
ap.add_argument("--fasta", required=True, help="path to the contigs.fasta file")
ap.add_argument("--taxon", required=True, help="path to the kraken2 output file")
ap.add_argument("--output", required=True, help="path to the output folder")
ap.add_argument("--prefix", required=False, help="prefix for the output file")
# ap.add_argument("--max_iteration", required=False, type=int, help="maximum number of iterations for label propagation algorithm. [default: 100]")
# ap.add_argument("--diff_threshold", required=False, type=float, help="difference threshold for label propagation algorithm. [default: 0.1]")

args = vars(ap.parse_args())

assembly_graph_file = args["graph"]
contig_paths = args["paths"]
contig_fasta = args["fasta"]
taxon_file = args["taxon"]
output_path = args["output"]
prefix = args["prefix"]
# max_iteration = args["max_iteration"]
# diff_threshold = args["diff_threshold"]


# Setup output path for log file
#---------------------------------------------------

fileHandler = logging.FileHandler(output_path+"/"+prefix+"metagnn.log")
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.info("Welcome to MetaGNN: Metagenomic Contigs classification using GNN.")
logger.info("This version of GraphBin makes use of the assembly graph produced by SPAdes which is based on the de Bruijn graph approach.")

logger.info("Input arguments:")
logger.info("Assembly graph file: "+assembly_graph_file)
logger.info("Contig paths file: "+contig_paths)
logger.info("Contig fasta file: "+contig_fasta)
logger.info("Kraken2 output file: "+taxon_file)
logger.info("Final binning output file: "+output_path)
# logger.info("Maximum number of iterations: "+str(max_iteration))
# logger.info("Difference threshold: "+str(diff_threshold))

logger.info("MetaGNN started")

logger.info("Constructing the assembly graph")

# Get contig paths from contigs.paths
#-------------------------------------

paths = {}
segment_contigs = {}
node_count = 0

my_map = BidirectionalMap()

current_contig_num = ""

try:
    with open(contig_paths) as file:
        name = file.readline()
        path = file.readline()
        
        while name != "" and path != "":
                
            while ";" in path:
                path = path[:-2]+","+file.readline()
            
            start = 'NODE_'
            end = '_length_'
            contig_num = str(int(re.search('%s(.*)%s' % (start, end), name).group(1)))
            
            segments = path.rstrip().split(",")

            if current_contig_num != contig_num:
                my_map[node_count] = int(contig_num)
                current_contig_num = contig_num
                node_count += 1
            
            if contig_num not in paths:
                paths[contig_num] = [segments[0], segments[-1]]
            
            for segment in segments:
                if segment not in segment_contigs:
                    segment_contigs[segment] = set([contig_num])
                else:
                    segment_contigs[segment].add(contig_num)
            
            name = file.readline()
            path = file.readline()

except:
    logger.error("Please make sure that the correct path to the contig paths file is provided.")
    logger.info("Exiting MetaGNN... Bye...!")
    sys.exit(1)

contigs_map = my_map
contigs_map_rev = my_map.inverse

logger.info("Total number of contigs available: "+str(node_count))

links = []
links_map = defaultdict(set)

## Construct the assembly graph
#-------------------------------

try:
    # Get links from assembly_graph_with_scaffolds.gfa
    with open(assembly_graph_file) as file:
        line = file.readline()
        
        while line != "":
            
            # Identify lines with link information
            if "L" in line:
                strings = line.split("\t")
                f1, f2 = strings[1]+strings[2], strings[3]+strings[4]
                links_map[f1].add(f2)
                links_map[f2].add(f1)
                links.append(strings[1]+strings[2]+" "+strings[3]+strings[4])
            line = file.readline()
            

    # Create graph
    assembly_graph = Graph()

    # Add vertices
    assembly_graph.add_vertices(node_count)

    # Create list of edges
    edge_list = []

    # Name vertices
    for i in range(node_count):
        assembly_graph.vs[i]["id"]= i
        assembly_graph.vs[i]["label"]= str(i)

    for i in range(len(paths)):
        segments = paths[str(contigs_map[i])]
        
        start = segments[0]
        start_rev = ""

        if start.endswith("+"):
            start_rev = start[:-1]+"-"
        else:
            start_rev = start[:-1]+"+"
            
        end = segments[1]
        end_rev = ""

        if end.endswith("+"):
            end_rev = end[:-1]+"-"
        else:
            end_rev = end[:-1]+"+"
        
        new_links = []
        
        if start in links_map:
            new_links.extend(list(links_map[start]))
        if start_rev in links_map:
            new_links.extend(list(links_map[start_rev]))
        if end in links_map:
            new_links.extend(list(links_map[end]))
        if end_rev in links_map:
            new_links.extend(list(links_map[end_rev]))
        
        for new_link in new_links:
            if new_link in segment_contigs:
                for contig in segment_contigs[new_link]:
                    if i!=int(contig):
                        # Add edge to list of edges
                        edge_list.append((i,contigs_map_rev[int(contig)]))

    # Add edges to the graph
    assembly_graph.add_edges(edge_list)
    assembly_graph.simplify(multiple=True, loops=False, combine_edges=None)

except:
    logger.error("Please make sure that the correct path to the assembly graph file is provided.")
    logger.info("Exiting MetaGNN... Bye...!")
    sys.exit(1)

logger.info("Total number of edges in the assembly graph: "+str(len(edge_list)))



