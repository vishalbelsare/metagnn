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
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from Bio import SeqIO
from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap

from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def resize(l, newsize, filling=None):
    if newsize > len(l):
        l.extend([filling for x in range(len(l), newsize)])
    else:
        del l[newsize:]

def peek_line(f):
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    return line

tetra_list = []
def compute_tetra_list():
    for a in ['A', 'C', 'T', 'G']:
        for b in ['A', 'C', 'T', 'G']:
            for c in ['A', 'C', 'T', 'G']:
                for d in ['A', 'C', 'T', 'G']:
                    tetra_list.append(a+b+c+d)

def compute_tetra_freq(seq):
    tetra_cnt = []
    for tetra in tetra_list:
        tetra_cnt.append(seq.count(tetra))
    return tetra_cnt

def compute_gc_bias(seq):
    seqlist = list(seq)
    gc_cnt = seqlist.count('G') + seqlist.count('C')
    gc_frac = gc_cnt/len(seq)
    return gc_frac

def compute_contig_features(file_name):
    compute_tetra_list()
    gc_map = defaultdict(float) 
    tetra_freq_map = defaultdict(list)
    with open(file_name) as file:
        line = file.readline()
        while '>' in line:
            name = line.lstrip('>')
            name = name.rstrip('\n')
            seq_line = peek_line(file)
            seq = ''
            while seq_line != "" and '>' not in seq_line:
                seq_line = file.readline()
                seq_line.rstrip('\n')
                seq = seq + seq_line
                seq_line = peek_line(file)
            gc_map[name] = compute_gc_bias(seq)
            tetra_freq_map[name] = compute_tetra_freq(seq)
            line = file.readline()
    return gc_map, tetra_freq_map

def read_features(gc_bias_f, tf_f):
    gc_map = pickle.load(open(gc_bias_f, 'rb'))
    tetra_freq_map = pickle.load(open(tf_f, 'rb'))
    return gc_map, tetra_freq_map

def write_features(file_name, gc_map, tetra_freq_map):
    gc_bias_f = file_name + '.gc'
    tf_f = file_name + '.tf'
    pickle.dump(gc_map, open(gc_bias_f, 'wb'))
    pickle.dump(tetra_freq_map, open(tf_f, 'wb'))
    
def read_or_compute_features(file_name):
    gc_bias_f = file_name + '.gc'
    tf_f = file_name + '.tf'
    if not os.path.exists(gc_bias_f) and not os.path.exists(tf_f):
        gc_bias, tf = compute_contig_features(file_name)
        write_features(file_name, gc_bias, tf)
    else:
        gc_bias, tf = read_features(gc_bias_f, tf_f)
    return gc_bias, tf

class Metagenomic(InMemoryDataset):
    r""" Assembly graph built over raw metagenomic data using spades.
        Nodes represent contigs and edges represent link between them.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"bacteria-10"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(Metagenomic, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['6species_with_readnames.graphmlz', 'shuffled_reads.fastq.gz']

    @property
    def processed_file_names(self):
        return ['pyg_meta_graph.pt']

    def download(self):
        pass

    def process(self):
        overlap_graph_file = osp.join(self.raw_dir, self.raw_file_names[0])
        read_file = osp.join(self.raw_dir, self.raw_file_names[1])
        # Read assembly graph and node features from the file into arrays

        overlap_graph = Graph()
        overlap_graph = overlap_graph.Read_GraphMLz(overlap_graph_file)

        source_nodes = []
        dest_nodes = []
        # Add edges to the graph
        overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)
        for e in overlap_graph.get_edgelist():
            source_nodes.append(e[0])
            dest_nodes.append(e[1])
        print("Edges: " + str(overlap_graph.ecount()))
        clusters = overlap_graph.clusters()
        print("Clusters: " + str(len(clusters)))
        # vertexes = overlap_graph.vs 
        # for v in vertexes:
            # print(v)
        for record in SeqIO.parse(read_file, 'fastq')
            print(record)
        
    def __repr__(self):
        return '{}()'.format(self.name)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128, cached=False)
        self.conv2 = GCNConv(128, int(dataset.num_classes), cached=False)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].long(), reduction='none')
        loss.mean().backward()
        optimizer.step()
        total_loss += loss.mean().item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples

@torch.no_grad()
def test(out):
    model.eval()
    for data in loader:
        data = data.to(device)
        logits, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            _, pred = logits[mask].max(1)
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

    return accs

@torch.no_grad()
def output(output_dir):
    gnn_f = output_dir + "/gnn.out"
    gf = open(gnn_f, "w")
    ext_taxon_map_rev = external_taxon_map.inverse
    for data in loader:
        data = data.to(device)
        _, preds = model(data).max(dim=1)
        pred_list = preds.tolist()
        perm = data.n.tolist()
        for idx,val in zip(perm,pred_list):
            name = name_map[idx]
            taxon = str(ext_taxon_map_rev[val])  
            gf.write(name + '\t' + str(ext_taxon_map_rev[val]) + '\n')
    
    gf.close()

# Sample command
# -------------------------------------------------------------------
# python meta_gnn.py            --input /path/to/raw_files
#                               --name /name/of/dataset
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

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help="path to the input files")
ap.add_argument("-n", "--name", required=True, help="name of the dataset")
ap.add_argument("-o", "--output", required=True, help="output directory")

args = vars(ap.parse_args())

input_dir = args["input"]
data_name = args["name"]
output_dir = args["output"]

# Setup output path for log file
#---------------------------------------------------

fileHandler = logging.FileHandler(output_dir+"/"+"metagnn.log")
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.info("Welcome to MetaGNN: Metagenomic Contigs classification using GNN.")
logger.info("This version of MetaGNN makes use of the assembly graph produced by SPAdes which is based on the de Bruijn graph approach.")

logger.info("Input arguments:")
logger.info("Input dir: "+input_dir)
logger.info("Dataset: "+data_name)

logger.info("MetaGNN started")

logger.info("Constructing the assembly graph and node feature vectors")

dataset = Metagenomic(root=input_dir, name=data_name)
data = dataset[0]
print(data)

logger.info("Graph construction done!")
elapsed_time = time.time() - start_time
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")

exit()
cluster_data = ClusterData(data, num_parts=100, recursive=False,
        save_dir=dataset.processed_dir)

loader = ClusterLoader(cluster_data, batch_size=5, shuffle=False,
        num_workers=5)

# for i in range(len(cluster_data)):
    # if i == 4:
        # print(cluster_data[i])
        # # print(cluster_data[i].edge_index)
        # print(cluster_data[i].x[1])
        # print(cluster_data[i].n[1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Running GNN on: "+str(device))
model = Net().to(device)

optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)

logger.info("Training model")
best_val_acc = test_acc = 0
for epoch in range(1, 20):
    train()
    out = True if epoch == 19 else False
    train_acc, val_acc, tmp_test_acc = test(out)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    logger.info(log.format(epoch, train_acc, best_val_acc, test_acc))
elapsed_time = time.time() - start_time
# Print elapsed time for the process
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")

#Print GCN model output
output(output_dir)

