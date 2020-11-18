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
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from Bio import SeqIO
from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap

from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

species_map = BidirectionalMap()

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

def compute_contig_features(read_file, read_names):
    compute_tetra_list()
    gc_map = defaultdict(float) 
    tetra_freq_map = defaultdict(list)
    for record in SeqIO.parse(read_file, 'fastq'):
        if record.name in read_names:
            gc_map[record.name] = compute_gc_bias(record.seq)
            tetra_freq_map[record.name] = compute_tetra_freq(record.seq)
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
    
def read_or_compute_features(file_name, read_names):
    gc_bias_f = file_name + '.gc'
    tf_f = file_name + '.tf'
    if not os.path.exists(gc_bias_f) and not os.path.exists(tf_f):
        gc_bias, tf = compute_contig_features(file_name, read_names)
        write_features(file_name, gc_bias, tf)
    else:
        gc_bias, tf = read_features(gc_bias_f, tf_f)
    return gc_bias, tf


def build_species_map(file_name):
    overlap_graph = Graph()
    overlap_graph = overlap_graph.Read_GraphMLz(file_name)
    overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)
    
    species = []
    for v in overlap_graph.vs:
        species.append(v['species'])

    # prepare vertex labels
    species_set = set(species)
    idx = 0
    for s in species_set:
        species_map[s] = idx
        idx += 1


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
        return ['6species_with_readnames.graphmlz', 'shuffled_reads.fastq', '6species_all.graphml', '6species_training.graphml']

    @property
    def processed_file_names(self):
        return ['pyg_meta_graph.pt']

    def download(self):
        pass

    def process(self):
        overlap_graph_file = osp.join(self.raw_dir, self.raw_file_names[0])
        read_file = osp.join(self.raw_dir, self.raw_file_names[1])
        all_file = osp.join(self.raw_dir, self.raw_file_names[2])
        training_file = osp.join(self.raw_dir, self.raw_file_names[3])
        # Read assembly graph and node features from the file into arrays

        overlap_graph = Graph()
        overlap_graph = overlap_graph.Read_GraphMLz(overlap_graph_file)

        source_nodes = []
        dest_nodes = []
        # Add edges to the graph
        overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)
        overlap_graph.write_graphml(all_file)

        # prepare edge list
        for e in overlap_graph.get_edgelist():
            source_nodes.append(e[0])
            dest_nodes.append(e[1])

        node_count = overlap_graph.vcount()
        print("Nodes: " + str(overlap_graph.vcount()))
        print("Edges: " + str(overlap_graph.ecount()))
        clusters = overlap_graph.clusters()
        print("Clusters: " + str(len(clusters)))

        # get all vertex names
        vertex_names = []
        vertexes = overlap_graph.vs
        for v in overlap_graph.vs:
            vertex_names.append(v['readname'])
        gc_map, tetra_freq_map = read_or_compute_features(read_file, vertex_names)

        # prepare node features
        node_gc = []
        node_tfq = []
        for v in overlap_graph.vs:
            node_gc.append(gc_map[v['readname']])
            node_tfq.append(tetra_freq_map[v['readname']])

        # prepare vertex labels
        node_labels = []
        for v in overlap_graph.vs:
            node_labels.append(species_map[v['species']])

        # prepare torch objects
        x = torch.tensor(node_tfq, dtype=torch.float)
        g = torch.tensor(node_gc, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.float)
        edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)

        # prepare train/validate/test vectors
        train_size = int(node_count/3)
        val_size = train_size
        train_index = torch.arange(train_size)
        val_index = torch.arange(train_size, train_size+val_size)
        test_index = torch.arange(train_size+val_size, node_count)
        train_mask = index_to_mask(train_index, size=node_count)
        val_mask = index_to_mask(val_index, size=node_count)
        test_mask = index_to_mask(test_index, size=node_count)

        training_graph = overlap_graph
        vertex_set = training_graph.vs
        for i in range(node_count):
          if test_mask[i]:
            vertex_set[i]['species'] = 'Unknown'
        training_graph.write_graphml(training_file)
        learned_graph = training_graph

        data = Data(x=x, edge_index=edge_index, y=y, g=g)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data_list = []
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
def test():
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
def output(output_dir, input_dir):
    overlap_graph_file = input_dir + '/cami6overlap/raw/6species_with_readnames.graphmlz'
    for data in loader:
        data = data.to(device)
        _, preds = model(data).max(dim=1)
        pred_list = preds.tolist()
        learned_graph = Graph()
        learned_graph = learned_graph.Read_GraphMLz(overlap_graph_file)
        rev_species_map = species_map.inverse
        vertex_set = learned_graph.vs
        miss_pred_vertices = []
        for i in range(len(vertex_set)):
            if vertex_set[i]['species'] == rev_species_map[pred_list[i]]:
                vertex_set[i]['pred'] = 'Correct'
            else:
                vertex_set[i]['pred'] = 'Wrong'
                miss_pred_vertices.append(vertex_set[i].index)
            vertex_set[i]['species'] == rev_species_map[pred_list[i]]
        learned_file = output_dir + '/6species_learned.graphml'
        learned_graph.write_graphml(learned_file)
        # print a subgraph
        bfsiter = learned_graph.bfsiter(miss_pred_vertices[0], OUT, True)
        vertex_set = set()
        for v in bfsiter:
            if v[1] < 2: 
                if v[1] > 0:
                    vertex_set.add(v[2].index)
                    vertex_set.add(v[0].index)
        vertex_list = list(vertex_set)
        subgraph = learned_graph.subgraph(vertex_list)
        subgraph_file = output_dir + '/6species_subgraph.graphml'
        subgraph.write_graphml(subgraph_file)

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

logger.info("Welcome to MetaGNN: Metagenomic reads classification using GNN.")
logger.info("This version of MetaGNN makes use of the overlap graph produced by Minimap2.")

logger.info("Input arguments:")
logger.info("Input dir: "+input_dir)
logger.info("Dataset: "+data_name)

logger.info("MetaGNN started")

logger.info("Constructing the overlap graph and node feature vectors")

build_species_map(osp.join(input_dir, data_name, 'raw', '6species_with_readnames.graphmlz'))
dataset = Metagenomic(root=input_dir, name=data_name)
data = dataset[0]
print(data)

#exit()
logger.info("Graph construction done!")
elapsed_time = time.time() - start_time
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")

#cluster_data = ClusterData(data, num_parts=100, recursive=False, save_dir=dataset.processed_dir)

#loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=5)

loader = DataLoader(dataset, batch_size=512, shuffle=True)

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
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    logger.info(log.format(epoch, train_acc, best_val_acc, test_acc))
elapsed_time = time.time() - start_time
# Print elapsed time for the process
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")

#Print GCN model output
output(output_dir, input_dir)

