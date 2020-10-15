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

import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

name_map = BidirectionalMap()
external_taxon_map = BidirectionalMap()

def populate_name_map(file_name):
    node_count = 0
    current_contig_num = ""
    with open(file_name) as file:
        line = file.readline()
        while line != "":
            strings = line.split("\t")
            name = strings[1]
            name = name.rstrip('\n')
            name = name.lstrip('>')
            name_map[node_count] = name;
            node_count += 1
            line = file.readline()

def populate_external_map(file_name):
    external_taxon_map[0] = 0
    with open(file_name) as file:
        line = file.readline()
        while line != "":
            strings = line.split("\t")
            taxon_id = int(strings[0])
            external_id = int(strings[1])
            external_taxon_map[external_id] = taxon_id
            line = file.readline()

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
            name = line.split(" ")[0]
            name = name.rstrip('\n')
            name = name.lstrip('>')
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
        return ['final_assembly.gfa', 'final_assembly.fasta', 'kraken2.out', 'taxa.encoding']

    @property
    def processed_file_names(self):
        return ['pyg_meta_graph.pt']

    def download(self):
        pass

    def process(self):
        assembly_graph_file = osp.join(self.raw_dir, self.raw_file_names[0])
        contig_fasta = osp.join(self.raw_dir, self.raw_file_names[1])
        taxa_file = osp.join(self.raw_dir, self.raw_file_names[2])
        taxa_encoding = osp.join(self.raw_dir, self.raw_file_names[3])
        
        logger.info("Constructing the assembly graph")
        node_count = len(name_map)
        nodes = []
        links = []
        # Get contig connections from .gfa file
        with open(assembly_graph_file) as file:
            line = file.readline()
            while line != "":
                # Identify lines with link information
                if "E" in line:
                    link = []
                    strings = line.split("\t")
                    if strings[2] != strings[3]:
                        start = strings[2][:-1]
                        end = strings[3][:-1]
                        link.append(start)
                        link.append(end)
                        links.append(link)
                line = file.readline()

        contigs_map = name_map
        contigs_map_rev = contigs_map.inverse
        logger.info("Total number of contigs available: "+str(node_count))

        source_nodes = []
        dest_nodes = []
        ## Construct the assembly graph
        #-------------------------------
        # Create the graph
        assembly_graph = Graph()
        # Create list of edges
        edge_list = []
        # Add vertices
        assembly_graph.add_vertices(node_count)
        # Name vertices
        for i in range(len(assembly_graph.vs)):
            assembly_graph.vs[i]["id"]= i
            assembly_graph.vs[i]["label"]= str(contigs_map[i])
        # Iterate links
        for link in links:
            # Remove self loops
            if link[0] != link[1]:
                # Add edge to list of edges
                src = contigs_map_rev["Contig" + link[0]]
                dest = contigs_map_rev["Contig" + link[1]]
                # print(str(src) + " " + str(dest))
                source_nodes.append(src)
                dest_nodes.append(dest)
                edge_list.append((src, dest))
                    
        # Add edges to the graph
        assembly_graph.add_edges(edge_list)
        assembly_graph.simplify(multiple=True, loops=False, combine_edges=None)
                    
        logger.info("Total number of edges in the assembly graph: "+str(len(edge_list)))

        # Add edges to the graph
        assembly_graph.simplify(multiple=True, loops=False, combine_edges=None)
        clusters = assembly_graph.clusters()
        print("Clusters: " + str(len(clusters)))

## Construct taxa encoding 
#-------------------------------
        taxon_vector_map = defaultdict(set)
        taxon_rank_map = defaultdict(str)
        with open(taxa_encoding) as file:
            line = file.readline()
            while line != "":
                strings = line.split("\t")
                taxon_id = int(strings[0])
                external_id = int(strings[1])
                rank = strings[2]
                hashes = strings[3].split(" ")[:-1]
                taxon_vector_map[external_id] = list(map(int, hashes))
                taxon_rank_map[external_id] = rank
                line = file.readline()

        gc_map, tetra_freq_map = read_or_compute_features(contig_fasta)
## Construct the feature vector from kraken2 output 
#-------------------------------
        data_list = []
        node_features = []
        node_taxon = []
        node_gc = []
        node_tetra_freq = []
        node_idxs = []
        idx = 0
        # Get tax labels from kraken2 output 
        with open(taxa_file) as file:
            line = file.readline()
            while line != "":
                strings = line.split("\t")
                name = strings[1]
                name = name.rstrip('\n')
                node_id = name.lstrip('>contig')
                taxon_id = int(strings[2])

                if taxon_id in taxon_vector_map:
                    node_features.append(taxon_vector_map[taxon_id]) 
                    node_taxon.append(external_taxon_map[taxon_id])
                    node_idxs.append(idx)
                else:
                    empty = [0] * len(taxon_vector_map[1])
                    node_features.append(empty) 
                    node_taxon.append(0)
                    
                if taxon_id in taxon_rank_map:
                    if taxon_rank_map[taxon_id] == 'species':
                        print(str(taxon_id))

                if name in gc_map:
                    node_gc.append(gc_map[name])
                else:
                    node_gc.append(0)

                if name in tetra_freq_map:
                    node_tetra_freq.append(tetra_freq_map[name])
                else:
                    print(name)
                    node_tetra_freq.append(empty)
                idx += 1
                line = file.readline()

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_taxon, dtype=torch.float)
        n = torch.tensor(list(range(0, node_count)), dtype=torch.int)
        g = torch.tensor(node_gc, dtype=torch.float)
        t = torch.tensor(node_tetra_freq, dtype=torch.float)
        edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)

        train_size = int(len(node_idxs)/3)
        val_size = train_size
        train_mask = index_to_mask(node_idxs[:train_size], size=node_count)
        val_mask = index_to_mask(node_idxs[train_size:train_size+val_size], size=node_count)
        test_mask = index_to_mask(node_idxs[train_size+val_size:], size=node_count)

        data = Data(x=x, edge_index=edge_index, y=y, n=n, g=g, t=t)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
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

populate_name_map(osp.join(input_dir, data_name, 'raw', 'kraken2.out'))
populate_external_map(osp.join(input_dir, data_name, 'raw', 'taxa.encoding'))
dataset = Metagenomic(root=input_dir, name=data_name)
data = dataset[0]
print(data)

logger.info("Graph construction done!")
elapsed_time = time.time() - start_time
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")


cluster_data = ClusterData(data, num_parts=1000, recursive=False,
        save_dir=dataset.processed_dir)

loader = ClusterLoader(cluster_data, batch_size=20, shuffle=False,
        num_workers=5)

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
