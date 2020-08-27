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
                name_map[node_count] = name.rstrip("\n")
                current_contig_num = contig_num
                node_count += 1
            name = file.readline()
            path = file.readline()

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
        return ['assembly_graph_with_scaffolds.gfa', 'contigs.paths', 'kraken2.out', 'taxa.encoding']

    @property
    def processed_file_names(self):
        return ['pyg_meta_graph.pt']

    def download(self):
        pass

    def process(self):
        assembly_graph_file = osp.join(self.raw_dir, self.raw_file_names[0])
        contig_paths = osp.join(self.raw_dir, self.raw_file_names[1])
        taxa_file = osp.join(self.raw_dir, self.raw_file_names[2])
        taxa_encoding = osp.join(self.raw_dir, self.raw_file_names[3])
        # Read assembly graph and node features from the file into arrays
        paths = {}
        segment_contigs = {}
        node_count = 0
        contigs_map = BidirectionalMap()
        current_contig_num = ""

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
                    contigs_map[node_count] = int(contig_num)
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

        contigs_map_rev = contigs_map.inverse

        links = []
        links_map = defaultdict(set)
        source_nodes = []
        dest_nodes = []

## Construct the assembly graph
#-------------------------------
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
                            source_nodes.append(i);
                            dest_nodes.append(contigs_map_rev[int(contig)]);
                            edge_list.append((i,contigs_map_rev[int(contig)]))

        # Add edges to the graph
        assembly_graph.add_edges(edge_list)
        assembly_graph.simplify(multiple=True, loops=False, combine_edges=None)
        clusters = assembly_graph.clusters()
        print("Clusters: " + str(len(clusters)))
        # visual_style = {}
        # visual_style["vertex_size"] = 20
        # visual_style["bbox"] = (300, 300)
        # visual_style["layout"] = assembly_graph.layout_random()
        # plot(assembly_graph, "assembly.pdf", **visual_style)

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

## Construct the feature vector from kraken2 output 
#-------------------------------
        data_list = []
        node_features = []
        node_taxon = []
        species_nodes = []
        other_nodes = []
        max_len = 0
        idx = 0
        # Get tax labels from kraken2 output 
        with open(taxa_file) as file:
            line = file.readline()
            while line != "":
                strings = line.split("\t")
                node_id = strings[1].split("_")[1]
                taxon_id = int(strings[2])
                # taxa = strings[4].split(" ")
                # feature_list = []
                # for taxon in taxa:
                    # if ":" in taxon:
                        # txid = int(taxon.split(":")[0])
                        # feature_list.append(txid)
                # print(taxon_id)

                if taxon_id in taxon_vector_map:
                    node_features.append(taxon_vector_map[taxon_id]) 
                    node_taxon.append(external_taxon_map[taxon_id])
                else:
                    empty = [0] * len(taxon_vector_map[1])
                    node_features.append(empty) 
                    node_taxon.append(0)

                if taxon_rank_map[taxon_id] in ['species', 'no rank']:
                    species_nodes.append(idx)
                else:
                    other_nodes.append(idx)
		#increment the node idx
                idx += 1

                # if max_len < len(feature_list):
                    # max_len = len(feature_list)
                # node_features.append(feature_list)
                line = file.readline()
        # print(len(node_taxon))
        # set_node_taxon = set(node_taxon)
        # print(len(set_node_taxon))
        # print(node_features)
        # feature_vector = []
        # for node_list in node_features:
            # if len(node_list) < max_len:
                # resize(node_list, max_len, 0)
            # feature_vector.append(node_list)

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_taxon, dtype=torch.float)
        edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)

        node_idxs = list(range(1, node_count))
        random.shuffle(node_idxs)
        train_size = int(node_count/3)
        val_size = train_size
        train_mask = index_to_mask(node_idxs[:train_size], size=node_count)
        val_mask = index_to_mask(node_idxs[train_size:train_size+val_size], size=node_count)
        test_mask = index_to_mask(node_idxs[train_size+val_size:], size=node_count)

        data = Data(x=x, edge_index=edge_index, y=y)
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
    k_f = output_dir + "/kraken_label.out"
    kf = open(k_f, "w")
    node_idx = 0
    node_idx_1 = 0
    ext_taxon_map_rev = external_taxon_map.inverse
    for data in loader:
        data = data.to(device)
        _, preds = model(data).max(dim=1)
        pred_list = preds.tolist()
        for val in pred_list:
            name = name_map[node_idx]
            taxon = str(ext_taxon_map_rev[val])  
            gf.write(name + '\t' + str(ext_taxon_map_rev[val]) + '\n')
            node_idx += 1
        label_list = data.y.tolist()
        for val in label_list:
            name = name_map[node_idx_1]
            taxon = str(ext_taxon_map_rev[val])  
            kf.write(name + '\t' + str(ext_taxon_map_rev[val]) + '\n')
            node_idx_1 += 1

    gf.close()
    kf.close()

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

populate_name_map(osp.join(input_dir, data_name, 'raw', 'contigs.paths'))
populate_external_map(osp.join(input_dir, data_name, 'raw', 'taxa.encoding'))
dataset = Metagenomic(root=input_dir, name=data_name)
data = dataset[0]
print(data)
# print(data.num_features)
# print(dataset.num_classes)
# print("X: " + data.x.type())
# print("Edge Index: " + data.edge_index.type())
# print("Y: " + data.y.type())

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

output(output_dir)

"""
dataset = dataset
train_size = int(dataset[0].num_nodes/3)
val_size = 1000
dataset = dataset.shuffle()
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size+val_size]
test_dataset = dataset[train_size+val_size:]
len(train_dataset), len(val_dataset), len(test_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()
batch_size=512
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

for epoch in range(1):
    train()

for epoch in range(1):
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)    
    test_acc = evaluate(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))
"""

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
"""

elapsed_time = time.time() - start_time

# Print elapsed time for the process
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")
