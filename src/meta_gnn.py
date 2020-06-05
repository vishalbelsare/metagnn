import sys
import os
import subprocess
import csv
import operator
import time
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
        my_map = BidirectionalMap()
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


        contigs_map = my_map
        contigs_map_rev = my_map.inverse

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

## Construct taxa encoding 
#-------------------------------

        taxon_vector_map = defaultdict(set)
        with open(taxa_encoding) as file:
            line = file.readline()
            while line != "":
                strings = line.split("\t")
                taxon_id = int(strings[0])
                hashes = strings[1].split(" ")[:-1]
                taxon_vector_map[taxon_id] = list(map(int, hashes))
                line = file.readline()


## Construct the feature vector from kraken2 output 
#-------------------------------

        data_list = []
        node_features = []
        node_taxon = []

        max_len = 0
        # Get tax labels from kraken2 output 
        with open(taxa_file) as file:
            line = file.readline()
            while line != "":

                if "C" in line:
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
                else: 
                    print(taxon_id)
                node_taxon.append(taxon_id)
                # if max_len < len(feature_list):
                    # max_len = len(feature_list)
                # node_features.append(feature_list)
                line = file.readline()

        # print(node_features)
        # feature_vector = []
        # for node_list in node_features:
            # if len(node_list) < max_len:
                # resize(node_list, max_len, 0)
            # feature_vector.append(node_list)

        x = torch.tensor(node_features, dtype=torch.long)
        y = torch.tensor(node_taxon, dtype=torch.long)
        edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)
       
        train_size = int(node_count/3)
        val_size = train_size
        train_index = torch.arange(train_size, dtype=torch.long)
        train_mask = index_to_mask(train_index, size=node_count)
        val_index = torch.arange(train_size, train_size+val_size, dtype=torch.long)
        val_mask = index_to_mask(val_index, size=node_count)
        test_index = torch.arange(train_size+val_size, node_count, dtype=torch.long)
        test_mask = index_to_mask(test_index, size=node_count)

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
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, int(dataset.num_classes), cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask].long()).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


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

ap.add_argument("--input", required=True, help="path to the assembly graph file")
ap.add_argument("--name", required=True, help="path to the contigs.paths file")
# ap.add_argument("--max_iteration", required=False, type=int, help="maximum number of iterations for label propagation algorithm. [default: 100]")
# ap.add_argument("--diff_threshold", required=False, type=float, help="difference threshold for label propagation algorithm. [default: 0.1]")

args = vars(ap.parse_args())

input_dir = args["input"]
data_name = args["name"]

# Setup output path for log file
#---------------------------------------------------

fileHandler = logging.FileHandler(input_dir+"/"+"metagnn.log")
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
print(data.train_mask)
# print(dataset[0].x)
# print(dataset[0].edge_index)

logger.info("Graph construction done!")
elapsed_time = time.time() - start_time
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")

logger.info("Training model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)

optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)

best_val_acc = test_acc = 0
for epoch in range(1, 20):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

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
