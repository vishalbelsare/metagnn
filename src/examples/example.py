import os.path as osp
import re

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap

from torch_geometric.data import Data 
from torch_geometric.data import InMemoryDataset

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
        return ['assembly_graph_with_scaffolds.gfa', 'contigs.paths', 'kraken2.out']

    @property
    def processed_file_names(self):
        return ['pyg_meta_graph.pt']

    def download(self):
        pass

    def process(self):
        assembly_graph_file = osp.join(self.raw_dir, self.raw_file_names[0])
        contig_paths = osp.join(self.raw_dir, self.raw_file_names[1])
        taxa_file = osp.join(self.raw_dir, self.raw_file_names[2]) 
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



## Construct the feature vector from kraken2 output 
#-------------------------------

        data_list = []
        node_features = []
        node_vector_map = defaultdict(set)

        # Get tax labels from kraken2 output 
        with open(taxa_file) as file:
            line = file.readline()
            while line != "":

                if "C" in line:
                    strings = line.split("\t")
                    node_id = strings[1].split("_")[1]
                    node_features.append(int(strings[2]))
                    taxa = strings[4].split(" ")
                    for taxon in taxa:
                        if ":" in taxon:
                            txid, abd = int(taxon.split(":")[0]), int(taxon.split(":")[1])
                            if node_id not in node_vector_map:
                                node_vector_map[node_id] = list([txid,abd])
                            else:
                                node_vector_map[node_id].append([txid,abd])
                # node_features.append(node_vector_map[node_id])
                line = file.readline()

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


dataset = Metagenomic(root='../data/', name='Bacteria')
print(dataset[0].x[0])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

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
