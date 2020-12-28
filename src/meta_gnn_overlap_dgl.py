import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as Func
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.convert import from_networkx
from dgl.convert import to_networkx
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url
from dgl.data.utils import deprecate_property, deprecate_function
from dgl.data.utils import generate_mask_tensor

import dgl.backend as F
from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN

import numpy as np
import pickle
from Bio import SeqIO
from igraph import *
from collections import defaultdict
from bidirectionalmap.bidirectionalmap import BidirectionalMap

from numpy import array
from numpy import argmax
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
from sys import stdout

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
    idx = 0
    for record in SeqIO.parse(read_file, 'fastq'):
        if record.name in read_names:
            gc_map[record.name] = compute_gc_bias(record.seq)
            tetra_freq_map[record.name] = compute_tetra_freq(record.seq)
        stdout.write("\r%d" % idx)
        stdout.flush()
        idx += 1
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
    overlap_graph = overlap_graph.Read_GraphML(file_name)
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

def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # print(features)
    # return np.asarray(features.todense())
    return np.asarray(features)

species_map = BidirectionalMap()

class Metagenomic(DGLBuiltinDataset):
    r"""The Metagenomic overlap graph dataset.
    Nodes mean reads and edges mean overlap relationship.

    Parameters
    -----------
    name: str
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    _urls = {}

    def __init__(self, name, raw_dir, force_reload=False, verbose=True):
        name = 'metagenomic'
        url = ''
        super(DGLBuiltinDataset, self).__init__(name,
                url=url,
                raw_dir=raw_dir,
                force_reload=force_reload,
                verbose=verbose)

    def download(self):
        pass

    def process(self):
        build_species_map(osp.join(self.raw_dir, 'species_with_readnames.graphml'))
        overlap_graph_file = osp.join(self.raw_dir, 'species_with_readnames.graphml')
        read_file = osp.join(self.raw_dir, 'shuffled_reads.fastq')
        # Read assembly graph and node features from the file into arrays

        overlap_graph = Graph()
        overlap_graph = overlap_graph.Read_GraphML(overlap_graph_file)
        # overlap_graph = overlap_graph.clusters().subgraph(1)

        # Add edges to the graph
        overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)
        # overlap_graph.write_graphml(all_file)

        node_count = overlap_graph.vcount()
        # print("Nodes: " + str(overlap_graph.vcount()))
        # print("Edges: " + str(overlap_graph.ecount()))
        # clusters = overlap_graph.clusters()
        # print("Clusters: " + str(len(clusters)))

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

        graph = nx.DiGraph()
        # graph.add_node(overlap_graph.vcount())
        graph.add_edges_from(overlap_graph.get_edgelist())

        features = torch.tensor(node_tfq, dtype=torch.float)
        values = array(node_labels)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_labels = onehot_encoder.fit_transform(integer_encoded)
        labels = np.argmax(onehot_labels, 1)

        train_size = int(node_count/3)
        
        all_indexes = [i for i in range(node_count)]
        random.shuffle(all_indexes)
        train_index = all_indexes[0:train_size]
        val_index = all_indexes[train_size:train_size*2]
        test_index = all_indexes[train_size*2:]
    
        train_mask = index_to_mask(train_index, size=node_count)
        val_mask = index_to_mask(val_index, size=node_count)
        test_mask = index_to_mask(test_index, size=node_count)

        self._graph = graph
        g = from_networkx(graph)

        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        g.ndata['label'] = F.tensor(labels)
        g.ndata['feat'] = F.tensor(_preprocess_features(features), dtype=F.data_type_dict['float32'])
        self._num_classes = onehot_labels.shape[1]
        self._labels = labels
        self._g = g

        if self.verbose:
            print('Finished data loading and preprocessing.')
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['test_mask']).shape[0]))

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.save_name + '.bin')
        info_path = os.path.join(self.save_path, self.save_name + '.pkl')
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                self.save_name + '.pkl')
        save_graphs(str(graph_path), self._g)
        save_info(str(info_path), {'num_classes': self.num_classes})

    def load(self):
        graph_path = os.path.join(self.save_path,
                self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                self.save_name + '.pkl')
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        graph = graphs[0]
        self._g = graph
        # for compatability
        graph = graph.clone()
        graph.ndata.pop('train_mask')
        graph.ndata.pop('val_mask')
        graph.ndata.pop('test_mask')
        graph.ndata.pop('feat')
        graph.ndata.pop('label')
        graph = to_networkx(graph)
        self._graph = nx.DiGraph(graph)

        self._num_classes = info['num_classes']
        self._g.ndata['train_mask'] = generate_mask_tensor(self._g.ndata['train_mask'].numpy())
        self._g.ndata['val_mask'] = generate_mask_tensor(self._g.ndata['val_mask'].numpy())
        self._g.ndata['test_mask'] = generate_mask_tensor(self._g.ndata['test_mask'].numpy())
        # hack for mxnet compatability

        if self.verbose:
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['test_mask']).shape[0]))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def num_classes(self):
        deprecate_property('dataset.num_classes', 'dataset.num_classes')
        return self.num_classes

    @property
    def num_classes(self):
        return self._num_classes

    """ Citation graph is used in many examples
        We preserve these properties for compatability.
    """
    @property
    def graph(self):
        deprecate_property('dataset.graph', 'dataset[0]')
        return self._graph

    @property
    def train_mask(self):
        deprecate_property('dataset.train_mask', 'g.ndata[\'train_mask\']')
        return F.asnumpy(self._g.ndata['train_mask'])

    @property
    def val_mask(self):
        deprecate_property('dataset.val_mask', 'g.ndata[\'val_mask\']')
        return F.asnumpy(self._g.ndata['val_mask'])

    @property
    def test_mask(self):
        deprecate_property('dataset.test_mask', 'g.ndata[\'test_mask\']')
        return F.asnumpy(self._g.ndata['test_mask'])

    @property
    def labels(self):
        deprecate_property('dataset.label', 'g.ndata[\'label\']')
        return F.asnumpy(self._g.ndata['label'])

    @property
    def features(self):
        deprecate_property('dataset.feat', 'g.ndata[\'feat\']')
        return self._g.ndata['feat']

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args.dataset == 'meta':
        data = Metagenomic('metagenomic', args.raw_dir)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    print(g)
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                Func.relu,
                args.dropout)

    if cuda:
        model.cuda()
    # loss_fcn = torch.nn.CrossEntropyLoss()
    # loss_fcn = Func.nll_loss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        # loss = loss_fcn(logits[train_mask], labels[train_mask])
        loss = Func.nll_loss(logits[train_mask], labels[train_mask], reduction='none')

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.mean().item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--raw-dir", type=str, default='~/.dgl/',
            help="raw data directory")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
