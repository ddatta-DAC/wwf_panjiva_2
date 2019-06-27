#!/usr/bin/env python
# coding: utf-8

# In[1]:

import functools
import networkx as nx
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import jaccard_score
import os
import math
import pickle
from collections import OrderedDict
from abc import abstractmethod
import networkx as nx
import os
import sys
from pprint import pprint
import numpy as np

np.set_printoptions(precision=3)
# ------------------------ #

try:
    from src_v1.utils_v1 import Goodall3
except:
    from utils_v1 import Goodall3

try:
    import src_v1.utils_v1  as _utils
except:
    import utils_v1 as _utils

# ------------------------ #
local_data_file_1 = 'datapkl_3.pkl'
# ------------------------ #
'''
Node objects
'''


class node_class:
    _id = 0

    def __init__(self, type):
        self.type = type
        self.score = 0
        self._id = node_class._id
        return

    @abstractmethod
    def set_score(self, s):
        self.score = 0

    @staticmethod
    def increment_id():
        node_class._id += 1


class node_entity(node_class):
    def __init__(self, domain_id, entity_id):
        node_class.__init__(self, type='entity')
        self.domain = domain_id
        self.entity = entity_id
        self.id = node_entity._id
        node_class.increment_id()


class node_record(node_class):

    def __init__(self, record_id):
        node_class.__init__(self, type='record')
        self.record_id = record_id
        node_class.increment_id()


'''
Some simple data to play around with 
'''


def create_data():

    _file_loc = './../test_data_1'
    _file_name1 = 'peru_100_anomalies.txt'
    _file_name2 = 'peru_100_good.txt'

    with open(os.path.join(_file_loc, _file_name1), 'r') as fh:
        inp = fh.read()
        anomalies_Pids = [int(_) for _ in inp.split('\n') if _ is not None and len(_) > 0]

    with open(os.path.join(_file_loc, _file_name2), 'r') as fh:
        inp = fh.read()
        normal_Pids = [int(_) for _ in inp.split('\n') if _ is not None and len(_) > 0]

    # 195 points

    data_df = pd.read_csv(os.path.join(_file_loc, 'peru_export_combined_data_1.csv'))
    del data_df['ShipperCountry']
    feature_columns = list(data_df.columns)
    feature_columns.remove(_utils.CONFIG['id_column'])
    feature_columns = list(sorted(feature_columns))


    # sort them
    all_cols = [_utils.CONFIG['id_column']]
    all_cols.extend(feature_columns)
    print(all_cols)
    data_df = data_df[all_cols]
    id_col = _utils.CONFIG['id_column']

    return data_df, anomalies_Pids, normal_Pids, id_col, feature_columns


'''
A_ft: F x T
'''


def get_A_ft(graph, F, T, F_id_dict):
    _F_id_dict = {
        v: k for k, v in F_id_dict.items()
    }
    a = np.zeros([len(F), len(T)])
    for e in graph.edges():

        i = None
        j = None

        if ((e[0] in F) and (e[1] in T)):
            j = _F_id_dict[e[0]]
            i = e[1]

        elif ((e[0] in T) and (e[1] in F)):
            i = _F_id_dict[e[1]]
            j = e[0]

        if i is not None and j is not None:
            a[i][j] = 1

    return a


# ------------------------ #
# find the _id of a node of entity type
# ------------------------ #


@functools.lru_cache(maxsize=8192)
def find_entity_id_in_graph(
        graph,
        domain,
        entity_identifier
):
    for n in list(graph.nodes):
        obj = graph.node[n]['data']
        if obj.type == 'entity' and obj.domain == domain and obj.entity == entity_identifier:
            return obj._id
    return None


@functools.lru_cache(maxsize=8192)
def find_record_id_in_graph(
        graph,
        record_id
):
    print('finding record PanjivaID', record_id, 'in graph')
    for n in list(graph.nodes):
        obj = graph.node[n]['data']
        if obj.type == 'record' and obj.record_id == record_id:
            return obj._id
    return None


def add_record_feature_edges(
        graph,
        df,
        feature_columns,
        id_col
):

    entity_cols = feature_columns
    entity_prob_dict = {}

    for domain in entity_cols:
        # probability of each entity
        count = Counter(list(df[domain]))
        for e_name, val in count.items():
            x = find_entity_id_in_graph(
                graph,
                domain,
                e_name
            )
            entity_prob_dict[x] = (val + 1) / len(df)

    for i, row in df.iterrows():
        _id = row[id_col]
        n1 = find_record_id_in_graph(graph, _id)

        for domain, e_name in row.to_dict().items():
            if domain == id_col:
                continue
            x = find_entity_id_in_graph(
                graph,
                domain,
                e_name
            )

            graph.add_edge(
                n1,
                x,
                weight=1 / entity_prob_dict[x]
            )

    return graph


def mark_targeted_hscodes(
        graph,
        label_vector,
        domain_entity_nodeID_dict,
        entity_node_dict
):
    hs_code_file = './../test_data_1/collated_hscode_filters.csv'

    _df = pd.read_csv(hs_code_file)
    targets = list(_df.loc[_df['count'] >= 1]['hscode_6'])
    _domain = 'hscode_6'
    _dict = domain_entity_nodeID_dict[_domain]

    for t in targets:
        if t in _dict.keys():
            _id = _dict[t]
            obj = entity_node_dict[_id]
            obj.score = +1
            label_vector[_id] = +1

    return graph, label_vector


def add_record_feature_edges_v2(
        graph,
        df,
        feature_columns,
        id_col,
        domain_entity_nodeID_dict,
        record_nodeID_dict
):
    entity_cols = feature_columns
    entity_prob_dict = {}
    new_df = pd.DataFrame(df)

    def proselytize_1(row, _domain, _graph, domain_entity_nodeID_dict):
        x = domain_entity_nodeID_dict[_domain][row[_domain]]
        return x

    def proselytize_2(row, _graph, record_nodeID_dict):
        x = record_nodeID_dict[row[id_col]]
        return x

    for _domain in feature_columns:
        new_df[_domain] = new_df.apply(
            proselytize_1,
            axis=1,
            args=(_domain, graph, domain_entity_nodeID_dict,)
        )

    new_df[id_col] = new_df.apply(
        proselytize_2,
        axis=1,
        args=(graph, record_nodeID_dict,)
    )

    for domain in entity_cols:
        entity_prob_dict[domain] = {}
        # probability of each entity
        count = Counter(list(new_df[domain]))
        for e_id, val in count.items():
            entity_prob_dict[domain][e_id] = (val + 1) / len(df)

    for i, row in new_df.iterrows():
        r_id = row[id_col]

        for domain, e_id in row.to_dict().items():
            if domain == id_col:
                continue
            if domain in _utils.CONFIG['predefined_domain_weights']:
                m = _utils.CONFIG['predefined_domain_weights'][domain]
            else:
                m = 1

            w =  entity_prob_dict[domain][e_id]
            # w = m * math.exp(w)
            w = m * w

            graph.add_edge(
                r_id,
                e_id,
                weight=w
            )

    return graph


# ------------------------ #
# Capture both feature interaction & feature importance(rarity)
# ------------------------ #
def add_feature_edges(
        graph,
        df,
        entities_dict,
        feature_cols
):
    num_entities = len(entities_dict)

    '''
    entity ids can be from k to k+n
    they are the actual _id s of the nodes , as per order of insertion into graph
    '''
    eId2arrayIndex = {}

    _start = min(entities_dict.keys())
    _end = max(entities_dict.keys())
    _i = 0
    for _ in range(_start, _end + 1):
        eId2arrayIndex[_] = _i
        _i += 1

    arrayIndex2eID = {
        v: k for k, v in eId2arrayIndex.items()
    }

    # idf is 1/p
    idf = np.zeros([num_entities])

    for domain in feature_cols:

        # count of each entity
        _prob = Counter(list(df[domain]))

        # find _id for each of the entities
        for entity_identifier, v in _prob.items():
            idx = find_entity_id_in_graph(
                graph,
                domain,
                entity_identifier
            )
            _idx = eId2arrayIndex[idx]

            p = (v + 1) / len(df)
            idf[_idx] = math.sqrt(1 / p)

    idf = np.reshape(
        normalize(
            np.reshape(
                idf,
                [1, -1]
            ),
            norm='max'
        ), [-1]
    )

    # ----------
    # Calculate co-occurrence matrix
    # ----------
    coocc_matrix = np.zeros(
        [num_entities, num_entities],
        dtype=np.float16
    )

    _tmp = {
        e[1]: e[0] for e in enumerate(feature_cols, 0)
    }

    for uv in itertools.combinations(feature_cols, 2):
        tmp_df = df[list(uv)]
        tmp_df_1 = pd.DataFrame(
            tmp_df.groupby(uv).size().reset_index(name='counts')
        )
        _d1 = uv[0]
        _d2 = uv[1]

        for i, row in tmp_df_1.iterrows():
            # u and v should be array indices of co-occ matrix
            val1 = row[_d1]
            val2 = row[_d2]
            _i1 = find_entity_id_in_graph(
                graph, _d1, val1
            )

            _i2 = find_entity_id_in_graph(
                graph, _d2, val2
            )

            u = eId2arrayIndex[_i1]
            v = eId2arrayIndex[_i2]

            coocc_matrix[u][v] = int(row['counts'])
            coocc_matrix[v][u] = coocc_matrix[u][v]

    print('un normalized co-occ', coocc_matrix)

    coocc_matrix = normalize(
        coocc_matrix,
        axis=1,
        norm='max'
    )

    # for i in range(coocc_matrix.shape[0]):
    #     coocc_matrix[i][i] = 1

    a = np.matmul(
        np.transpose(idf),
        idf
    )
    b = coocc_matrix
    # show_heatmap(b)

    # element wise multiplication
    c = a * b

    # add the edges with weights
    epsilon = 0.00001
    for i in range(c.shape[0]):
        for j in range(i, c.shape[1]):
            _n1 = arrayIndex2eID[i]
            _n2 = arrayIndex2eID[j]
            if b[i][j] > epsilon:
                graph.add_edge(_n1, _n2, weight=b[i][j])

    return graph


# ---------------------------------- #
def show_heatmap(arr):
    if len(arr.shape) == 1:
        arr = np.reshape(arr, [-1, 1])
    plt.imshow(arr, cmap='hot')
    plt.tight_layout()
    plt.show()
    plt.close()


'''
Create graph 
'''


def preprocess_1(
        refresh=False
):
    global local_data_file_1

    data_df, anomalies_Pids, normal_Pids, id_col, feature_columns = create_data()

    if os.path.exists(local_data_file_1) and refresh == False:
        with open(local_data_file_1, 'rb') as fh:
            result = pickle.load(fh)
            initial_labels = result[0]
            g = result[1]
            record_nodeID_dict = result[2]
            domain_entity_nodeID_dict = result[3]
            return data_df, initial_labels, g, record_nodeID_dict, domain_entity_nodeID_dict


    # to remove duplicates
    df_collated = data_df.groupby(feature_columns).count().reset_index()
    del df_collated['PanjivaRecordID']
    del df_collated['Count']




    # consistent ordering of nodes
    graph = nx.OrderedGraph()

    '''
        Record & Entity nodes
        _id : object
    '''
    record_nodes_dict = OrderedDict()
    entity_node_dict = OrderedDict()

    # --------------------- #
    # Add nodes
    # --------------------- #
    record_nodes_list = list(data_df[id_col])
    record_nodeID_dict = OrderedDict()

    for rn in record_nodes_list:
        _node = node_record(
            record_id=rn
        )
        graph.add_node(_node._id, data=_node)
        record_nodes_dict[_node._id] = _node
        record_nodeID_dict[rn] = _node._id

    domain_entity_nodeID_dict = {}
    for domain in feature_columns:
        domain_entity_nodeID_dict[domain] = {}
        # set of entities
        entity_set = sorted(set(data_df[domain]))
        for _e in entity_set:
            _node = node_entity(
                domain_id=domain,
                entity_id=_e
            )
            graph.add_node(
                _node._id,
                data=_node
            )
            domain_entity_nodeID_dict[domain][_e] = _node._id
            entity_node_dict[_node._id] = _node

    print(
        'Number of nodes',
        graph.number_of_nodes()
    )

    '''
    Add in weights
    '''

    graph = add_feature_edges(
        graph,
        data_df,
        entity_node_dict,
        feature_columns
    )
    print(
        'Number of edges after adding feature edges',
        graph.number_of_edges()
    )

    # Add in edges between features & records
    graph = add_record_feature_edges_v2(
        graph,
        data_df,
        feature_columns,
        id_col,
        domain_entity_nodeID_dict,
        record_nodeID_dict
    )
    print('Number of edges after adding feature-record edges', graph.number_of_edges())

    '''
    Initial labels
    Set positive to +1
    Set negative to -1
    Set unlabelled to 0
    '''
    num_nodes = graph.number_of_nodes()
    initial_labels = np.zeros([num_nodes])

    for a in anomalies_Pids:
        x = record_nodeID_dict[a]
        initial_labels[x] = 1
        record_nodes_dict[x].score = 1

    for a in normal_Pids:
        x = record_nodeID_dict[a]
        initial_labels[x] = -1
        record_nodes_dict[x].score = -1

    # ---------- #
    graph, initial_labels = mark_targeted_hscodes(
        graph,
        initial_labels,
        domain_entity_nodeID_dict,
        entity_node_dict
    )

    # Save this
    result = [
        initial_labels,
        graph,
        record_nodeID_dict,
        domain_entity_nodeID_dict
    ]
    with open(local_data_file_1, 'wb') as fh:
        pickle.dump(result, fh, pickle.HIGHEST_PROTOCOL)

    return data_df, initial_labels, graph, record_nodeID_dict, domain_entity_nodeID_dict


# ----------------------------------------- #


def semi_supervised(g, Y):
    # calculate laplacian
    L = nx.linalg.normalized_laplacian_matrix(g)
    Y_orig = np.reshape(Y, [-1, 1])
    Y_old = np.reshape(Y, [-1, 1])
    print(L.shape)
    alpha = 0.01
    iterate = True
    Y_new = None
    iter = 0
    error_epsilon = 0.0000001
    max_iter = 10000

    while iterate:
        a = (1 - alpha) * Y_orig
        _t = L @ np.reshape(Y_old, [-1, 1])
        b = alpha * _t
        Y_new = a + b

        # ensure old labels are maintained
        for i in range(Y_orig.shape[0]):
            if Y_orig[i][0] == 1.0 or Y_orig[i][0] == -1.0:
                Y_new[i][0] = Y_orig[i][0]

        diff = np.max(np.abs(Y_old - Y_new))

        if iter % 100 == 0:
            print('iter :', iter, ' |  diff :', diff)

        iter += 1
        Y_old = np.array(Y_new)

        if diff <= error_epsilon or iter > max_iter:
            iterate = False
            print(' difference = ', diff)

    print(' Number of iterations ', iter)
    return Y_new

# ------------------------- #

'''
update record to +1 or -1
'''
def process_input(
        val,
        graph,
        record_id,
        label_vector,
        record_nodeID_dict
):
    n_id = record_nodeID_dict[record_id]
    label_vector[n_id] = val
    graph.nodes[n_id]['data'].score = val
    return graph, label_vector



def main():
    # original lables in labelled_indices
    df, labelled_indices, g, record_nodeID_dict, domain_entity_nodeID_dict = preprocess_1(False)
    print(
        g.number_of_nodes(),
        g.number_of_edges()
    )

    # new labels
    Y_new = semi_supervised(
        g,
        labelled_indices
    )
    return labelled_indices, Y_new


# df, labelled_indices, g, record_nodeID_dict, domain_entity_nodeID_dict = preprocess_1(True)

