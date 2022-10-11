import csv
import sys
import os
import re
from collections import defaultdict

from scipy import sparse as sps
import networkx as nx
import numpy as np
import torch


def graph_reader(args):
    return oag_graph_reader(args)


def oag_graph_reader(args):
    data_dir = args.data_dir + args.dataset

    feats = defaultdict(lambda: np.array)
    labels = defaultdict(lambda: defaultdict(lambda: np.array))
    dists = defaultdict(lambda: np.array)
    edges = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    edge_nums = defaultdict(lambda: 1)

    target_types = defaultdict(set)
    relations = defaultdict(lambda: defaultdict(str))
    feat_dims = defaultdict(int)
    label_dims = defaultdict(lambda: defaultdict(int))
    for file in os.listdir(data_dir):
        if file.endswith(".npy") and 'nodes-' in file:
            with open(os.path.join(data_dir, file), 'rb') as f:
                _type = re.search('nodes-(.*?).npy', file).group(1)
                feats[_type] = np.load(f).astype(np.float64)
                feat_dims[_type] = feats[_type].shape[1]
                if _type in ('paper', 'author', 'venue'):
                    labels[_type]['L1'] = np.load(f).astype(np.int64)
                    label_dims[_type]['L1'] = np.load(f).astype(np.int64).item()
                    dists[_type] = np.load(f).astype(np.int64)
                    labels[_type]['L2'] = np.load(f).astype(np.int64)
                    label_dims[_type]['L2'] = np.load(f).astype(np.int64).item()
                if _type == 'paper':
                    labels[_type]['venue'] = np.load(f).astype(np.int64)
                    label_dims[_type]['venue'] = np.load(f).astype(np.int64).item()

        if file.endswith(".csv") and 'edges-' in file:
            with open(os.path.join(data_dir, file)) as f:
                _target = re.search('to-(.*?)-from', file).group(1)
                _source = re.search('from-(.*?).csv', file).group(1)
                # Fine-grained edge types
                _rel_type = re.search('edges-(.*?)-to', file).group(1)
                if _rel_type == "rev_PF_in_L0" or _rel_type == "PF_in_L0" \
                    or _rel_type == "rev_PF_in_L1" or _rel_type == "PF_in_L1":
                    continue
                if _rel_type == "rev_AP_write_first" or _rel_type == "AP_write_first":
                    continue

                # Merged edge types
                rel_type = _source + "-" + _target
                target_types[_target].update([rel_type])
                relations[rel_type]['source'] = _source
                relations[rel_type]['target'] = _target
                reader = csv.DictReader(f)
                for row in reader:
                    edges[_target][_source][rel_type][int(row['Target'])].update([int(row['Source'])])
                    edge_nums[rel_type] += 1

    """
    # Summary
    for _type in feats.keys():
        print(_type, '>> #nodes: ', feats[_type].shape[0], ', #attributes: ',feats[_type].shape[1])
        if _type not in ('paper', 'author', 'venue'):
            continue
        train_node_pool = np.argwhere(dists[_type] > 1).squeeze()
        test_node_pool = np.argwhere(dists[_type] == 1).squeeze()
        print('#training_node: ', len(train_node_pool), ' , #test_node: ', len(test_node_pool))
        for _task in label_dims[_type].keys():
            print(_task, ">> #label", label_dims[_type][_task])
    for rel_type in edge_nums.keys():
        print(rel_type, ': ', edge_nums[rel_type])
    """
    return feats, labels, dists, edges, target_types, relations, feat_dims, label_dims


def graph_sampler(args, feats, labels, edges, label_dim, dists):
    # Source node type
    train_node_pool = np.argwhere(dists[args.source_node] > 1).squeeze()
    test_node_pool = np.argwhere(dists[args.source_node] == 1).squeeze()
    np.random.shuffle(train_node_pool)
    np.random.shuffle(test_node_pool)

    source_data = []
    b = 0
    while len(source_data) < args.train_batch:
        seed_nodes = train_node_pool[b * args.batch_size : (b+1) * args.batch_size]
        sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.source_node, args.source_task, seed_nodes, args.sample_num, args.sample_depth)
        while type(sampled_labels) == type(None):
            import ipdb; ipdb.set_trace()
            print('Failed to find a complete computation graph for ', args.source_node, ' for training set..')
            b += 1
            seed_nodes = train_node_pool[b * args.batch_size : (b+1) * args.batch_size]
            sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.source_node, args.source_task, seed_nodes, args.sample_num, args.sample_depth)
        b += 1
        source_data.append((sampled_feats, sampled_adjs, sampled_labels, sampled_ids_))

    b = 0
    while len(source_data) < args.train_batch + args.test_batch:
        seed_nodes = test_node_pool[b * args.batch_size : (b+1) * args.batch_size]
        sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.source_node, args.source_task, seed_nodes, args.sample_num, args.sample_depth)
        while type(sampled_labels) == type(None):
            import ipdb; ipdb.set_trace()
            print('Failed to find a complete computation graph for ', args.source_node, ' for test set..')
            b += 1
            seed_nodes = test_node_pool[b * args.batch_size : (b+1) * args.batch_size]
            sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.source_node, args.source_task, seed_nodes, args.sample_num, args.sample_depth)
        b += 1
        source_data.append((sampled_feats, sampled_adjs, sampled_labels, sampled_ids_))

    # Target nodetype
    train_node_pool = np.argwhere(dists[args.target_node] > 1).squeeze()
    test_node_pool = np.argwhere(dists[args.target_node] == 1).squeeze()
    np.random.shuffle(train_node_pool)
    np.random.shuffle(test_node_pool)

    target_data = []
    b = 0
    while len(target_data) < args.train_batch2:
        seed_nodes = train_node_pool[b * args.batch_size : (b+1) * args.batch_size]
        sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.target_node, args.target_task, seed_nodes, args.sample_num, args.sample_depth)
        while type(sampled_labels) == type(None):
            import ipdb; ipdb.set_trace()
            print('Failed to find a complete computation graph for ', args.target_node, ' for training set..')
            b += 1
            seed_nodes = train_node_pool[b * args.batch_size : (b+1) * args.batch_size]
            sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.target_node, args.target_task, seed_nodes, args.sample_num, args.sample_depth)
        b += 1
        target_data.append((sampled_feats, sampled_adjs, sampled_labels, sampled_ids_))

    b = 0
    while len(target_data) < args.train_batch2 + args.test_batch2:
        seed_nodes = test_node_pool[b * args.batch_size : (b+1) * args.batch_size]
        sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.target_node, args.target_task, seed_nodes, args.sample_num, args.sample_depth)
        while type(sampled_labels) == type(None):
            import ipdb; ipdb.set_trace()
            print('Failed to find a complete computation graph for ', args.target_node, ' for test set..')
            b += 1
            seed_nodes = test_node_pool[b * args.batch_size : (b+1) * args.batch_size]
            sampled_feats, sampled_adjs, sampled_labels, sampled_ids_ = graph_sampler_in(feats, labels, edges, label_dim, args.target_node, args.target_task, seed_nodes, args.sample_num, args.sample_depth)
        b += 1
        target_data.append((sampled_feats, sampled_adjs, sampled_labels, sampled_ids_))

    return source_data, target_data


def graph_sampler_in(feats, labels, edges, label_nums, task_node, task_name, target_ids, sample_num, step_num):

    edge_types = []
    edge_budget = []
    for target_type in edges.keys():
        for source_type in edges[target_type].keys():
            for relation_type in edges[target_type][source_type].keys():
                edge_types.append(relation_type)
                edge_budget.append(0)

    sampled_nodes = defaultdict(set)
    sampled_edges = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    sampled_nodes[task_node].update(target_ids)

    target_type_list = set([task_node])
    for _ in range(step_num):
        new_target_type_list = set()
        for target_type in target_type_list:
            for source_type in edges[target_type].keys():
                new_target_type_list.add(source_type)
                for relation_type in edges[target_type][source_type].keys():
                    if edge_budget[edge_types.index(relation_type)] >= 1:
                        continue
                    curr_sampled_nodes = sampled_nodes[target_type].copy()
                    for target_id in curr_sampled_nodes:
                        # Do not sample twice for each node
                        if target_type in sampled_edges.keys() and source_type in sampled_edges[target_type].keys()\
                                and relation_type in sampled_edges[target_type][source_type].keys()\
                                and target_id in sampled_edges[target_type][source_type][relation_type].keys():
                            continue
                        if target_id not in edges[target_type][source_type][relation_type].keys():
                            continue

                        source_ids = list(edges[target_type][source_type][relation_type][target_id])
                        if len(source_ids) == 0:
                            continue
                        elif len(source_ids) < sample_num:
                            sampled_ids = source_ids
                        else:
                            sampled_ids = np.random.choice(source_ids, sample_num, replace = False)
                        sampled_nodes[source_type].update(sampled_ids)
                        sampled_edges[target_type][source_type][relation_type][target_id].update(sampled_ids)
                        edge_budget[edge_types.index(relation_type)] += 1
        target_type_list = new_target_type_list

    sampled_feats = {}
    sampled_adjs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    sampled_labels = None
    sampled_ids_ = []

    for budget in edge_budget:
        if budget == 0:
            return sampled_feats, sampled_adjs, sampled_labels, sampled_ids_

    # Indexing nodes
    index_ = defaultdict(lambda: defaultdict(int))
    for type_ in sampled_nodes.keys():
        for sampled_id in sampled_nodes[type_]:
            index_[type_][sampled_id] = len(list(index_[type_].keys()))
            if type_ == task_node and sampled_id in target_ids:
                sampled_ids_.append(index_[type_][sampled_id])
    # Convert to torch object
    for type_ in sampled_nodes.keys():
        sampled_feats[type_] = torch.FloatTensor(feats[type_][list(sampled_nodes[type_])])
        # Normalize features
        row_sum = torch.norm(sampled_feats[type_], dim=1)
        row_sum = row_sum.masked_fill_(row_sum == 0, 1.)
        row_sum = torch.diag(1/row_sum)
        sampled_feats[type_] = torch.spmm(row_sum, sampled_feats[type_])

        # Normalize labels
        #if type_ in labels.keys(): # possible label leakage
        if type_ == task_node:
            if label_nums[type_][task_name] != labels[type_][task_name].shape[1]:
                sampled_labels = torch.zeros((len(sampled_nodes[type_]), label_nums[type_][task_name]))
                for idx, sampled_id in enumerate(sampled_nodes[type_]):
                    labeled_num = np.count_nonzero(labels[type_][task_name][sampled_id] + 1)
                    label_idx = labels[type_][task_name][sampled_id][:labeled_num]
                    #label_idx = labels[type_][task_name][sampled_id][0]
                    sampled_labels[idx, label_idx] = 1
            else:
                sampled_labels = torch.FloatTensor(labels[type_][task_name][list(sampled_nodes[type_])])
            row_sum = torch.sum(sampled_labels, dim=1)
            row_sum = row_sum.masked_fill_(row_sum == 0, 1.)
            row_sum = torch.diag(1/row_sum)
            sampled_labels = torch.spmm(row_sum, sampled_labels)

    # Generate adjacency matrices for each edge type
    for target_type in sampled_edges.keys():
        for source_type in sampled_edges[target_type].keys():
            for relation_type in sampled_edges[target_type][source_type].keys():
                rows = []
                cols = []
                agg_cols = []
                for target_id in sampled_edges[target_type][source_type][relation_type].keys():
                    target_index_ = index_[target_type][target_id]
                    for source_id in sampled_edges[target_type][source_type][relation_type][target_id]:
                        source_index_ = index_[source_type][source_id]
                        rows.append(target_index_)
                        cols.append(source_index_)
                        agg_cols.append(len(agg_cols))
                sampled_adjs[target_type][source_type][relation_type]['source'] = torch.LongTensor(cols)
                sampled_adjs[target_type][source_type][relation_type]['target'] = torch.LongTensor(rows)
                # Define adjacency matrix
                indices = torch.stack([torch.tensor(rows), torch.tensor(cols)], dim=0)
                attention = torch.ones(len(cols))
                dense_shape = torch.Size([len(list(index_[target_type].keys())), len(list(index_[source_type].keys()))])
                adj = torch.sparse.FloatTensor(indices, attention, dense_shape).to_dense()
                # Normalize
                row_sum = torch.sum(adj, dim=1)
                row_sum = row_sum.masked_fill_(row_sum == 0, 1.)
                row_sum = torch.diag(1/row_sum)
                adj = torch.spmm(row_sum, adj)
                sampled_adjs[target_type][source_type][relation_type]['adj'] = adj

                # Define aggregation adjacency matrix
                indices = torch.stack([torch.tensor(rows), torch.tensor(agg_cols)], dim=0)
                attention = torch.ones(len(agg_cols))
                dense_shape = torch.Size([len(list(index_[target_type].keys())), len(agg_cols)])
                agg_adj = torch.sparse.FloatTensor(indices, attention, dense_shape).to_dense()
                # Normalize
                row_sum = torch.sum(agg_adj, dim=1)
                row_sum = row_sum.masked_fill_(row_sum == 0, 1.)
                row_sum = torch.diag(1/row_sum)
                agg_adj = torch.spmm(row_sum, agg_adj)
                sampled_adjs[target_type][source_type][relation_type]['agg_adj'] = agg_adj

    return sampled_feats, sampled_adjs, sampled_labels, sampled_ids_


