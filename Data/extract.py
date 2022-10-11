import csv
import os
import re
import numpy as np
import dill

# This function extracts Machine Learning (ML) or Computer Network (CN) subgraphs from the graph_CS.pk
def extract():
    graph = dill.Unpickler(open('graph_CS/graph_CS.pk', 'rb')).load()
    nfl_key = list(graph.node_forward['field'].keys())

    data_dir = 'graph_CS/'
    #domain = 'ML'
    #domain_id = '119857082'
    domain = 'CN'
    domain_id = '31258907'
    target_field_id = nfl_key.index(domain_id)

    paper_list = set(graph.edge_list['field']['paper']['PF_in_L1'][target_field_id].keys())

    field_list = set()
    for paper_id in paper_list:
        for relation_type in graph.edge_list['paper']['field'].keys():
            if paper_id in graph.edge_list['paper']['field'][relation_type].keys():
                field_list.update(list(graph.edge_list['paper']['field'][relation_type][paper_id].keys()))

    venue_list = set()
    for paper_id in paper_list:
        for relation_type in graph.edge_list['paper']['venue'].keys():
            if paper_id in graph.edge_list['paper']['venue'][relation_type].keys():
                venue_list.update(list(graph.edge_list['paper']['venue'][relation_type][paper_id].keys()))

    author_list = set()
    for paper_id in paper_list:
        for relation_type in graph.edge_list['paper']['author'].keys():
            if paper_id in graph.edge_list['paper']['author'][relation_type].keys():
                author_list.update(list(graph.edge_list['paper']['author'][relation_type][paper_id].keys()))

    affiliation_list = set()
    for author_id in author_list:
        for relation_type in graph.edge_list['author']['affiliation'].keys():
            if author_id in graph.edge_list['author']['affiliation'][relation_type].keys():
                affiliation_list.update(list(graph.edge_list['author']['affiliation'][relation_type][author_id].keys()))

    paper_list = list(paper_list)
    author_list = list(author_list)
    venue_list = list(venue_list)
    field_list = list(field_list)
    affiliation_list = list(affiliation_list)

    for file in os.listdir(data_dir):
        if file.endswith(".npy") and 'nodes-' in file:
            with open(os.path.join(data_dir, file), 'rb') as f:
                _type = re.search('nodes-(.*?).npy', file).group(1)
                _type_list = eval("%s_list" % _type)
                sub_feats = np.load(f).astype(np.float64)[_type_list]
                if _type in ('paper', 'author', 'venue'):
                    sub_labels1 = np.load(f).astype(np.float64)[_type_list]
                    sub_labels_dim1 = np.load(f).astype(np.int64)
                    sub_dists = np.load(f).astype(np.float64)[_type_list]
                    sub_labels2 = np.load(f).astype(np.float64)[_type_list]
                    sub_labels_dim2 = np.load(f).astype(np.int64)
                if _type == 'paper':
                    sub_labels3 = np.load(f).astype(np.float64)[_type_list]
                    sub_labels_dim3 = np.load(f).astype(np.int64)
            filename = 'graph_{}/nodes-{}.npy'.format(domain, _type)
            with open(filename, 'wb') as npyfile:
                np.save(npyfile, sub_feats)
                if _type in ('paper', 'author', 'venue'):
                    np.save(npyfile, sub_labels1)
                    np.save(npyfile, sub_labels_dim1)
                    np.save(npyfile, sub_dists)
                    np.save(npyfile, sub_labels2)
                    np.save(npyfile, sub_labels_dim2)
                if _type == 'paper':
                    np.save(npyfile, sub_labels3)
                    np.save(npyfile, sub_labels_dim3)

    for target_type in graph.edge_list.keys():
        for source_type in graph.edge_list[target_type].keys():
            for relation_type in graph.edge_list[target_type][source_type].keys():
                target_list = eval("%s_list" % target_type)
                source_list = eval("%s_list" % source_type)
                filename = 'graph_{}/edges-{}-to-{}-from-{}'.format(domain, relation_type, target_type, source_type) + '.csv'
                with open(filename, 'w') as csvfile:
                    column_names = ['Source', 'Target', 'Time']
                    csv_writer = csv.DictWriter(csvfile, fieldnames=column_names)
                    csv_writer.writeheader()
                    for target_id_new, target_id in enumerate(target_list):
                        if target_id not in graph.edge_list[target_type][source_type][relation_type].keys():
                            continue
                        for source_id_new, source_id in enumerate(source_list):
                            if source_id not in graph.edge_list[target_type][source_type][relation_type][target_id].keys():
                                continue
                            _time = graph.edge_list[target_type][source_type][relation_type][target_id][source_id]
                            csv_writer.writerow({'Source': source_id_new, 'Target': target_id_new, 'Time' : _time})

if __name__ == "__main__":
    extract()

