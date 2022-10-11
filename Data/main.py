import csv
import dill
import numpy as np

def paper_dist(time, labeled):
    if labeled == 0:
        return -1
    if time <= 2015:
        return 1
    elif time <= 2016:
        return 2
    else:
        return 3


def author_dist(nfl, nfl_dict, us_universities, aff_id, labeled):
    org_aff_id = nfl[aff_id]
    aff_name = nfl_dict[org_aff_id]
    if labeled == 0:
        return -1
    for university in us_universities:
        if university.find(aff_name) != -1:
            return 1
    return 2


def main():
    domain = 'CS'
    data_dir = 'graph_{}/graph_{}.pk'.format(domain, domain)
    graph = dill.Unpickler(open(data_dir, 'rb')).load()

    ###################################################
    # Author label distribution
    nfl = list(graph.node_forward['affiliation'])
    nfl_dict = {}
    tsv_file = open('graph_CS/SeqName_CS_20190919.tsv')
    for row in csv.reader(tsv_file, delimiter="\t"):
        nfl_dict[row[0]] = row[1]
    us_universities = []
    csv_file = open('graph_CS/us_universities.csv')
    for row in csv.reader(csv_file, delimiter="\t"):
        us_universities.append(row[0].lower())
    ###################################################

    types = graph.get_types()
    lengths = {}
    _label_num = 50
    _field_L1_list = list(graph.edge_list['field']['paper']['PF_in_L1'].keys())
    _field_L2_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())
    _venue_list = list(graph.edge_list['venue']['paper']['PV_Conference'].keys())

    for _type in types:
        _len = graph.node_feature[_type].shape[0]
        default_node_emb = np.zeros([_len, 400], dtype=np.float64)
        filename = 'graph_{}/nodes-{}.npy'.format(domain, _type)
        with open(filename, 'wb') as npyfile:
            lengths[_type] = _len
            emb = list(graph.node_feature[_type].loc[:, 'emb'])
            citation = np.log10(np.array(list(graph.node_feature[_type].loc[:, 'citation'])).reshape(-1, 1) + 0.01)
            if 'node_emb' in graph.node_feature[_type]:
                node_emb = list(graph.node_feature[_type].loc[:, 'node_emb'])
                _feature = np.concatenate((node_emb, emb, citation), axis=1)
            else:
                _feature = np.concatenate((default_node_emb, emb, citation), axis=1)
            np.save(npyfile, _feature)
            del _feature

            if _type == 'paper':
                # L1 classification
                _label = -1 * np.ones([_len, _label_num], dtype=np.int64)
                _dist = np.zeros([_len], dtype=np.int64)
                for paper_id in range(_len):
                    if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L1'].keys():
                        continue
                    _time = 2020
                    labels = np.zeros(len(_field_L1_list), dtype=np.int64)
                    for _field in graph.edge_list['paper']['field']['rev_PF_in_L1'][paper_id].keys():
                        labels[_field_L1_list.index(_field)] += 1
                        _time = graph.edge_list['paper']['field']['rev_PF_in_L1'][paper_id][_field]
                    labeled = np.count_nonzero(labels)
                    labeled_num = min(labeled, _label_num)
                    if labeled_num > 0:
                        _label[paper_id, :labeled_num] = np.argsort(labels)[-labeled_num:]
                    _dist[paper_id] = paper_dist(_time, labeled)
                np.save(npyfile, _label)
                np.save(npyfile, np.array([len(_field_L1_list)]))
                np.save(npyfile, _dist)
                del _label
                del _dist
                # L2 classification
                _label = -1 * np.ones([_len, _label_num], dtype=np.int64)
                for paper_id in range(_len):
                    labels = np.zeros(len(_field_L2_list), dtype=np.int64)
                    if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L2'].keys():
                        continue
                    for _field in graph.edge_list['paper']['field']['rev_PF_in_L2'][paper_id].keys():
                        labels[_field_L2_list.index(_field)] += 1
                    labeled = np.count_nonzero(labels)
                    labeled_num = min(labeled, _label_num)
                    if labeled_num > 0:
                        _label[paper_id, :labeled_num] = np.argsort(labels)[-labeled_num:]
                np.save(npyfile, _label)
                np.save(npyfile, np.array([len(_field_L2_list)]))
                del _label
                # Venue classification
                _label = -1 * np.ones([_len, _label_num], dtype=np.int64)
                for paper_id in range(_len):
                    labels = np.zeros(len(_venue_list), dtype=np.int64)
                    if paper_id not in graph.edge_list['paper']['venue']['rev_PV_Conference'].keys():
                        continue
                    for _venue in graph.edge_list['paper']['venue']['rev_PV_Conference'][paper_id].keys():
                        labels[_venue_list.index(_venue)] += 1
                    labeled = np.count_nonzero(labels)
                    labeled_num = min(labeled, _label_num)
                    if labeled_num > 0:
                        _label[paper_id, :labeled_num] = np.argsort(labels)[-labeled_num:]
                np.save(npyfile, _label)
                np.save(npyfile, np.array([len(_venue_list)]))
                del _label

            if _type == 'author':
                # L1 Classification
                _label = -1 * np.ones([_len, _label_num], dtype=np.int64)
                _dist = np.zeros([_len], dtype=np.int64)
                for author_id in range(_len):
                    if author_id not in graph.edge_list['author']['paper']['rev_AP_write_first'].keys():
                        continue
                    if author_id not in graph.edge_list['author']['affiliation']['rev_in'].keys():
                        continue
                    labels = np.zeros(len(_field_L1_list), dtype=np.int64)
                    for paper_id in graph.edge_list['author']['paper']['rev_AP_write_first'][author_id].keys():
                        if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L1'].keys():
                            continue
                        for _field in graph.edge_list['paper']['field']['rev_PF_in_L1'][paper_id].keys():
                            labels[_field_L1_list.index(_field)] += 1
                    labeled = np.count_nonzero(labels)
                    labeled_num = min(labeled, _label_num)
                    if labeled_num > 0:
                        _label[author_id, :labeled_num] = np.argsort(labels)[-labeled_num:]
                    aff_id = list(graph.edge_list['author']['affiliation']['rev_in'][author_id].keys())[0]
                    _dist[author_id] = author_dist(nfl, nfl_dict, us_universities, aff_id, labeled)
                np.save(npyfile, _label)
                np.save(npyfile, np.array([len(_field_L1_list)]))
                np.save(npyfile, _dist)
                del _label
                del _dist
                # L2 Classifiation
                _label = -1 * np.ones([_len, _label_num], dtype=np.int64)
                for author_id in range(_len):
                    if author_id not in graph.edge_list['author']['paper']['rev_AP_write_first'].keys():
                        continue
                    if author_id not in graph.edge_list['author']['affiliation']['rev_in'].keys():
                        continue
                    labels = np.zeros(len(_field_L2_list), dtype=np.int64)
                    for paper_id in graph.edge_list['author']['paper']['rev_AP_write_first'][author_id].keys():
                        if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L2'].keys():
                            continue
                        for _field in graph.edge_list['paper']['field']['rev_PF_in_L2'][paper_id].keys():
                            labels[_field_L2_list.index(_field)] += 1
                    labeled = np.count_nonzero(labels)
                    labeled_num = min(labeled, _label_num)
                    if labeled_num > 0:
                        _label[author_id, :labeled_num] = np.argsort(labels)[-labeled_num:]

                np.save(npyfile, _label)
                np.save(npyfile, np.array([len(_field_L2_list)]))
                del _label

            if _type == 'venue':
                # L1 Classification
                _label = -1 * np.ones([_len, _label_num], dtype=np.int64)
                _dist = np.zeros([_len], dtype=np.int64)
                for venue_id in range(_len):
                    if venue_id in graph.edge_list['venue']['paper']['PV_Conference'].keys():
                        _dist[venue_id] = 2
                    elif venue_id in graph.edge_list['venue']['paper']['PV_Journal'].keys():
                        _dist[venue_id] = 1
                    else:
                        continue
                    labels = np.zeros(len(_field_L1_list), dtype=np.int64)
                    if _dist[venue_id] == 2:
                        for paper_id in graph.edge_list['venue']['paper']['PV_Conference'][venue_id].keys():
                            if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L1'].keys():
                                continue
                            for _field in graph.edge_list['paper']['field']['rev_PF_in_L1'][paper_id].keys():
                                labels[_field_L1_list.index(_field)] += 1
                    elif _dist[venue_id] == 1:
                        for paper_id in graph.edge_list['venue']['paper']['PV_Journal'][venue_id].keys():
                            if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L1'].keys():
                                continue
                            for _field in graph.edge_list['paper']['field']['rev_PF_in_L1'][paper_id].keys():
                                labels[_field_L1_list.index(_field)] += 1
                    labeled = np.count_nonzero(labels)
                    labeled_num = min(labeled, _label_num)
                    if labeled_num > 0:
                        _label[venue_id, :labeled_num] = np.argsort(labels)[-labeled_num:]
                    if labeled == 0:
                        _dist[venue_id] = -1
                np.save(npyfile, _label)
                np.save(npyfile, np.array([len(_field_L1_list)]))
                np.save(npyfile, _dist)
                del _label
                del _dist
                # L2 Classification
                _label = -1 * np.ones([_len, _label_num], dtype=np.int64)
                _dist = 0
                for venue_id in range(_len):
                    if venue_id in graph.edge_list['venue']['paper']['PV_Conference'].keys():
                        _dist = 2
                    elif venue_id in graph.edge_list['venue']['paper']['PV_Journal'].keys():
                        _dist = 1
                    else:
                        continue
                    labels = np.zeros(len(_field_L2_list), dtype=np.int64)
                    if _dist == 2:
                        for paper_id in graph.edge_list['venue']['paper']['PV_Conference'][venue_id].keys():
                            if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L2'].keys():
                                continue
                            for _field in graph.edge_list['paper']['field']['rev_PF_in_L2'][paper_id].keys():
                                labels[_field_L2_list.index(_field)] += 1
                    elif _dist == 1:
                        for paper_id in graph.edge_list['venue']['paper']['PV_Journal'][venue_id].keys():
                            if paper_id not in graph.edge_list['paper']['field']['rev_PF_in_L2'].keys():
                                continue
                            for _field in graph.edge_list['paper']['field']['rev_PF_in_L2'][paper_id].keys():
                                labels[_field_L2_list.index(_field)] += 1
                    labeled = np.count_nonzero(labels)
                    labeled_num = min(labeled, _label_num)
                    if labeled_num > 0:
                        _label[venue_id, :labeled_num] = np.argsort(labels)[-labeled_num:]
                np.save(npyfile, _label)
                np.save(npyfile, np.array([len(_field_L2_list)]))
                del _label


    for target_type in graph.edge_list.keys():
        for source_type in graph.edge_list[target_type].keys():
            for relation_type in graph.edge_list[target_type][source_type].keys():
                filename = 'graph_{}/edges-{}-to-{}-from-{}'.format(domain, relation_type, target_type, source_type) + '.csv'
                with open(filename, 'w') as csvfile:
                    column_names = ['Source', 'Target', 'Time']
                    csv_writer = csv.DictWriter(csvfile, fieldnames=column_names)
                    csv_writer.writeheader()
                    for target_id in graph.edge_list[target_type][source_type][relation_type].keys():
                        if target_id >= lengths[target_type]:
                            print('r')
                            continue
                        for source_id in graph.edge_list[target_type][source_type][relation_type][target_id].keys():
                            if source_id >= lengths[source_type]:
                                print('l')
                                continue
                            _time = graph.edge_list[target_type][source_type][relation_type][target_id][source_id]
                            csv_writer.writerow({'Source': source_id, 'Target': target_id, 'Time' : _time})

if __name__ == "__main__":
    main()

