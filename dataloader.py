import networkx as nx

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def read_event_one_hot(event_onehot_path, one_hot_length):
    event_name_map = {}
    with open(event_onehot_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            ori_feat = [float(x) for x in line_new.split(",") if x]
            event_one_hot = ori_feat + [float(0) for i in range(one_hot_length - len(ori_feat))]
            if line_old in event_name_map:
                continue
            else:
                event_name_map[line_old] = event_one_hot
    return event_name_map


def read_file_one_hot(file_embedding_path, one_hot_length):
    file_name_map = {}
    with open(file_embedding_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            ori_feat = [float(x) for x in line_new.split(",") if x]
            file_one_hot = ori_feat + [float(0) for i in range(one_hot_length - len(ori_feat))]
            if line_old in file_name_map:
                continue
            else:
                file_name_map[line_old] = file_one_hot
    return file_name_map


def read_exception_one_hot(exception_embedding_path, one_hot_length):
    excep_name_map = {}
    with open(exception_embedding_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            ori_feat = [float(x) for x in line_new.split(",") if x]
            file_one_hot = ori_feat + [float(0) for i in range(one_hot_length - len(ori_feat))]
            if line_old in excep_name_map:
                continue
            else:
                excep_name_map[line_old] = file_one_hot
    return excep_name_map


def construct_graph_to_nx_with_feature(file_path, event_map, file_name_map, exception_map, gid):
    anomalous_nodes = []  # anomalous nodes
    error_type = 'no error'
    edges_with_props = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("Error_Type="):
                error_type = line.strip().split("=")[1]
            if line.startswith("traceID="):
                trace_id = line.strip().split("=")[1]
            if line.startswith("label="):
                ano_node = line.strip().split("=")[1]
                anomalous_nodes.append(ano_node)

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                edge = parts[0].split("<-")
                edge_info = [edge, parts[3], parts[4], parts[5],
                             1 if edge[0] in anomalous_nodes else 0]  # edge, cost, event, exception
                edges_with_props.append(edge_info)

    G = nx.Graph()
    feature_dim = []

    for edge_info in edges_with_props:
        target, source = edge_info[0]
        cost = edge_info[1]
        event = edge_info[2]
        exception = edge_info[3]
        label = edge_info[4]

        if not G.has_edge(target, source):
            G.add_edge(target, source)
        target_feature = event_map[event.replace("event=", "", 1)] + list(file_name_map[target]) + \
                         exception_map[
                             exception.replace("exception=", "", 1)] + [
                             float(cost.split("=")[1].split("m")[0])]
        if not feature_dim:
            feature_dim.append(len(target_feature))
        G.nodes[target]["label"] = label
        G.nodes[target]["gid"] = gid
        if label == 1:
            G.nodes[target]["error_type"] = error_type
        else:
            G.nodes[target]["error_type"] = 'no error'
        G.nodes[target]["name"] = target
        if G.nodes[target].get('exception', None) is None:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True
            else:
                G.nodes[target]["exception"] = False
        else:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True

        feature_value = G.nodes[target].get('feature', None)
        if feature_value is not None:
            feature_value = [x + y for x, y in zip(target_feature, feature_value)]
            G.nodes[target]["feature"] = feature_value
        else:
            G.nodes[target]["feature"] = target_feature
    root_path = []
    for i, node in enumerate(list(G.nodes())):
        if node != "root":
            G.nodes[node]["call_paths"] = [i]
            root_path.append(i)
    """add root node attributes"""
    G.nodes["root"]["feature"] = [float(0)] * feature_dim[0]
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["exception"] = False
    G.nodes["root"]["name"] = "root"
    G.nodes["root"]["call_paths"] = root_path
    G.nodes["root"]["error_type"] = 'no error'
    G.nodes["root"]["gid"] = gid
    return G, error_type


def parse_graphs_to_dataset(graphs, error_type_dict):
    dataset = []
    for graph in graphs:
        feats = torch.tensor([graph.nodes[node]["feature"] for node in graph.nodes()], dtype=torch.float)
        edges = [[list(graph.nodes).index(u), list(graph.nodes).index(v)] for u, v in
                 graph.edges]
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([graph.nodes[node]["label"] for node in graph.nodes()], dtype=torch.long)
        error_types = torch.tensor([error_type_dict[graph.nodes[node]["error_type"]] for node in graph.nodes()],
                                   dtype=torch.long)
        # gids = torch.tensor([int(graph.nodes[node]["gid"]) for node in graph.nodes()],
        #                     dtype=torch.long)
        file_names = [node for node in graph.nodes()]
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, file_names=file_names)
        dataset.append(data)
    return dataset


def load_specific_train_val_test_set(dataset_path, event_map, file_name_map, exception_map, n):
    with open(dataset_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
    train_idx = lines.index("train set paths:\n")
    val_idx = lines.index("validation set paths:\n")
    test_idx = lines.index("test set paths:\n")
    train_paths = [line.strip() for line in lines[train_idx + 1:val_idx] if line.strip()]
    val_paths = [line.strip() for line in lines[val_idx + 1:test_idx] if line.strip()]
    test_paths = [line.strip() for line in lines[test_idx + 1:] if line.strip()]
    train_graph_set = []
    val_graph_set = []
    test_graph_set = []
    gid_num_dict = {}
    for line in train_paths:
        gid, path = line.split(" : ")
        if gid not in gid_num_dict:
            gid_num_dict[gid] = 1
            g, _ = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map, gid)
            train_graph_set.append(g)
        else:
            if n == 0 or gid_num_dict[gid] < n:
                gid_num_dict[gid] = gid_num_dict[gid] + 1
                g, _ = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map, gid)
                train_graph_set.append(g)
    for line in val_paths:
        gid, path = line.split(" : ")
        if gid not in gid_num_dict:
            gid_num_dict[gid] = 1
            g, _ = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map, gid)
            val_graph_set.append(g)
        else:
            if n == 0 or gid_num_dict[gid] < n:
                gid_num_dict[gid] = gid_num_dict[gid] + 1
                g, _ = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map, gid)
                val_graph_set.append(g)
    gid_num_dict_4_test = {}
    for line in test_paths:
        gid, path = line.split(" : ")
        if gid not in gid_num_dict_4_test:
            gid_num_dict_4_test[gid] = 1
            g, _ = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map, gid)
            test_graph_set.append(g)
        else:
            if n == 0 or gid_num_dict_4_test[gid] < n:
                gid_num_dict_4_test[gid] = gid_num_dict_4_test[gid] + 1
                g, _ = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map, gid)
                test_graph_set.append(g)
    return train_graph_set, val_graph_set, test_graph_set


def load_dataset(args, batch_size=64, batch=True):
    root = f"./{args.dataset_name}_data"
    event_embedding_path = root + "/events_compress_one_hot.txt"
    file_embedding_path = root + "/file_name_one_hot.txt"
    exception_embedding_path = root + "/exception_list.txt"
    if args.dataset_name == "forum":
        one_hot_length = 82
    elif args.dataset_name == "halo":
        one_hot_length = 161
    elif args.dataset_name == "novel":
        one_hot_length = 166
    else:
        raise ValueError("not supportive dataset name for one ont length.")
    event_map = read_event_one_hot(event_embedding_path, one_hot_length)
    file_name_map = read_file_one_hot(file_embedding_path, one_hot_length)
    exception_map = read_exception_one_hot(exception_embedding_path, one_hot_length)
    specific_train_val_test_set_path = root + f"/specific_dataset_{args.dataset_id}.txt"

    train_graph_set, val_graph_set, test_graph_set = load_specific_train_val_test_set(
        specific_train_val_test_set_path, event_map, file_name_map, exception_map, args.most_n_same_graph_structure)

    print(
        f"train graph num: {len(train_graph_set)}, validation graph num:{len(val_graph_set)}, test graph num:{len(test_graph_set)}")
    "%---------error type dict----------%"
    "0: no error; 1: chain change; 2: call change; 3: condition change; 4: argument change"
    error_type_dict = {'no error': 0, 'chain change': 1, 'call change': 2, 'condition change': 3, 'argument change': 4}
    "%----------------------------------%"
    train_dataset = parse_graphs_to_dataset(train_graph_set, error_type_dict)
    val_dataset = parse_graphs_to_dataset(val_graph_set, error_type_dict)
    test_dataset = parse_graphs_to_dataset(test_graph_set, error_type_dict)

    if not batch:
        return train_dataset, val_dataset, test_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader
