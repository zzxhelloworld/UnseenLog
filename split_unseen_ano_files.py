import os
import random
import re

from args_config import get_args
from dataloader import load_dataset
from utils import count_label_dist_per_file


def split_graphs_4_unseen_anomalies(ano_files_with_gids, positive_train_ratio, anomalous_split_attempt_num,
                                    test_unseen_ano_file_num=1):
    """
        将图数据划分为训练集和测试集，确保测试集的图的异常文件不会出现在训练集中。
        :param positive_train_ratio: 正常图在训练集/测试集的比例
        :param anomalous_split_attempt_num: 划分尝试的最大次数
        :param test_unseen_ano_file_num: 测试集的未见过异常文件的最小数
        """
    # 分离 positive 和 anomalous 图
    positive_graphs = {g_id: files for g_id, files in ano_files_with_gids.items() if
                       'positive' in files or 'Normal' in files or 'Anomaly' in files}
    anomalous_graphs = {g_id: files for g_id, files in ano_files_with_gids.items() if
                        'positive' not in files and 'Normal' not in files and 'Anomaly' not in files}

    """split positive log graph into train/test set"""
    positive_graph_ids = list(positive_graphs.keys())
    random.shuffle(positive_graph_ids)
    positive_train_num = int(len(positive_graph_ids) * positive_train_ratio)
    positive_train_gids = positive_graph_ids[:positive_train_num]
    positive_test_gids = positive_graph_ids[positive_train_num:]

    """split anomalous graphs into train/test set"""
    attempt_num = 1
    anomalous_files = set()
    for _, files in anomalous_graphs.items():
        anomalous_files.update(files)
    anomalous_files = list(anomalous_files)
    print("anomalous files: ", anomalous_files)
    print("anomalous file number: ", len(anomalous_files))
    print("number of anomalous file only in test set: ", test_unseen_ano_file_num)
    # test_anomalous_ratio = 0.3
    anomalous_train_gids = set()
    anomalous_test_gids = set()
    while attempt_num < anomalous_split_attempt_num:
        random.shuffle(anomalous_files)  # shuffle
        # least_test_size = int(test_anomalous_file_ratio * len(anomalous_files))  # least number of anomalous files in test set
        test_anomalous_files = set(anomalous_files[:test_unseen_ano_file_num])
        print("anomalous file only in test set: ", str(test_anomalous_files))
        for gid, files in anomalous_graphs.items():
            if bool(set(files) & test_anomalous_files):  # has intersection, then go to test set
                anomalous_test_gids.add(gid)
            else:
                anomalous_train_gids.add(gid)
        if not anomalous_test_gids or not anomalous_train_gids:
            print("Not found suitable split, retry...")
            attempt_num += 1
            anomalous_train_gids.clear()
            anomalous_test_gids.clear()
        else:
            break
    if attempt_num >= anomalous_split_attempt_num:
        print("max attempt reaches!")
    if anomalous_train_gids and anomalous_test_gids:
        print("train set gids(positive): ", str(positive_train_gids))
        print("train set gids(anomalous): ", str(list(anomalous_train_gids)))
        print("test set gids(positive): ", str(positive_test_gids))
        print("test set gids(anomalous): ", str(list(anomalous_test_gids)))
        return positive_train_gids, list(anomalous_train_gids), positive_test_gids, list(anomalous_test_gids)
    else:
        print("No suitable split found!")
        return None


def generate_specific_split(test_gids, dataset_dir, output_file):
    file_add_dict = {}  # gid: list of graph path
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.txt'):
            g_id = int(filename.split("-")[0])
            file_path = os.path.join(dataset_dir, filename)
            if g_id not in file_add_dict:
                file_add_dict[g_id] = []
            file_add_dict[g_id].append(file_path)

    train_set_info = []
    val_set_info = []
    test_set_info = []
    for gid in test_gids:
        for add in file_add_dict[gid]:
            # print(str(gid) + " : " + add)
            test_set_info.append(str(gid) + " : " + add)

    for gid, add_list in file_add_dict.items():
        if gid not in test_gids:
            for add in add_list:
                train_set_info.append(str(gid) + " : " + add)

    val_set_info.extend(test_set_info)

    with open(output_file, "w") as file:
        file.write("train set paths:\n")
        for item in train_set_info:
            file.write(item + "\n")
        file.write("validation set paths:\n")
        for item in val_set_info:
            file.write(item + "\n")
        file.write("test set paths:\n")
        for item in test_set_info:
            file.write(item + "\n")


def main():
    args = get_args()
    dataset_dir = f"./{args.dataset_name}_dataset"
    ano_files_with_gids = {}
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.txt'):
            g_id = int(filename.split("-")[0])
            if g_id not in ano_files_with_gids:
                file_path = os.path.join(dataset_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                ano_files = re.findall(r"label=([^\n]+)", text)
                ano_files_with_gids[g_id] = ano_files

    pos_train_gids, ano_train_gids, pos_test_gids, ano_test_gids = split_graphs_4_unseen_anomalies(ano_files_with_gids,
                                                                                                   0.8, 10000,
                                                                                                   test_unseen_ano_file_num=3)
    train_set_gids = []
    train_set_gids.extend(pos_train_gids)
    train_set_gids.extend(ano_train_gids)
    test_set_gids = []
    test_set_gids.extend(pos_test_gids)
    test_set_gids.extend(ano_test_gids)
    output_file = f"./{args.dataset_name}_data/specific_dataset_{args.dataset_id}.txt"
    generate_specific_split(test_set_gids, dataset_dir, output_file)
    print(f"Generated dataset saved into {output_file}")

    print("\nGenerated dataset split info:\n")
    train_dataloader, val_dataloader, test_dataloader = load_dataset(args)
    print("train dataloader:")
    count_label_dist_per_file(train_dataloader)
    print("\ntest dataloader:")
    count_label_dist_per_file(test_dataloader)


if __name__ == "__main__":
    main()
