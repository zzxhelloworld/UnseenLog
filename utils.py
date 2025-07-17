import os
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, precision_recall_curve, auc


def pretrain(model, train_dataloader, criterion, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        batch = batch.to(device)
        y_logits = model(batch)
        loss = criterion(y_logits, batch.y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_dataloader)  # avg loss
    print("average loss per batch", average_loss)


def val(model, val_dataloader, criterion, args):
    model.eval()
    total_loss = 0.0
    val_true_labels = []
    val_predicted_prob = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(args.device)
            y_logits = model(batch)
            loss = criterion(y_logits, batch.y.float())
            total_loss += loss.item()
            val_predicted_prob.extend(torch.sigmoid(y_logits).cpu().numpy())
            val_true_labels.extend(batch.y.cpu().numpy())
    val_predicted_labels = (np.array(val_predicted_prob) >= args.threshold).astype(int)
    f1, recall, precision, auprc, rocauc = show_metrics(val_true_labels, val_predicted_labels, val_predicted_prob)
    avg_loss = total_loss / len(val_dataloader)
    print("validation avg loss per batch:", avg_loss)
    best_f1, best_recall, best_precision, best_threshold = find_best_threshold(val_true_labels, val_predicted_prob, 0, 1.0, 0.01)
    print(
        f"Validation performance | f1: {best_f1}, recall: {best_recall}, precision: {best_precision}, prauc: {auprc}, rocauc: {rocauc}")
    return f1, auprc, best_f1


def test(model, test_dataloader, args):
    model.eval()
    test_true_labels = []
    test_predicted_prob = []
    test_corresponding_files = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(args.device)
            y_prob = torch.sigmoid(model(batch))
            test_predicted_prob.extend(y_prob.cpu().numpy())
            test_true_labels.extend(batch.y.cpu().numpy())
            files = [name for name_list in batch.file_names for name in name_list]
            test_corresponding_files.extend(files)
    test_predicted_labels = (np.array(test_predicted_prob) >= args.threshold).astype(int)
    stat_pred_file_metrics(test_true_labels, test_predicted_labels, test_corresponding_files)
    f1, recall, precision, auprc, rocauc = show_metrics(test_true_labels, test_predicted_labels, test_predicted_prob)
    best_f1, best_recall, best_precision, best_threshold = find_best_threshold(test_true_labels, test_predicted_prob, 0, 1.0, 0.01)
    print(
        f"Test performance | f1: {best_f1}, recall: {best_recall}, precision: {best_precision}, prauc: {auprc}, rocauc: {rocauc}")
    return best_f1, best_recall, best_precision, auprc, rocauc


def read_code_files(file_id_dict_path):
    file_names = set()
    with open(file_id_dict_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split(' -> ')
            file_name = parts[0].strip()
            file_names.add(file_name)
    file_names.add("root")
    return file_names


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # fix model deterministic
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_metrics(true_labels, predicted_labels, predicted_prob):
    f1 = f1_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    # print(f"true F1 Score: {f1}, true Recall: {recall}, true Precision: {precision}")
    # output.append(f"true F1 Score: {f1}, true Recall: {recall}, true Precision: {precision}")
    rocauc = roc_auc_score(true_labels, predicted_prob)
    # print(f"AUC Score: {auc_value}")
    # output.append(f"AUC Score: {auc_value}")
    """compute AUPRC"""
    p, r, t = precision_recall_curve(true_labels, predicted_prob)
    auprc = auc(r, p)
    # print(f"AUPRC Score: {auprc}")
    # output.append(f"AUPRC Score: {auprc}")
    # print(
    #     f"{f1},{recall},{precision},{auprc},{auc_value}")
    # output.append(
    #     f"{f1},{recall},{precision},{auprc},{auc_value}")
    return f1, recall, precision, auprc, rocauc


def find_best_threshold(y_true, y_pred_score, thres_start, thres_end, thres_step):
    thresholds = np.arange(thres_start, thres_end, thres_step)
    f1_thres_list = []
    for thres in thresholds:
        predicted_labels = (np.array(y_pred_score) >= thres).astype(int)
        f1 = f1_score(y_true, predicted_labels)
        rec = recall_score(y_true, predicted_labels)
        prec = precision_score(y_true, predicted_labels)
        f1_thres_list.append((f1, rec, prec, thres))
    f1_thres_list.sort(key=lambda x: x[0], reverse=True)
    print("f1 threshold:", f1_thres_list)
    return f1_thres_list[0][0], f1_thres_list[0][1], f1_thres_list[0][2], f1_thres_list[0][3]


def stat_pred_file_metrics(true_labels, predicted_labels, corresponding_files):
    file_pred_stat = {}
    for true_label, pred_label, file in zip(true_labels, predicted_labels, corresponding_files):
        if file not in file_pred_stat:
            file_pred_stat[file] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        if true_label == 1 and pred_label == 0:  # fn
            file_pred_stat[file]['fn'] = file_pred_stat[file]['fn'] + 1
        elif true_label == 1 and pred_label == 1:  # tp
            file_pred_stat[file]['tp'] = file_pred_stat[file]['tp'] + 1
        elif true_label == 0 and pred_label == 1:  # fp
            file_pred_stat[file]['fp'] = file_pred_stat[file]['fp'] + 1
        elif true_label == 0 and pred_label == 0:  # tn
            file_pred_stat[file]['tn'] = file_pred_stat[file]['tn'] + 1
    sorted_file_pred_stat = {k: file_pred_stat[k] for k in sorted(file_pred_stat)}
    for file, file_stat in sorted_file_pred_stat.items():
        print(f"Prediction stat for file: {file}")
        tp, fp, tn, fn = file_stat['tp'], file_stat['fp'], file_stat['tn'], file_stat['fn']
        recall = tp / (tp + fn) if (tp + fn) != 0 else -1
        precision = tp / (tp + fp) if (tp + fp) != 0 else -1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else -1
        print(
            f"f1: {f1}, recall: {recall}, precision: {precision} | tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn} | sum: {tp + fp + tn + fn}")
        print("#" * 40)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'epoch': epoch
    }, path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cuda'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    print(f"Model loaded from {path}...")
    return model


def count_label_dist_per_file(dataloader):
    file_label_dist_dict = defaultdict(lambda: defaultdict(int))

    for batch in dataloader:
        if not hasattr(batch, 'file_names') or not hasattr(batch, 'y'):
            raise ValueError("Graph data must have attributes 'file_names' and 'y'!")
        file_names = [name for sublist in batch.file_names for name in sublist]  # flatten
        for file_name, label in zip(file_names, batch.y):
            label = str(int(label.item())) if isinstance(label, torch.Tensor) else str(int(label))
            file_label_dist_dict[file_name][label] += 1

    sorted_file_label_dist_dict = dict()
    for file in sorted(file_label_dist_dict.keys()):
        sorted_label_dist = dict(sorted(file_label_dist_dict[file].items(), key=lambda x: x[0]))
        sorted_file_label_dist_dict[file] = sorted_label_dist

    for idx, (file, dist) in enumerate(sorted_file_label_dist_dict.items()):
        short_name = file.split(".")[-1]
        label_0 = dist.get("0", 0)
        label_1 = dist.get("1", 0)
        print(f"{idx + 1},{short_name},{label_0},{label_1}")  # format: id,file name,number of label 0,number of label 1

    return sorted_file_label_dist_dict


def print_gpu_usage_stat(gpu_id=torch.cuda.current_device()):
    used_mem = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3  # GB
    free_mem = torch.cuda.memory_reserved(gpu_id) / 1024 ** 3
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3
    print(f"GPU {gpu_id} GPU Memory Usage: {used_mem:.2f} GB / {total_mem:.2f} GB")


def stat_label_num(data_iterable):
    from collections import defaultdict
    from torch_geometric.data import Data

    stat_dict = defaultdict(int)

    for item in data_iterable:
        if isinstance(item, Data):
            labels = item.y.view(-1).tolist()
        else:
            labels = item.y.view(-1).tolist()

        for label in labels:
            label_int = int(round(label))  # e.g. -1.0 -> -1
            if label_int in [-1, 0, 1]:
                stat_dict[label_int] += 1
            else:
                print(f"Warning: Unexpected label {label} found.")

    for lbl in [-1, 0, 1]:
        stat_dict[lbl] = stat_dict.get(lbl, 0)

    print("Label statistics:")
    for lbl in [-1, 0, 1]:
        print(f"Label {lbl}: {stat_dict[lbl]}")

    return stat_dict
