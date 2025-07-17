import time

import numpy as np
import torch
from tqdm import tqdm

import base_model
from args_config import get_args
from dataloader import load_dataset
from utils import stat_pred_file_metrics, show_metrics, set_random_seed, load_model, find_best_threshold


def infer(model, test_dataloader, args):
    model.eval()
    test_true_labels = []
    test_predicted_prob = []
    test_corresponding_files = []
    total_infer_time = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = batch.to(args.device)

            start_infer_time = time.time()
            y_prob = torch.sigmoid(model(batch))
            end_infer_time = time.time()
            total_infer_time += (end_infer_time - start_infer_time)

            test_predicted_prob.extend(y_prob.cpu().numpy())
            test_true_labels.extend(batch.y.cpu().numpy())
            files = [name for name_list in batch.file_names for name in name_list]
            test_corresponding_files.extend(files)
    test_predicted_labels = (np.array(test_predicted_prob) >= args.threshold).astype(int)
    stat_pred_file_metrics(test_true_labels, test_predicted_labels, test_corresponding_files)
    f1, recall, precision, auprc, rocauc = show_metrics(test_true_labels, test_predicted_labels, test_predicted_prob)
    print(f"Test performance | f1: {f1}, recall: {recall}, precision: {precision}, prauc: {auprc}, rocauc: {rocauc}")
    print(f"Inference time: {(total_infer_time / 3600):.4f} hours")


def main_anomaly_infer():
    set_random_seed()
    args = get_args()
    print(args)
    _, _, test_dataloader = load_dataset(args, batch_size=512, batch=True)
    """LOAD MODEL"""
    model = base_model.MetaLearner(args).to(args.device)
    model_save_path = f'./{args.dataset_name}_models/{args.dataset_name}_{args.dataset_id}_checkpoint.pth'
    model = load_model(model, model_save_path)
    print("Start to infer anomalies...")
    infer(model, test_dataloader, args)


if __name__ == "__main__":
    main_anomaly_infer()
