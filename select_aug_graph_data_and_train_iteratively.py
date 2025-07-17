import random

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import base_model
from dataloader import load_dataset
from graph_augmentation import FeatureNoise, EdgeDropout, EdgeAddition
from model_train import main_model_train
from utils import load_model, stat_label_num


def quick_graph_augmentation(data_list, aug_times, noise_std, drop_ratio, add_ratio):
    """for each graph data,produce aug_times augmented graph data for each augmentation method"""
    aug_data = []
    aug_feature_noise = FeatureNoise(noise_std=noise_std)
    aug_edge_dropout = EdgeDropout(drop_ratio=drop_ratio)
    aug_edge_add = EdgeAddition(add_ratio=add_ratio)
    print("Start to augment graph data...")
    for idx, data in enumerate(data_list):
        if (idx + 1) % 10000 == 0:
            print(f"Augment graph data: {idx + 1}/{len(data_list)}")
        for _ in range(aug_times):
            aug_data.append(aug_feature_noise(data))
            aug_data.append(aug_edge_dropout(data))
            aug_data.append(aug_edge_add(data))
    print("Finish augmenting graph data...")
    return aug_data


def select_trustworthy_aug_data_by_expert(model, aug_dataloader, trustworthy_bound, device):
    """
    Args:
        trustworthy_bound (float): trustworthy bound
        model (nn.Module): model
        aug_dataloader (DataLoader): augmented data
    Returns:
        List[Data]: selected trustworthy data
    """
    model.eval()
    labeled_aug_data_list = []

    with torch.no_grad():
        for idx, batch in enumerate(aug_dataloader):
            if (idx + 1) % 10000 == 0:
                print(f"Select trustworthy aug data: {idx + 1}/{len(aug_dataloader)}")
            batch = batch.to(device)

            y_prob = torch.sigmoid(model(batch))

            node_mask = y_prob > trustworthy_bound

            new_labels = torch.full_like(y_prob, -1.0)
            new_labels[node_mask] = 1.0

            filtered_data = Data(
                x=batch.x,
                edge_index=batch.edge_index,
                y=new_labels,
                file_names=[name for sublist in batch.file_names for name in sublist]
            )

            labeled_aug_data_list.append(filtered_data.cpu())
    print(f"Select trustworthy aug data finished...")
    return labeled_aug_data_list


def get_labeled_aug_data_and_train_model(args):
    """LOAD NORMAL AND TRUSTWORTHY AUGMENTED GRAPH NODE DATA"""
    train_data_list, val_data_list, test_data_list = load_dataset(args, batch=False)

    """Training model with original and augmented graph data"""
    print("Prepare to load original data for training model...")
    # for supervised loss
    origin_train_dataloader = DataLoader(train_data_list, batch_size=args.training_batch_size, shuffle=True)
    origin_val_dataloader = DataLoader(val_data_list, batch_size=args.training_batch_size, shuffle=True)

    print(f"Start to augment data for {args.aug_total_times} times...")
    total_aug_train_data_list = quick_graph_augmentation(train_data_list, args.aug_total_times, args.noise_std,
                                                         args.drop_ratio,
                                                         args.add_ratio)

    print("Start to train model...")
    best_metric_list = [0.0]
    for iter_num in range(args.training_iter_num):
        aug_data_with_metric = []
        selected_aug_data_list = []  # selected aug data for updating model
        for split_num in range(args.training_split_num):
            print(
                f"\nData sampling & Trustworthy augmented data selection | sample times {args.aug_sample_times} | split {split_num + 1}/{args.training_split_num} | iter {iter_num + 1}/{args.training_iter_num} | current best {args.metric} {best_metric_list}")
            """Sample part of augmented graphs from total augmented graphs"""
            sample_aug_train_data_list = random.sample(total_aug_train_data_list,
                                                       len(train_data_list) * args.aug_sample_times * 3)
            sample_aug_train_dataloader = DataLoader(sample_aug_train_data_list, batch_size=args.batch_size,
                                                     shuffle=True)
            """LABEL AUGMENTED GRAPH DATA"""
            # (ONLY keep positive nodes labelling 1)
            print("Select trustworthy augmented data by best model...")
            """LOAD BEST MODEL"""
            model = base_model.MetaLearner(args).to(args.device)
            model_save_path = f'./{args.dataset_name}_models/{args.dataset_name}_{args.dataset_id}_checkpoint.pth'
            model = load_model(model, model_save_path)
            labeled_aug_train_data_list = select_trustworthy_aug_data_by_expert(model,
                                                                                sample_aug_train_dataloader,
                                                                                args.trustworthy_bound, args.device)

            print("#" * 30)
            print("Original data stat:")
            stat_label_num(origin_train_dataloader)
            print("#" * 30)
            print("Aug data stat after min:")
            stat_label_num(labeled_aug_train_data_list)
            print("#" * 30)

            max_metric = main_model_train(origin_train_dataloader, labeled_aug_train_data_list, args,
                                          origin_val_dataloader, best_metric_list, eval=True)
            """augmented data which is used to train model, combining metric of trained model on validation set"""
            print(
                f"Max {args.metric} score: {max_metric} for split {split_num + 1}/{args.training_split_num} and iteration {iter_num + 1}/{args.training_iter_num}")
            aug_data_with_metric.append((max_metric, labeled_aug_train_data_list))
        aug_data_with_metric.sort(key=lambda x: x[0], reverse=True)
        """Select top k best aug data"""
        for x in aug_data_with_metric[:args.top_k_in_split]:
            selected_aug_data_list.extend(x[1])
        print("\n" + "#" * 40)
        print("Training model with best augmented data in current iteration...")
        main_model_train(origin_train_dataloader, selected_aug_data_list, args, origin_val_dataloader, best_metric_list,
                         eval=True)
