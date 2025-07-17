import torch
from torch import nn
from torch_geometric.loader import DataLoader

import base_model
from utils import load_model, save_model, val


def select_topk_distant_embeddings_from_aug(real_embeddings, aug_embeddings, k):
    """
    Select the top-k most distant augmented data embeddings.

    Args:
    - real_embeddings (torch.Tensor): Embeddings of real data, shape [N, D].
    - aug_embeddings (torch.Tensor): Embeddings of augmented data, shape [M, D].
    - k (int): The number of distant augmented data to select.

    Returns:
    - torch.Tensor: Filtered augmented embeddings.
    - torch.Tensor: Mask of selected augmented data indices.
    """
    with torch.no_grad():
        distances = torch.cdist(aug_embeddings, real_embeddings, p=2)

        min_distances, _ = torch.min(distances, dim=1)

        topk_indices = torch.topk(min_distances, k=k, largest=True).indices

        filtered_aug_embeddings = aug_embeddings[topk_indices]

    return filtered_aug_embeddings, topk_indices


def single_model_train(model, origin_dataloader, aug_dataloader, cro_ent, optimizer, args):
    model.train()
    sum_loss = 0.0
    total_num_after_max = 0
    for origin_batch, aug_batch in zip(origin_dataloader, aug_dataloader):
        origin_batch = origin_batch.to(args.device)
        """supervised loss with data of ground-truth labels"""
        origin_graph_emb = model.gnn(origin_batch.x, origin_batch.edge_index)
        stu_y_logits = model.classifier(origin_graph_emb).squeeze(-1)
        ce_loss = cro_ent(stu_y_logits, origin_batch.y.float())

        aug_batch = aug_batch.to(args.device)
        mask = (aug_batch.y != -1)  # filter data which pseudo label is -1
        aug_graph_emb = model.gnn(aug_batch.x, aug_batch.edge_index)
        aug_graph_emb = aug_graph_emb[mask]  # trustworthy (min)
        k = int(args.topk_from_trustworthy * len(aug_graph_emb))
        topk_aug_graph_emb, topk_mask = select_topk_distant_embeddings_from_aug(origin_graph_emb, aug_graph_emb, k=k)
        total_num_after_max += topk_aug_graph_emb.size(0)
        stu_y_aug_logits = model.classifier(topk_aug_graph_emb).squeeze(-1)
        """supervised loss with augmented data of trustworthy pseudo labels"""
        y_aug_pseudo_label = aug_batch.y[mask][topk_mask]
        ce_aug_loss = cro_ent(stu_y_aug_logits, y_aug_pseudo_label.float())

        """Total loss weight:5-5"""
        total_loss = (1 - args.aug_loss_weight) * ce_loss + args.aug_loss_weight * ce_aug_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()  # update student model params
        sum_loss += total_loss.item()
    avg_loss = sum_loss / len(origin_dataloader)  # avg loss
    print("Average loss for model training: ", avg_loss)
    return total_num_after_max


def main_model_train(origin_train_dataloader, aug_data_list, args, origin_val_dataloader, best_metric_list,
                     eval):
    model = base_model.MetaLearner(args).to(args.device)
    model_save_path = f'./{args.dataset_name}_models/{args.dataset_name}_{args.dataset_id}_checkpoint.pth'
    if args.training_from_random:  # Initialize model from random params
        pass
    else:  # Initialize model from best model
        model = load_model(model, model_save_path)

    cro_ent = nn.BCEWithLogitsLoss().to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.training_lr)

    metric_list = []
    sum_aug_data_num_after_max = 0
    for epoch in range(args.training_num_epochs):
        """sample some of augmented data for training model"""
        batch_num = len(origin_train_dataloader)
        aug_batch_size = len(aug_data_list) // batch_num
        aug_dataloader = DataLoader(aug_data_list, batch_size=aug_batch_size, shuffle=True)

        print(f"Model training for epoch {epoch + 1}/{args.training_num_epochs}")
        sum_aug_data_num_after_max += single_model_train(model, origin_train_dataloader, aug_dataloader, cro_ent, optimizer, args)

        f1, prauc, best_f1 = val(model, origin_val_dataloader, cro_ent, args)
        if args.metric == 'f1':
            metric_list.append(f1)
        elif args.metric == 'prauc':
            metric_list.append(prauc)
        elif args.metric == 'bestf1':
            metric_list.append(best_f1)
        else:
            raise ValueError(f"Unknown metric: {args.metric}!")
        if eval:
            if args.metric == 'f1':
                if f1 > max(best_metric_list):
                    print(f"Save model with current best {args.metric} {f1} against {max(best_metric_list)}")
                    best_metric_list.append(f1)
                    """Save model if best on f1"""
                    save_model(model, model_save_path)
                    if args.training_early_stop:
                        print("Early stop in the round due to get a best model!")
                        break
            elif args.metric == 'prauc':
                if prauc > max(best_metric_list):
                    print(f"Save model with current best {args.metric} {prauc} against {max(best_metric_list)}")
                    best_metric_list.append(prauc)
                    """Save model if best on prauc"""
                    save_model(model, model_save_path)
                    if args.training_early_stop:
                        print("Early stop in the round due to get a best model!")
                        break
            elif args.metric == 'bestf1':
                if best_f1 > max(best_metric_list):
                    print(
                        f"Save model with current best {args.metric} {best_f1} against {max(best_metric_list)}")
                    best_metric_list.append(best_f1)
                    """Save model if best on bestf1"""
                    save_model(model, model_save_path)
                    if args.training_early_stop:
                        print("Early stop in the round due to get a best model!")
                        break
            else:
                raise ValueError(f"Unknown metric: {args.metric}!")
    print("#" * 30)
    print("Aug data stat after max:")
    print(f"1: {sum_aug_data_num_after_max / args.training_num_epochs:.2f}")
    print("#" * 30)
    return max(metric_list)
