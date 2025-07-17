import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import base_model
from args_config import get_args
from dataloader import load_dataset
from utils import set_random_seed, save_model, count_label_dist_per_file, val, test, pretrain, load_model


def main_model_pretrain():
    set_random_seed()
    args = get_args()
    print(args)
    train_dataloader, val_dataloader, test_dataloader = load_dataset(args)
    print("train dataloader:")
    count_label_dist_per_file(train_dataloader)
    print("\ntest dataloader:")
    count_label_dist_per_file(test_dataloader)

    model = base_model.MetaLearner(args).to(args.device)

    criterion = nn.BCEWithLogitsLoss().to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    """TRAIN"""
    best_metric = 0.0
    teacher_model_save_path = f'./{args.dataset_name}_models/{args.dataset_name}_{args.dataset_id}_checkpoint.pth'
    for epoch in range(args.num_epochs):
        print(f"Model pretraining for epoch {epoch + 1}/{args.num_epochs}")
        pretrain(model, train_dataloader, criterion, optimizer)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate for epoch {epoch + 1}: {current_lr}")
        scheduler.step()

        """VALIDATION"""
        if (epoch + 1) % 1 == 0:
            print(f"Evaluating for epoch {epoch + 1}")
            f1, prauc, best_f1 = val(model, val_dataloader, criterion, args)
            if args.metric == 'f1':
                if f1 > best_metric:
                    print(f"Save model with current best {args.metric} {f1} against {best_metric}")
                    best_metric = f1
                    """Save model if best on f1"""
                    save_model(model, teacher_model_save_path)
            elif args.metric == 'prauc':
                if prauc > best_metric:
                    print(f"Save model with current best {args.metric} {prauc} against {best_metric}")
                    best_metric = prauc
                    """Save model if best on prauc"""
                    save_model(model, teacher_model_save_path)
            elif args.metric == 'bestf1':
                if best_f1 > best_metric:
                    print(f"Save model with current best {args.metric} {best_f1} against {best_metric}")
                    best_metric = best_f1
                    """Save model if best on bestf1"""
                    save_model(model, teacher_model_save_path)
            else:
                raise ValueError(f"Unknown metric: {args.metric}!")

    """TEST"""
    print("Start to test...")
    model = base_model.MetaLearner(args).to(args.device)
    model = load_model(model, teacher_model_save_path)
    test(model, test_dataloader, args)


if __name__ == "__main__":
    main_model_pretrain()
