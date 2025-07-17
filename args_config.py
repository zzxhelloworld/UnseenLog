import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration")

    # General setting
    parser.add_argument("--dataset_name", type=str, default="novel", help="Name of the dataset")
    parser.add_argument("--dataset_id", type=int, default=22, help="ID of the dataset")
    parser.add_argument("--feature_dim", type=int, default=499, help="Feature dimension of the dataset (forum: 247; halo: 484; novel: 499)")
    parser.add_argument("--most_n_same_graph_structure", type=int, default=0,
                        help="Limit on the number of graphs with the same structure (0 for all)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the dataloader")
    parser.add_argument("--device", type=str, default="cuda", help="Device for experiments")

    # Model pretraining
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--step_size", type=int, default=10, help="Step size of learning rate reduction")
    parser.add_argument("--gamma", type=float, default=0.8, help="Ratio of learning rate reduction")
    parser.add_argument("--gnn_type", type=str, default="Transformer", help="Type of the GNN (GCN GAT GIN Transformer)")
    parser.add_argument("--gnn_hidden_dim", type=int, default=128, help="Hidden dimension size of the GNN")
    parser.add_argument("--gnn_head_num", type=int, default=1, help="Number of heads in transformer-based GNN")
    parser.add_argument("--gnn_num_layers", type=int, default=2, help="Number of layers of the GNN")
    parser.add_argument("--mlp_hidden_units", type=lambda s: [int(item) for item in s.split(',')], default=[256, 128, 64, 32], help="Hidden layer dimension of MLP")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for predictions")

    # Augmentation parameters
    parser.add_argument("--aug_total_times", type=int, default=20, help="Total times of each augmentation method for each graph data as sampling pool")
    parser.add_argument("--aug_sample_times", type=int, default=2, help="Single sample times for initializing student model")
    parser.add_argument("--noise_std", type=float, default=0.3, help="Noise level of Gaussian noise ranges from 0 to 1")
    parser.add_argument("--drop_ratio", type=float, default=0.3, help="The ratio of dropping off edges for augmentation")
    parser.add_argument("--add_ratio", type=float, default=0.3, help="The ratio of adding edges for augmentation")

    # Trustworthy selection from augmentation data with minmax
    parser.add_argument("--trustworthy_bound", type=float, default=0.8, help="The lower bound of trustworthy confidence for an augmented anomalous graph node (min)")
    parser.add_argument("--topk_from_trustworthy", type=float, default=0.8, help="Select top k distant aug data from real data in trustworthy aug data (max)")

    # Performance metric
    parser.add_argument("--metric", type=str, default="bestf1", help="Performance metric of evaluating models (f1/prauc/bestf1)")

    # Model training
    parser.add_argument("--training_lr", type=float, default=5e-4, help="Learning rate for training model")
    parser.add_argument("--training_num_epochs", type=int, default=10, help="Number of training epochs for training model")
    parser.add_argument("--training_batch_size", type=int, default=256, help="Batch size for the dataloader")
    parser.add_argument("--aug_loss_weight", type=float, default=0.5, help="Weight of augmentation supervised loss for training model")
    parser.add_argument("--training_from_random", type=bool, default=False, help="Initializing model from lastest model or from randomness")
    parser.add_argument("--training_iter_num", type=int, default=3, help="The number of iteration for training model")
    parser.add_argument("--training_split_num", type=int, default=3, help="The number of split for training model")
    parser.add_argument("--top_k_in_split", type=int, default=2, help="Pick top k splits of metric to train model for current iteration")
    parser.add_argument("--training_early_stop", type=bool, default=False, help="Early stop model training in the current round if getting best model")

    args = parser.parse_args()
    return args
