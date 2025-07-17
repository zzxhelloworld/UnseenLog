from args_config import get_args
from select_aug_graph_data_and_train_iteratively import get_labeled_aug_data_and_train_model
from utils import set_random_seed


def main_model_train_framework():
    set_random_seed()
    args = get_args()
    print(args)

    print(f"Get best augmented data | Train model by iterations...")
    get_labeled_aug_data_and_train_model(args)


if __name__ == '__main__':
    main_model_train_framework()
