import time

from anomaly_infer import main_anomaly_infer
from model_train_framework import main_model_train_framework
from model_pretrain import main_model_pretrain


def measure_time(func):
    start_time = time.time()
    func()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    return elapsed_time


if __name__ == "__main__":
    time_pretrain = measure_time(main_model_pretrain)
    time_train = measure_time(main_model_train_framework)
    time_inference = measure_time(main_anomaly_infer)
    total_time = time_pretrain + time_train + time_inference
    print("#" * 40)
    print(f"Model pretrain time: {time_pretrain:.2f} hours...")
    print(f"Model train time: {time_train:.2f} hours...")
    print(f"Anomaly inference time: {time_inference:.2f} hours...")
    print(f"Total time: {total_time:.2f} hours...")
    print("#" * 40)
