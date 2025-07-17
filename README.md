# Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This repository contains various artifacts, such as source code, experimental results, and other materials, that supplement our work on **Unseen Log Anomaly Detection from System Logs**. This work proposes a novel framework for code file anomaly detection from system logs, UnseenLog. The framework of UnseenLog is listed below.
![UnseenLog framework diagram](https://github.com/zzxhelloworld/UnseenLog/blob/main/Artifact/framework.jpg)
# Preliminaries: Getting ready for experiments
>  Software requirements
>>python 3.10+; pytorch 2.1.2+; torch-geometric 2.4.0+
# How to run code from the command line
1. Clone the repository:
   ```bash
   git clone https://github.com/zzxhelloworld/UnseenLog.git
   ```
2. Navigate to the project directory:
   ```bash
   cd <your_project_directory>
   ```
3. Run code from the command line for experiments:
   ```bash
   python3 --dataset_name <dataset> --dataset_id <split id> --device <device name> --num_epochs <training epoch> --gnn_type <GNN encoder> main.py
   ```

   

