# encoding: utf-8
"""
split_data.py

This script processes datasets by splitting large JSONL files into smaller parts based on a maximum token limit.

Functions:
    parse_args(): Parses command line arguments.
    main(): Main function to process datasets.
    write_to_file(): Writes the current batch of data to a file.
    add_data(data): Adds data to the current batch.
    load_data(fin): Loads data from a file.

Command Line Arguments:
    --father_datasets: Comma-separated list of dataset paths.

Usage:
    python split_data.py --father_datasets <dataset_paths>

Example:
    python split_data.py --father_datasets /path/to/dataset1,/path/to/dataset2
"""

import os
import json
import glob
import pathlib
import numpy as np
from IPython import embed
from tqdm import tqdm
import argparse

# Define the maximum number of tokens allowed in a single file
MAX_TOKEN = int(0.01 * 1000 * 1000 * 1000)


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process some datasets.")
    parser.add_argument(
        "--father_datasets",
        type=str,
        required=True,
        help="Comma-separated list of dataset paths",
    )
    return parser.parse_args()


# Process datasets
def main():
    args = parse_args()
    father_datasets = args.father_datasets.split(",")

    for fd in father_datasets:
        datasets = os.listdir(fd)
        for dataset_name in tqdm(datasets):
            raw_src_folder = os.path.join(fd, dataset_name)
            print("Processing {} ...".format(raw_src_folder))
            embed()
            folder2file = {}
            try:
                # Traverse the directory to find JSONL files
                for root_dir, _, files in os.walk(raw_src_folder, topdown=False):
                    for fp in files:
                        if "splited_part" in fp:
                            continue
                        if not fp.endswith(".jsonl"):
                            continue
                        if root_dir not in folder2file:
                            folder2file[root_dir] = []
                        folder2file[root_dir].append(os.path.join(root_dir, fp))

            except FileNotFoundError:
                print("Error Dataset: {}".format(dataset_name))
                continue
            except NotADirectoryError:
                print("Error Dataset: {}".format(dataset_name))
                continue

            if len(folder2file) == 0:
                print("Error Dataset: {}".format(dataset_name))
                continue

            for src_folder, src_files in folder2file.items():
                all_data = []
                tokens_num = []
                num_tokens = 0
                cur_idx = 0

                # Write data to a file
                def write_to_file():
                    nonlocal all_data, num_tokens, cur_idx, tokens_num
                    tgt_path = os.path.join(
                        src_folder, "splited_part-{}.jsonl".format(cur_idx)
                    )
                    tokens_num_tgt_path = os.path.join(
                        src_folder,
                        "splited_part-{}-tokens_{}.jsonl".format(cur_idx, num_tokens),
                    )
                    print(tgt_path)
                    with open(tgt_path, "w") as fout:
                        for tmp_data in all_data:
                            fout.write(json.dumps(tmp_data, ensure_ascii=False) + "\n")
                    with open(tokens_num_tgt_path, "w") as fout:
                        for tmp_data in tokens_num:
                            fout.write(str(tmp_data) + "\n")
                    num_tokens = 0
                    cur_idx = cur_idx + 1
                    all_data = []
                    tokens_num = []

                # Add data to the current batch
                def add_data(data):
                    nonlocal all_data, num_tokens, cur_idx
                    all_data.append(data[0])
                    tokens_num.append(data[1])
                    num_tokens = num_tokens + data[1]
                    if num_tokens > MAX_TOKEN:
                        write_to_file()

                # Load data from a file
                def load_data(fin):
                    data = fin.readline()
                    if not data:
                        return None
                    else:
                        json_data = json.loads(data)
                        new_data = {"input_ids": json_data["input_ids"]}
                        return (new_data, len(json_data["input_ids"]))

                src_fin = []
                src_data = []
                for fp in src_files:
                    if "splited_part" in fp:
                        continue
                    fin = open(os.path.join(src_folder, fp))
                    src_fin.append(fin)
                    src_data.append(load_data(fin))

                # Process the data files
                while True:
                    idx = None
                    for i in range(len(src_data)):
                        if src_data[i] is None:
                            continue
                        if idx is None:
                            idx = i
                            break

                    if idx is None:
                        break

                    add_data(src_data[idx])
                    src_data[idx] = load_data(src_fin[idx])

                if len(all_data) > 0:
                    write_to_file()


if __name__ == "__main__":
    main()
