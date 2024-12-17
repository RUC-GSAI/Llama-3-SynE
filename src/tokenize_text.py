# encoding: utf-8
"""
This script tokenizes text data using a specified tokenizer and saves the tokenized data to a target folder.
It supports multiprocessing to speed up the tokenization process.

Functions:
    get_tgt_folder(file_path, model_name):
        Get the target folder path based on the file path and model name.
    
    tokenize_text(dataset, tgt_folder, idx, text_key, is_first):
        Tokenize text data and save it to the target folder.
    
    start_mp(dataset, is_first):
        Start multiprocessing for tokenizing text data.

Main:
    The script takes several command-line arguments:
        --tokenizer_path: Path to the tokenizer.
        --model_name: Name of the model.
        --data_path: Path to the data files.
        --num_files: Number of files to process.
        --text_key: Key to access text data in the dataset.
        --num_worker: Number of worker processes for multiprocessing.
        --skip_exist: Whether to skip existing processed files.

    The script processes each file in the specified data path, tokenizes the text data, and saves the tokenized data to the target folder.
"""

import argparse
import os
import json
import random
import pathlib
import numpy as np
import multiprocessing as mp
from tqdm import tqdm, trange
from transformers import AutoTokenizer

random.seed(45)
MAX_DATA = int(1e6)


# Get the target folder path based on the file path and model name
def get_tgt_folder(file_path, model_name):
    file_path = file_path.replace("/data", f"/{model_name}_data_ids")
    tgt_folder = file_path[: file_path.rfind(".")]
    tgt_folder = os.path.join(tgt_folder, "wo_ppl")
    if os.path.exists(tgt_folder) == True:
        is_exists = True
    else:
        is_exists = False
    pathlib.Path(tgt_folder).mkdir(parents=True, exist_ok=True)
    return tgt_folder, is_exists


# Tokenize text data and save it to the target folder
def tokenize_text(dataset, tgt_folder, idx, text_key, is_first):
    tgt_path = os.path.join(tgt_folder, "part-{}.jsonl".format(idx))
    if is_first == False:
        write_mode = "a"
    else:
        write_mode = "w"
    fout = open(tgt_path, write_mode)
    for data in tqdm(dataset, desc="Process {}".format(idx)):
        input_ids = tokenizer(data[text_key], add_special_tokens=False)["input_ids"]
        new_data = {"input_ids": input_ids}
        fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
    fout.close()


# Start multiprocessing for tokenizing text data
def start_mp(dataset, is_first):
    if len(dataset) == 0:
        return
    if isinstance(dataset, list) == False:
        return
    try:
        assert args.text_key in dataset[0]
        text_key = args.text_key
    except AssertionError:
        print("Available Keys:", dataset[0].keys())
        raise Exception("Unknown Key!")

    random.shuffle(dataset)
    part_num = args.num_worker
    slice_idx = np.linspace(0, len(dataset), part_num + 1).astype("int")
    p = mp.Pool(part_num)
    for start_id in range(part_num):
        start, end = slice_idx[start_id], slice_idx[start_id + 1]
        new_lines = dataset[start:end]
        p.apply_async(
            tokenize_text, args=(new_lines, tgt_folder, start_id, text_key, is_first)
        )
    p.close()
    p.join()
    print("All of the child processes over!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_files", type=int)
    parser.add_argument("--text_key", type=str)
    parser.add_argument("--num_worker", type=int)
    parser.add_argument("--skip_exist", type=bool, default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    for root, _, files in os.walk(args.data_path, topdown=False):
        step = 0
        random.shuffle(files)
        for fp in tqdm(files):
            file_path = os.path.join(root, fp)
            tgt_folder, is_exists = get_tgt_folder(file_path, args.model_name)
            if is_exists == True and args.skip_exist == True:
                continue

            print("Process {}".format(file_path))
            print("Target Folder: {}".format(tgt_folder))

            fin = open(file_path, "r")
            is_jsonl = False
            if file_path.endswith(".json") == True:
                try:
                    dataset = json.load(fin)
                    start_mp(dataset, True)
                    step = step + 1
                    if step >= args.num_files:
                        break
                    continue
                except json.decoder.JSONDecodeError:
                    is_jsonl = True
                    fin.close()
                    fin = open(file_path, "r")

            if file_path.endswith(".jsonl") == True or is_jsonl == True:
                is_finish = False
                is_first = True
                while True:
                    dataset = []
                    for i in trange(MAX_DATA, desc="Reading Data"):
                        tmp_data = fin.readline()
                        if not tmp_data:
                            is_finish = True
                            break
                        try:
                            tmp_data = json.loads(tmp_data)
                            dataset.append(tmp_data)
                        except json.decoder.JSONDecodeError:
                            continue

                    start_mp(dataset, is_first)
                    is_first = False
                    if is_finish == True:
                        break
            else:
                continue

            fin.close()
            step = step + 1
            if step >= args.num_files:
                break
