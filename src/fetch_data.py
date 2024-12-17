# encoding: utf-8
"""
fetch_data.py

This script is used to fetch and organize token data for training purposes. It reads token ID files from specified directories, selects a specified number of tokens based on given ratios, and organizes the selected token data into groups for further processing.

Functions:
    main(total_token_num, cn_ratio, en_ratio, syn_ratio, root_dir, tokenizer_path)
        Main function to fetch and organize token data.

Arguments:
    --total_token_num (int): Total number of tokens to be selected.
    --cn_ratio (float): Ratio of CN tokens.
    --en_ratio (float): Ratio of EN tokens.
    --syn_ratio (float): Ratio of SYN tokens.
    --root_dir (str): Root directory for token IDs.
    --tokenizer_path (str): Path to the tokenizer.

Usage:
    python fetch_data.py --total_token_num 40 --cn_ratio 0.1 --en_ratio 0.7 --syn_ratio 0.2 --root_dir /path/to/root --tokenizer_path meta-llama/Meta-Llama-3-8B
"""

import os
import argparse
import json
import glob
from datetime import datetime

# Set the environment variable to use the cache
SAVE_PATH = "~/.cache"
os.environ["TMPDIR"] = os.path.join(SAVE_PATH, "tmp")
os.environ["HF_DATASETS_CACHE"] = os.path.join(SAVE_PATH, "hf_datasets_cache")
os.environ["HF_HOME"] = os.path.join(SAVE_PATH, "hf_home")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)


def main(
    total_token_num,
    cn_ratio,
    en_ratio,
    syn_ratio,
    root_dir,
    tokenizer_path,
):
    print(f"Start | {datetime.now().strftime('%Y%m%d_%H%M%S')}")

    cn_token_num = total_token_num * cn_ratio  # CN
    en_token_num = total_token_num * en_ratio  # EN
    syn_token_num = total_token_num * syn_ratio  # SYNTH

    cn_ratio_lst = [
        0.7,  # web-cn
        0.05,  # encyclopedia-cn
        0.2,  # book-cn
        0.05,  # qa_forum-cn
    ]
    en_ratio_lst = [
        0.4,  # web-en
        0.05,  # encyclopedia-en
        0.15,  # book-en
        0.05,  # qa_forum-en
        0.1,  # paper-en
        0.1,  # math-en
        0.15,  # code-en
    ]

    data_select_info = {
        ## CN
        "web-cn": cn_token_num * cn_ratio_lst[0],
        "encyclopedia-cn": cn_token_num * cn_ratio_lst[1],
        "book-cn": cn_token_num * cn_ratio_lst[2],
        "qa_forum-cn": cn_token_num * cn_ratio_lst[3],
        ## EN
        "web-en": en_token_num * en_ratio_lst[0],
        "encyclopedia-en": en_token_num * en_ratio_lst[1],
        "book-en": en_token_num * en_ratio_lst[2],
        "qa_forum-en": en_token_num * en_ratio_lst[3],
        "paper-en": en_token_num * en_ratio_lst[4],
        "math-en": en_token_num * en_ratio_lst[5],
        "code-en": en_token_num * en_ratio_lst[6],
        ## SYNTH
        "synthesis-en": syn_token_num,
    }
    print(data_select_info)

    # Billion
    B = 10**9

    # Model name
    model_name = tokenizer_path.split("/")[-1]

    # Token ids root directory
    token_ids_dir = os.path.join(root_dir, f"{model_name}_data_ids")

    # Time stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save trained data info
    trained_info_dir = os.path.join(
        root_dir, f"train_info/{model_name}/trained_data_info/{timestamp}"
    )
    os.makedirs(trained_info_dir, exist_ok=True)

    # Save train circle info
    train_circle_dir = os.path.join(
        root_dir, f"train_info/{model_name}/train_circle_info"
    )
    os.makedirs(train_circle_dir, exist_ok=True)

    # Hf datasets save directory
    hf_datasets_save_dir = os.path.join(root_dir, "hf_dataset", model_name, timestamp)

    # Total tokens selected
    total_tokens = 0

    # Number of selected tokens for each data folder
    data_folder2num_tokens = dict()

    # Data folder name to selected token id path group
    data_folder2token_ids_paths = dict()

    # Hf dataset save step
    hf_dataset_save_step = 3  # 3B

    # Get trained files
    trained_files = []
    for file in glob.glob(train_circle_dir + "/*.json"):
        print(f"Reading trained data info from {file}")
        with open(file, "r") as f:
            trained_file_info = json.load(f)
            path_lst = [
                paths
                for info in trained_file_info["Token ID Files Information"].values()
                for paths in [ii["token id file paths"] for ii in info]
            ]
            trained_files.extend([iii for jjj in path_lst for iii in jjj])

    trained_files = [i for i in trained_files]

    for data_folder_name, target_token_num in data_select_info.items():

        if target_token_num == 0:
            continue

        print(f"Processing {data_folder_name}")

        # Read trained data info for this data folder
        save_path_data_info = os.path.join(trained_info_dir, data_folder_name)
        os.makedirs(save_path_data_info, exist_ok=True)

        # Get untrained token id files for this data folder
        dirpath_filename_lst = [
            (dirpath, filename)
            for dirpath, dirnames, filenames in os.walk(
                os.path.join(token_ids_dir, data_folder_name)
            )
            for filename in filenames
            if filename.endswith(".jsonl")
            and "splited_part" in filename
            and "tokens_" not in filename
            and os.path.join(dirpath, filename) not in trained_files
        ]

        if len(dirpath_filename_lst) == 0:
            continue

        # Sort by file name to ensure the order of ppl, etc
        dirpath_filename_lst.sort(key=lambda x: x[1])

        # Train data info for this data folder
        token_infos = []
        token_num = 0

        # Seleceted token id file paths for this data folder
        paths_groups = []

        # Hf dataset piece info for this data folder
        file_paths_piece = []
        token_num_piece = 0

        # Read file names, select token id files and get token num
        for dir_path, file_name in dirpath_filename_lst:
            file_path = os.path.join(dir_path, file_name)

            # Get token num
            # E.g. ···/splited_part-1.jsonl => ···/splited_part-1-tokens_100073149.jsonl
            glob_files = glob.glob(
                os.path.join(dir_path, file_name.split(".")[0] + "-tokens_*.jsonl")
            )
            if len(glob_files) == 0:
                continue
            assert len(glob_files) == 1
            file_token_num = int(glob_files[0].split("_")[-1].split(".")[0]) / B
            print(f"{file_path} has {file_token_num} tokens")

            # Update train data info
            token_infos.append(
                {
                    "token_ids_path": file_path,
                    "token_num(B)": file_token_num,
                }
            )
            token_num += file_token_num
            print(
                f"Total token updated: {token_num}B/{target_token_num}B for {data_folder_name}"
            )

            # Update hf dataset piece info
            file_paths_piece.append(file_path)
            token_num_piece += file_token_num

            # Update hf dataset piece info if token num exceeds save step
            if token_num_piece > hf_dataset_save_step:
                paths_groups.append([file_paths_piece, token_num_piece])
            # Unset hf dataset piece info
            file_paths_piece = []
            token_num_piece = 0

            # Check if target token num is reached
            if token_num > target_token_num:
                break

        # Save last hf dataset piece
        if len(file_paths_piece) > 0:
            paths_groups.append([file_paths_piece, token_num_piece])

        # Save train data info
        with open(
            os.path.join(save_path_data_info, f"{timestamp}.jsonl"),
            "w",
            encoding="utf-8",
        ) as f:
            for i in token_infos:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")

        # Update global train data info
        total_tokens += token_num
        data_folder2num_tokens[data_folder_name] = token_num
        data_folder2token_ids_paths[data_folder_name] = [
            {
                "group index": idx,
                "huggingface datasets directory": os.path.join(
                    hf_datasets_save_dir,
                    f"{data_folder_name.replace('/', '<sep>')}_{idx}",
                ),
                "token num(B)": info[1],
                "token id file paths": info[0],
            }
            for idx, info in enumerate(paths_groups)
        ]

        # Print train data info for this data folder
        print(
            f"Token id files selected for {data_folder_name}:\n{data_folder2token_ids_paths[data_folder_name]}"
        )
        print(
            f"Select {len([i for j in paths_groups for i in j])} token id files for {data_folder_name}"
        )
        print(
            f"Select token num: {token_num}B/{target_token_num}B for {data_folder_name}"
        )

    # Show train circle info
    train_circle_info = {
        "Total Tokens(B)": total_tokens,
        "Total Huggingface Datasets Directory": hf_datasets_save_dir,
        "Number of Selected Tokens(B)": data_folder2num_tokens,
        "Manual Selected Tokens(B)": data_select_info,
        "Token ID Files Information": data_folder2token_ids_paths,
    }
    print(f"Train circle info:\n{train_circle_info}")

    # Save train circle info
    train_circle_info_path = os.path.join(train_circle_dir, f"{timestamp}.json")
    print(f"Save train circle info to {train_circle_info_path}")
    with open(train_circle_info_path, "w", encoding="utf-8") as f:
        json.dump(train_circle_info, f, ensure_ascii=False, indent=4)

    print(f"Finished | {timestamp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch data script")
    parser.add_argument(
        "--total_token_num",
        type=int,
        required=True,
        default=40,
        help="Total number of tokens",
    )
    parser.add_argument(
        "--cn_ratio", type=float, required=True, default=0.1, help="Ratio of CN tokens"
    )
    parser.add_argument(
        "--en_ratio", type=float, required=True, default=0.7, help="Ratio of EN tokens"
    )
    parser.add_argument(
        "--syn_ratio",
        type=float,
        required=True,
        default=0.2,
        help="Ratio of SYN tokens",
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Root directory for token ids"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        default="meta-llama/Meta-Llama-3-8B",
        help="Path to the tokenizer",
    )

    args = parser.parse_args()

    main(
        args.total_token_num,
        args.cn_ratio,
        args.en_ratio,
        args.syn_ratio,
        args.root_dir,
        args.tokenizer_path,
    )
