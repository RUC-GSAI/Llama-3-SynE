# encoding: utf-8
"""
This script processes and saves Hugging Face datasets based on provided configurations.

Functions:
    main(timestamp_lst, tokenizer_path, model_max_length, num_workers, min_text_length, root_dir, show_case):
        Main function to process and save datasets.
        
    parse_args():
        Parses command-line arguments.

Arguments:
    --timestamp_lst (str): Comma-separated list of timestamps.
    --tokenizer_path (str): Path to the tokenizer. Default is "meta-llama/Meta-Llama-3-8B".
    --model_max_length (int): Maximum length of the model. Default is 8192.
    --num_workers (int): Number of workers. Default is 32.
    --min_text_length (int): Minimum text length to filter. Default is 20.
    --root_dir (str): Root directory.
    --show_case (bool): Whether to show the first case.

Environment Variables:
    TMPDIR: Temporary directory path.
    HF_DATASETS_CACHE: Hugging Face datasets cache directory path.
    HF_HOME: Hugging Face home directory path.

Usage:
    Run the script with the required arguments to process and save datasets.
"""

import os

# Set the environment variable to use the cache
SAVE_PATH = "~/.cache"
os.environ["TMPDIR"] = os.path.join(SAVE_PATH, "tmp")
os.environ["HF_DATASETS_CACHE"] = os.path.join(SAVE_PATH, "hf_datasets_cache")
os.environ["HF_HOME"] = os.path.join(SAVE_PATH, "hf_home")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
import json
import datasets
from transformers import AutoTokenizer
import argparse


def main(
    timestamp_lst,
    tokenizer_path,
    model_max_length,
    num_workers,
    min_text_length,
    root_dir,
    show_case,
):
    # timestamp
    timestamp_lst = timestamp_lst.split(",")

    # tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"

    # model name
    model_name = tokenizer_path.split("/")[-1]

    # save train circle info
    train_circle_dir = os.path.join(
        root_dir, f"train_info/{model_name}/train_circle_info"
    )

    def group_texts(examples):
        processed_data = []
        tmp = []
        for example in examples["input_ids"]:
            if len(example) < min_text_length:
                continue
            example.append(tokenizer.eos_token_id)
            tmp.extend(example)
            while len(tmp) >= model_max_length:
                processed_data.append(tmp[:model_max_length])
                tmp = tmp[model_max_length:]
        return {"input_ids": processed_data}

    for timestamp in timestamp_lst:

        train_circle_info_file_path = os.path.join(
            train_circle_dir, f"{timestamp}.json"
        )

        print(f"Reading train circle information from {train_circle_info_file_path}")
        with open(train_circle_info_file_path, "r", encoding="utf-8") as f:
            data_folder2token_ids_paths = json.load(f)["Token ID Files Information"]

        for data_folder, info in data_folder2token_ids_paths.items():

            print(f"Processing {data_folder}")

            info_lst = [
                (
                    i["group index"],
                    i["huggingface datasets directory"],
                    i["token id file paths"],
                    i["token num(B)"],
                )
                for i in info
            ]

            for idx, hf_datasets_dir, data_files, token_num in info_lst:

                print(
                    f"Saving {data_folder} | group {idx} | {token_num}B tokens | {len(data_files)} files | to {hf_datasets_dir}"
                )
                os.makedirs(hf_datasets_dir, exist_ok=True)

                # load dataset
                raw_train_dataset = datasets.load_dataset(
                    "json",
                    data_files=data_files,
                    split="train",
                )

                print(raw_train_dataset)
                print("=========" * 9)
                if show_case:
                    print(raw_train_dataset[0]["input_ids"])
                    print("=========" * 9)
                    print(len(raw_train_dataset[0]["input_ids"]))
                    print("=========" * 9)

                # group texts
                print(
                    f"Grouping texts with sequence length {model_max_length}. Filter texts shorter than {min_text_length}"
                )
                if len(raw_train_dataset) < 1000 * num_workers:
                    total_data = [sample["input_ids"] for sample in raw_train_dataset]
                    total_data = group_texts({"input_ids": total_data})
                    if len(total_data["input_ids"]) == 0:
                        os.removedirs(hf_datasets_dir)
                        continue
                    train_dataset = datasets.Dataset.from_dict(total_data)
                    train_dataset = train_dataset.cast_column(
                        "input_ids", datasets.features.Sequence(datasets.Value("int64"))
                    )
                else:
                    train_dataset = raw_train_dataset.map(
                        group_texts,
                        batched=True,
                        num_proc=num_workers,
                        desc=f"Grouping texts with sequence length {model_max_length}",
                    )
                print(train_dataset)
                print("=========" * 9)
                if show_case:
                    print(train_dataset[0]["input_ids"])
                    print("=========" * 9)
                    print(len(train_dataset[0]["input_ids"]))
                    print("=========" * 9)
                    print(tokenizer.decode(train_dataset[0]["input_ids"]))
                    print("=========" * 9)

                train_dataset.save_to_disk(hf_datasets_dir)
                print(f"Dataset saved to {hf_datasets_dir}")

                del raw_train_dataset
                del train_dataset

    print("Finished!")


def parse_args():
    parser = argparse.ArgumentParser(description="Process and save HF datasets.")
    parser.add_argument(
        "--timestamp_lst",
        type=str,
        required=True,
        help="Comma-separated list of timestamps.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        default="meta-llama/Meta-Llama-3-8B",
        help="Path to the tokenizer",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        required=True,
        default=8192,
        help="Maximum length of the model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        default=32,
        help="Number of workers.",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        required=True,
        default=20,
        help="Minimum text length to filter.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory.",
    )
    parser.add_argument(
        "--show_case",
        action="store_true",
        help="Whether to show the first case.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.timestamp_lst,
        args.tokenizer_path,
        args.model_max_length,
        args.num_workers,
        args.min_text_length,
        args.root_dir,
        args.show_case,
    )
