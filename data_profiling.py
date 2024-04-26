# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Callable, Dict
import numpy as np
import time
import torch
from torch.utils.data import DataLoader, IterableDataset

# from torchtitan.datasets.tokenizer import Tokenizer
# from torchtitan.logging_utils import logger

import datasets
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


# from datasets.distributed import split_dataset_by_node



class HuggingFaceDataset(IterableDataset):
    """
    from https://github.com/pytorch/torchtitan/blob/42549a9205cc325cef80309d36494007bfb6bb00/torchtitan/datasets/hf_datasets.py#L114
    PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently support the c4 dataset and a subset of it:
    c4_mini (45K training entries)
    c4 (177M training entries - this dataset is streamed due to the size)

    >> c4 (EN) <<:
    c4 cleaned, English version
    Data input format (c4):
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at ...',
    'timestamp': '2019-04-25T12:57:54Z'
    }

    Example use (c4):
    >>> ds = HuggingFaceDataset(dataset_name="c4", dataset_path=None, tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Callable,
        seq_len: int = 2048,
        infinite: bool = False,
    ) -> None:

        ds = load_dataset(dataset_name, cache_dir=dataset_path)["train"]

        self.dataset_name = dataset_name
        self._data = ds
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        all_tokens: List[int] = []

        while True:
            for sample in iter(self._data):
                sample_text = sample["text"]
                sample_tokens = self._tokenizer.encode(sample_text)
                all_tokens.extend(sample_tokens)

                while len(all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    all_tokens = all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label
            if not self.infinite:
                print(f"Dataset {self.dataset_name} has run out of data.")
                break
            else:
                print(
                    f"Dataset {self.dataset_name} is being re-looped. "
                    "Loss related metrics might be misleading."
                )


def clm_process(
    raw_dataset,
    tokenizer: Callable,
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
):
    """Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`, where chunks overlap by a single token."""
    # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439

    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)

    train_dataset = raw_dataset.map(
        _tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=not dataset_overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
    )
    return train_dataset


dataset_name = "roneneldan/TinyStories"
dataset_path = "/raid/xind/datasets/tinystories"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
seq_len = 1024
infinite = True
batch_size = 1024
num_workers = 1


onthefly_ds = HuggingFaceDataset(
        dataset_name, dataset_path, tokenizer, seq_len, infinite
    )

onethefly_dl = DataLoader(onthefly_ds, batch_size=batch_size, num_workers=num_workers)



hf_ds = load_dataset(dataset_name, cache_dir=dataset_path)["train"]
pretokenized_ds = clm_process(
        hf_ds, tokenizer, "text", 8, False, seq_len
)

pretokenized_dl = DataLoader(pretokenized_ds, batch_size=batch_size, num_workers=num_workers)


# print(next(iter(onethefly_dl))[0].shape)
# print("\n\n")
# print(torch.stack(next(iter(pretokenized_dl))['input_ids']).shape)

def profiling(dl, steps):
    start_time = time.time()

    for step_idx, data in enumerate(dl):
        if step_idx >= steps:
            break
    
    avg_time = (time.time() - start_time) / steps

    return avg_time



print("pre tokenization: %fs"%(profiling(pretokenized_dl, 100)))
print("on the fly: %fs"%(profiling(onethefly_dl, 100)))

