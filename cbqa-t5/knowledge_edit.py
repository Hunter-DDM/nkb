#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

import transformers
from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from nkb_t5 import T5ForConditionalGeneration
from transformers import T5Config, T5TokenizerFast
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    optim_group: str = field(
        default='all',
        metadata={"help": "indicate which parameter groups to be learned, could be ['all', 'ori', 'ex', 'ori-ada']"}
    )
    sec_lr: str = field(
        default='zero',
        metadata={"help": "how much learning rate the other parameters use, could be ['zero', '/10', '/100', '/1000']"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # config for NKB settings
    kb_layer: Optional[str] = field(
        default='',
        metadata={"help": "Layers to be extended with NKB, should be separated by commas, e.g., '10,11' or '11'"}
    )
    ex_size: Optional[int] = field(
        default=3072,
        metadata={"help": "The number of the extended FFN slots"}
    )
    adaptive_nkb: bool = field(
        default=False,
        metadata={"help": "Whether to add adaptive layer outside the NKB"},
    )
    dropout: Optional[float] = field(
        default=None,
        metadata={"help": "dropout rate for the model"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    question_column: Optional[str] = field(
        default="question",
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    answer_column: Optional[str] = field(
        default="answers",
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_answer_length: int = field(
        default=10,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    val_max_answer_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_answer_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_answer_length is None:
            self.val_max_answer_length = self.max_answer_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MySeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = T5Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = T5TokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # update config for NKB settings
    config.kb_layer = model_args.kb_layer
    config.ex_size = model_args.ex_size
    config.adaptive_nkb = model_args.adaptive_nkb
    if model_args.dropout is not None:
        config.dropout_rate = model_args.dropout
    # model = T5ForConditionalGeneration(
    #     config=config,
    # )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to generate and tokenize inputs and targets.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    question_column = data_args.question_column
    if question_column not in column_names:
        raise ValueError(
            f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
        )
    answer_column = data_args.answer_column
    if answer_column not in column_names:
        raise ValueError(
            f"--answer_column' value '{data_args.answer_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Temporarily set max_answer_length for training.
    max_answer_length = data_args.max_answer_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_batch(
        examples,
        question_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        answers = examples[answer_column]

        # inputs = [question.strip() for question in questions]
        inputs = ['question: ' + question.strip() for question in questions]
        # inputs = ['answer the question: ' + question.strip() for question in questions]
        # inputs = [question + '?' if not question.endswith('?') else question for question in inputs]
        # inputs = [question.lower() for question in inputs]

        targets = [answer[0].strip() if len(answer) > 0 else "" for answer in answers]
        # targets = [answer.lower() for answer in targets]
        return inputs, targets

    def preprocess_function(examples):
        inputs, targets = preprocess_batch(examples, question_column, answer_column)

        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Validation preprocessing
    def preprocess_validation_function(examples):
        inputs, targets = preprocess_batch(examples, question_column, answer_column)

        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # ============== knowledge editing ==============

    import json
    import random
    import copy

    model = model.cuda()
    model.requires_grad_(False)
    # dataset = train_dataset
    dataset = eval_dataset

    # single_token_examples = []
    # for i in list(range(len(dataset))):
    #     sample = dataset[i]
    #     if len(sample["labels"]) > 2:
    #         continue
    #     single_token_examples.append(i)
    # print("single_token_examples", len(single_token_examples), '/', len(dataset))

    # w2r_indices = []
    # r2w_indices = []

    # for i, index in enumerate(single_token_examples):
    #     sample = dataset[index]
    #     if 'offset_mapping' in sample:
    #         del sample['offset_mapping']
    #     if 'example_id' in sample:
    #         del sample['example_id']
    #     question = tokenizer.decode(sample["input_ids"])
    #     answer_token_ids = sample["labels"][:-1]
    #     answer = tokenizer.decode(answer_token_ids)
    #     answer_len = len(sample["labels"])
    #     for key in sample.keys():
    #         sample[key] = torch.LongTensor([sample[key]]).cuda()
    #     del sample['labels']
    #     results = model.generate(**sample)
    #     g_answer_token_ids = results[0].data.tolist()[1:-1]
    #     g_answer = tokenizer.decode(g_answer_token_ids)
    #     if len(g_answer_token_ids) > 1:
    #         continue
    #     print('============' * 5, index)
    #     print('question|', question)
    #     print('answer|', answer)
    #     print('answer_ids|', answer_token_ids, '|', answer_len)
    #     print('generated_answer_ids|', g_answer_token_ids)
    #     print('generated_answer|', g_answer)
    #     if g_answer == answer:
    #         r2w_indices.append(index)
    #     else:
    #         w2r_indices.append(index)
    #     if i % 20 == 0:
    #         print(f'{i}/{len(single_token_examples)} examples done')

    # print('r2w_indices number:', len(r2w_indices))
    # print('w2r_indices number:', len(w2r_indices))

    # with open(f'tmp_data/knowledge_editing/r2w_indices.json', 'w', encoding='utf-8') as f:
    #     json.dump(r2w_indices, f, indent=2)
    # with open(f'tmp_data/knowledge_editing/w2r_indices.json', 'w', encoding='utf-8') as f:
    #     json.dump(w2r_indices, f, indent=2)

    with open(f'tmp_data/knowledge_editing/r2w_indices.json', 'r', encoding='utf-8') as f:
        r2w_indices = json.load(f)
    with open(f'tmp_data/knowledge_editing/w2r_indices.json', 'r', encoding='utf-8') as f:
        w2r_indices = json.load(f)
    single_indices = r2w_indices + w2r_indices

    # ============== correct erroneous knowledge ==============

    def knowledge_updating_w2r(index):
        sample = dataset[index]
        if 'offset_mapping' in sample:
            del sample['offset_mapping']
        if 'example_id' in sample:
            del sample['example_id']
        question = tokenizer.decode(sample["input_ids"])
        answer_token_ids = sample["labels"][:-1]
        answer = tokenizer.decode(answer_token_ids)
        answer_len = len(sample["labels"])
        with open(f'tmp_data/knowledge_editing/tmp_weights.txt', 'w', encoding='utf-8') as f:
            print(f'{question}', file=f)
            print(f'{answer}', file=f)
            print(f'{answer_token_ids}', file=f)
            print(f'{answer_len}', file=f)
        for key in sample.keys():
            sample[key] = torch.LongTensor([sample[key]]).cuda()
        del sample['labels']
        results = model.generate(**sample)
        g_answer_token_ids = results[0].data.tolist()[1:-1]
        g_answer = tokenizer.decode(g_answer_token_ids)
        print('index|', index)
        print('question|', question)
        print('answer|', answer)
        print('answer_ids|', answer_token_ids, '|', answer_len)
        print('generated_answer_ids|', g_answer_token_ids)
        print('generated_answer|', g_answer)

        o_embedding = model.lm_head.weight
        nkb_values = model.decoder.block[11].layer[2].DenseReluDense.wo_ex.weight
        all_prob = o_embedding.matmul(nkb_values).t().softmax(dim=1)  # n_value, n_vocab

        to_update_pos = []
        with open(f'tmp_data/knowledge_editing/tmp_weights.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            weights = [float(x) for x in lines[4].split()]

            # update high-weight value
            weights_indices = [[float(x), i] for i, x in enumerate(lines[4].split())]
            weights_indices.sort(key=lambda x:-x[0])
            to_update_pos.append(weights_indices[0][1])

            # # update hard corresponding value
            # for i, weight in enumerate(weights):
            #     if weight <= 1e-10:
            #         continue
            #     topk, topk_pos = all_prob[i].topk(1)
            #     if topk_pos[0] != g_answer_token_ids[0]:
            #         continue
            #     to_update_pos.append(i)
            #     # print(i, weight, tokenizer.decode(topk_pos))

            # # update soft corresponding value
            # affinity = all_prob[:, g_answer_token_ids[0]]  # n_value
            # topk, topk_pos = affinity.topk(3072)
            # topk_pos = topk_pos.tolist()
            # for pos in topk_pos:
            #     if weights[pos] > 10:
            #         to_update_pos.append(pos)
            #         break

        print('to_update_pos|', len(to_update_pos), ':', to_update_pos)

        ori_token_id = g_answer_token_ids[0]
        new_token_id = answer_token_ids[0]
        ori_emb = o_embedding[ori_token_id, :]
        new_emb = o_embedding[new_token_id, :]
        scaling_factor = 0.09
        # editing
        for i, pos in enumerate(to_update_pos):
            nkb_values[:, pos] += new_emb * scaling_factor - ori_emb * scaling_factor

        results = model.generate(**sample)
        g_answer_token_ids = results[0].data.tolist()[1:-1]
        g_answer_token_ids = g_answer_token_ids[0:1]
        g_answer = tokenizer.decode(g_answer_token_ids)
        print('(after editing) generated_answer_ids|', g_answer_token_ids)
        print('(after editing) generated_answer|', g_answer)

        other_sample_indices = random.sample(list(range(len(dataset))), 5)

        other_answers_new = []
        for other_sample_index in other_sample_indices:
            other_sample = dataset[other_sample_index]
            if 'labels' in other_sample:
                del other_sample['labels']
            if 'offset_mapping' in other_sample:
                del other_sample['offset_mapping']
            if 'example_id' in other_sample:
                del other_sample['example_id']
            for key in other_sample.keys():
                other_sample[key] = torch.LongTensor([other_sample[key]]).cuda()
            other_results = model.generate(**other_sample)
            other_g_answer_token_ids = other_results[0].data.tolist()[1:-1]
            other_g_answer = tokenizer.decode(other_g_answer_token_ids)
            other_answers_new.append(other_g_answer)

        # restoring
        for i, pos in enumerate(to_update_pos):
            nkb_values[:, pos] += ori_emb * scaling_factor - new_emb * scaling_factor

        other_answers_ori = []
        for other_sample_index in other_sample_indices:
            other_sample = dataset[other_sample_index]
            if 'labels' in other_sample:
                del other_sample['labels']
            if 'offset_mapping' in other_sample:
                del other_sample['offset_mapping']
            if 'example_id' in other_sample:
                del other_sample['example_id']
            for key in other_sample.keys():
                other_sample[key] = torch.LongTensor([other_sample[key]]).cuda()
            other_results = model.generate(**other_sample)
            other_g_answer_token_ids = other_results[0].data.tolist()[1:-1]
            other_g_answer = tokenizer.decode(other_g_answer_token_ids)
            other_answers_ori.append(other_g_answer)

        succ = int(g_answer == answer)
        change = (5 - len(set(other_answers_ori).intersection(set(other_answers_new)))) / 5

        return [succ, change]

    succ_cnt = 0
    change_rate = []
    # w2r_indices = w2r_indices[:10]
    for index in w2r_indices:
        print('============' * 5, f'Editing {index}-th example')
        succ, change = knowledge_updating_w2r(index)
        print(f'**** Succ: {succ}, Change: {change}')
        succ_cnt += succ
        change_rate.append(change)
    print(f'success rate: {succ_cnt/len(w2r_indices)*100:.4}% ({succ_cnt}/{len(w2r_indices)})')
    print(f'change rate: {sum(change_rate)/len(change_rate)*100:.4}%')

    # # ============== erase knowledge ==============

    # def knowledge_erasing(index):
    #     sample = dataset[index]
    #     if 'offset_mapping' in sample:
    #         del sample['offset_mapping']
    #     if 'example_id' in sample:
    #         del sample['example_id']
    #     question = tokenizer.decode(sample["input_ids"])
    #     answer_token_ids = sample["labels"][:-1]
    #     answer = tokenizer.decode(answer_token_ids)
    #     answer_len = len(sample["labels"])
    #     with open(f'tmp_data/knowledge_editing/tmp_weights.txt', 'w', encoding='utf-8') as f:
    #         print(f'{question}', file=f)
    #         print(f'{answer}', file=f)
    #         print(f'{answer_token_ids}', file=f)
    #         print(f'{answer_len}', file=f)
    #     for key in sample.keys():
    #         sample[key] = torch.LongTensor([sample[key]]).cuda()
    #     del sample['labels']
    #     results = model.generate(**sample)
    #     g_answer_token_ids = results[0].data.tolist()[1:-1]
    #     g_answer = tokenizer.decode(g_answer_token_ids)
    #     print('index|', index)
    #     print('question|', question)
    #     print('answer|', answer)
    #     print('answer_ids|', answer_token_ids, '|', answer_len)
    #     print('generated_answer_ids|', g_answer_token_ids)
    #     print('generated_answer|', g_answer)

    #     assert answer == g_answer

    #     o_embedding = model.lm_head.weight
    #     nkb_values = model.decoder.block[11].layer[2].DenseReluDense.wo_ex.weight
    #     all_prob = o_embedding.matmul(nkb_values).t().softmax(dim=1)  # n_value, n_vocab

    #     to_erase_pos = []
    #     with open(f'tmp_data/knowledge_editing/tmp_weights.txt', 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         weights = [float(x) for x in lines[4].split()]

    #         # # update high-weight value
    #         # weights_indices = [[float(x), i] for i, x in enumerate(lines[4].split())]
    #         # weights_indices.sort(key=lambda x:-x[0])
    #         # to_erase_pos.append(weights_indices[0][1])

    #         # update hard corresponding value
    #         for i, weight in enumerate(weights):
    #             if weight <= 1e-10:
    #                 continue
    #             topk, topk_pos = all_prob[i].topk(1)
    #             if topk_pos[0] != g_answer_token_ids[0]:
    #                 continue
    #             to_erase_pos.append(i)
    #             # print(i, weight, tokenizer.decode(topk_pos))

    #         # # update soft corresponding value
    #         # affinity = all_prob[:, g_answer_token_ids[0]]  # n_value
    #         # topk, topk_pos = affinity.topk(3072)
    #         # topk_pos = topk_pos.tolist()
    #         # for pos in topk_pos:
    #         #     if weights[pos] > 10:
    #         #         to_erase_pos.append(pos)
    #         #         break

    #     print('to_erase_pos|', len(to_erase_pos))

    #     backup_values = []
    #     # editing
    #     for i, pos in enumerate(to_erase_pos):
    #         backup_values.append(nkb_values[:, pos])
    #         # nkb_values[:, pos] = torch.zeros_like(nkb_values[:, pos])
    #         nkb_values[:, pos] -= o_embedding[g_answer_token_ids[0], :]

    #     results = model.generate(**sample)
    #     g_answer_token_ids = results[0].data.tolist()[1:-1]
    #     g_answer_token_ids = g_answer_token_ids[0:1]
    #     g_answer = tokenizer.decode(g_answer_token_ids)
    #     print('(after editing) generated_answer_ids|', g_answer_token_ids)
    #     print('(after editing) generated_answer|', g_answer)

    #     other_sample_indices = random.sample(list(range(len(dataset))), 5)

    #     other_answers_new = []
    #     for other_sample_index in other_sample_indices:
    #         other_sample = dataset[other_sample_index]
    #         if 'labels' in other_sample:
    #             del other_sample['labels']
    #         if 'offset_mapping' in other_sample:
    #             del other_sample['offset_mapping']
    #         if 'example_id' in other_sample:
    #             del other_sample['example_id']
    #         for key in other_sample.keys():
    #             other_sample[key] = torch.LongTensor([other_sample[key]]).cuda()
    #         other_results = model.generate(**other_sample)
    #         other_g_answer_token_ids = other_results[0].data.tolist()[1:-1]
    #         other_g_answer = tokenizer.decode(other_g_answer_token_ids)
    #         other_answers_new.append(other_g_answer)

    #     # restoring
    #     for i, pos in enumerate(to_erase_pos):
    #         nkb_values[:, pos] = backup_values[i]

    #     other_answers_ori = []
    #     for other_sample_index in other_sample_indices:
    #         other_sample = dataset[other_sample_index]
    #         if 'labels' in other_sample:
    #             del other_sample['labels']
    #         if 'offset_mapping' in other_sample:
    #             del other_sample['offset_mapping']
    #         if 'example_id' in other_sample:
    #             del other_sample['example_id']
    #         for key in other_sample.keys():
    #             other_sample[key] = torch.LongTensor([other_sample[key]]).cuda()
    #         other_results = model.generate(**other_sample)
    #         other_g_answer_token_ids = other_results[0].data.tolist()[1:-1]
    #         other_g_answer = tokenizer.decode(other_g_answer_token_ids)
    #         other_answers_ori.append(other_g_answer)

    #     erase = int(g_answer != answer)
    #     change = (5 - len(set(other_answers_ori).intersection(set(other_answers_new)))) / 5

    #     return [erase, change]

    # erase_cnt = 0
    # change_rate = []
    # # r2w_indices = r2w_indices[:10]
    # for index in r2w_indices:
    #     print('============' * 5, f'Erasing {index}-th example')
    #     erase, change = knowledge_erasing(index)
    #     print(f'**** Erase: {erase}, Change: {change}')
    #     erase_cnt += erase
    #     change_rate.append(change)
    # print(f'erasing rate: {erase_cnt/len(r2w_indices)*100:.4}% ({erase_cnt}/{len(w2r_indices)})')
    # print(f'change rate: {sum(change_rate)/len(change_rate)*100:.4}%')

if __name__ == "__main__":
    main()
