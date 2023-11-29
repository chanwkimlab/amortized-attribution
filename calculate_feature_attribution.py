#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
import copy
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import ipdb
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.nn import functional as F

# import tqdm
from tqdm.auto import tqdm
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from arguments import ClassifierArguments, DataTrainingArguments, SurrogateArguments
from feature_attribution_methods import (
    BanzhafRegression,
    BanzhafSampling,
    ShapleyRegression,
    ShapleySampling,
    ShapleySGD,
)
from models import (
    SurrogateForImageClassification,
    SurrogateForImageClassificationConfig,
)
from utils import (
    MaskDataset,
    configure_dataset,
    generate_mask,
    get_checkpoint,
    get_image_transform,
    log_dataset,
    setup_dataset,
)

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.32.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/image-classification/requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class OtherArguments:
    antithetical_sampling: bool = field(
        default=False,
        metadata={
            "help": "Whether to use antithetical sampling for the surrogate model."
        },
    )

    shapley_SGD: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    shapley_permutation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    shapley_regression: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    banzhaf_regression: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    banzhaf_sampling: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    target_subset_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )


def main():
    ########################################################
    # Parse arguments
    #######################################################
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (
            ClassifierArguments,
            SurrogateArguments,
            DataTrainingArguments,
            TrainingArguments,
            OtherArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            classifier_args,
            surrogate_args,
            data_args,
            training_args,
            other_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            classifier_args,
            surrogate_args,
            data_args,
            training_args,
            other_args,
        ) = parser.parse_args_into_dataclasses()

    ########################################################
    # Setup logging
    #######################################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    ########################################################
    # Correct cache dir if necessary
    ########################################################
    if not os.path.exists(
        os.sep.join((data_args.dataset_cache_dir).split(os.sep, 2)[:2])
    ):
        if os.path.exists("/data2"):
            data_args.dataset_cache_dir = os.sep.join(
                ["/data2"] + (data_args.dataset_cache_dir).split(os.sep, 2)[2:]
            )
            logger.info(
                f"dataset_cache_dir {data_args.dataset_cache_dir} not found, using {data_args.dataset_cache_dir}"
            )
        elif os.path.exists("/sdata"):
            data_args.dataset_cache_dir = os.sep.join(
                ["/sdata"] + (data_args.dataset_cache_dir).split(os.sep, 2)[2:]
            )
            logger.info(
                f"dataset_cache_dir {data_args.dataset_cache_dir} not found, using {data_args.dataset_cache_dir}"
            )
        else:
            raise ValueError(
                f"dataset_cache_dir {data_args.dataset_cache_dir} not found"
            )

    ########################################################
    # Set seed before initializing model.
    ########################################################
    set_seed(training_args.seed)

    ########################################################
    # Initialize our dataset and prepare it for the 'image-classification' task.
    ########################################################
    dataset_original, labels, label2id, id2label = setup_dataset(
        data_args=data_args, other_args=other_args
    )

    ########################################################
    # Initialize classifier model
    ########################################################
    classifier_config = AutoConfig.from_pretrained(
        classifier_args.classifier_config_name
        or classifier_args.classifier_model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=classifier_args.classifier_cache_dir,
        revision=classifier_args.classifier_model_revision,
        token=other_args.token,
    )

    ########################################################
    # Initialize surrogate model
    ########################################################
    surrogate_config = AutoConfig.from_pretrained(
        surrogate_args.surrogate_config_name
        or surrogate_args.surrogate_model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=surrogate_args.surrogate_cache_dir,
        revision=surrogate_args.surrogate_model_revision,
        token=other_args.token,
    )

    if os.path.isfile(
        f"{surrogate_args.surrogate_model_name_or_path}/config.json"
    ) and (
        json.loads(
            open(f"{surrogate_args.surrogate_model_name_or_path}/config.json").read()
        )["architectures"][0]
        == "SurrogateForImageClassification"
    ):
        surrogate = SurrogateForImageClassification.from_pretrained(
            surrogate_args.surrogate_model_name_or_path,
            from_tf=bool(".ckpt" in surrogate_args.surrogate_model_name_or_path),
            config=surrogate_config,
            cache_dir=surrogate_args.surrogate_cache_dir,
            revision=surrogate_args.surrogate_model_revision,
            token=other_args.token,
            ignore_mismatched_sizes=surrogate_args.surrogate_ignore_mismatched_sizes,
        )
    else:
        surrogate_for_image_classification_config = SurrogateForImageClassificationConfig(
            classifier_pretrained_model_name_or_path=classifier_args.classifier_model_name_or_path,
            classifier_config=classifier_config,
            classifier_from_tf=bool(
                ".ckpt" in classifier_args.classifier_model_name_or_path
            ),
            classifier_cache_dir=classifier_args.classifier_cache_dir,
            classifier_revision=classifier_args.classifier_model_revision,
            classifier_token=other_args.token,
            classifier_ignore_mismatched_sizes=classifier_args.classifier_ignore_mismatched_sizes,
            surrogate_pretrained_model_name_or_path=surrogate_args.surrogate_model_name_or_path,
            surrogate_config=surrogate_config,
            surrogate_from_tf=bool(
                ".ckpt" in surrogate_args.surrogate_model_name_or_path
            ),
            surrogate_cache_dir=surrogate_args.surrogate_cache_dir,
            surrogate_revision=surrogate_args.surrogate_model_revision,
            surrogate_token=other_args.token,
            surrogate_ignore_mismatched_sizes=surrogate_args.surrogate_ignore_mismatched_sizes,
        )

        surrogate = SurrogateForImageClassification(
            config=surrogate_for_image_classification_config,
        )

    surrogate_image_processor = AutoImageProcessor.from_pretrained(
        surrogate_args.surrogate_image_processor_name
        or surrogate_args.surrogate_model_name_or_path,
        cache_dir=surrogate_args.surrogate_cache_dir,
        revision=surrogate_args.surrogate_model_revision,
        token=other_args.token,
    )

    ########################################################
    # Configure dataset (set max samples, transforms, etc.)
    ########################################################
    dataset_surrogate = copy.deepcopy(dataset_original)
    dataset_surrogate = configure_dataset(
        dataset=dataset_surrogate,
        image_processor=surrogate_image_processor,
        training_args=training_args,
        data_args=data_args,
        train_augmentation=False,
        validation_augmentation=False,
        test_augmentation=False,
        logger=logger,
    )
    log_dataset(dataset=dataset_surrogate, logger=logger)

    ########################################################
    # Initalize the surrogate trainer
    ########################################################

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(
            predictions=np.argmax(p.predictions[0][:, 0, :], axis=1),
            references=p.label_ids,
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        masks = torch.tensor(np.array([example["masks"] for example in examples]))

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "masks": masks,
        }

    surrogate_trainer = Trainer(
        model=surrogate,
        args=training_args,
        train_dataset=dataset_surrogate["train"] if training_args.do_train else None,
        eval_dataset=dataset_surrogate["validation"]
        if training_args.do_train
        else None,
        compute_metrics=compute_metrics,
        tokenizer=surrogate_image_processor,
        data_collator=collate_fn,
    )

    if other_args.shapley_regression:
        if (
            isinstance(other_args.shapley_regression, str)
            and "," in other_args.shapley_regression
        ):
            shapley_regression_key = {
                "train": int(other_args.shapley_regression.split(",")[0]),
                "validation": int(other_args.shapley_regression.split(",")[1]),
                "test": int(other_args.shapley_regression.split(",")[2]),
            }
        else:
            shapley_regression_key = {
                "train": int(other_args.shapley_regression),
                "validation": int(other_args.shapley_regression),
                "test": int(other_args.shapley_regression),
            }
        dataset_shapley_regression = copy.deepcopy(dataset_original)
        dataset_shapley_regression = configure_dataset(
            dataset=dataset_shapley_regression,
            image_processor=surrogate_image_processor,
            training_args=training_args,
            data_args=data_args,
            train_augmentation=False,
            validation_augmentation=False,
            test_augmentation=False,
            logger=logger,
        )
        log_dataset(dataset=dataset_shapley_regression, logger=logger)
        for dataset_key in dataset_shapley_regression.keys():
            if shapley_regression_key[dataset_key] == 0:
                continue
            from scipy.special import softmax

            # for sample_idx in tqdm(
            #     list(range(len(dataset_shapley_regression[dataset_key])))
            # ):
            count_temp = 0
            for sample_idx in tqdm(
                np.random.RandomState(seed=42)
                .permutation(list(range(len(dataset_shapley_regression[dataset_key]))))
                .tolist()[6:]
            ):
                print(sample_idx)

                class SampleDataset:
                    def __init__(self, dataset, sample_idx, masks_list):
                        self.dataset = dataset
                        self.sample_idx = sample_idx
                        self.masks_list = masks_list

                    def __getitem__(self, idx):
                        item = self.dataset[self.sample_idx]
                        item["masks"] = self.masks_list[idx]
                        return item

                    def __len__(self):
                        return len(self.masks_list)

                func = lambda x: softmax(
                    surrogate_trainer.predict(
                        SampleDataset(
                            dataset=dataset_shapley_regression[dataset_key],
                            sample_idx=sample_idx,
                            masks_list=x,
                        ),
                    ).predictions[0],
                    axis=2,
                )
                if other_args.target_subset_size is None:
                    _, tracking_dict, ratio = ShapleyRegression(
                        surrogate=func,
                        num_players=196,
                        num_subsets=shapley_regression_key[dataset_key],
                        detect_convergence=True,
                        paired_sampling=other_args.antithetical_sampling,
                        return_all=True,
                        variance_batches=2,
                        min_variance_samples=2,
                    )

                    save_path = os.path.join(
                        training_args.output_dir,
                        "extract_output",
                        dataset_key,
                        str(sample_idx),
                        f"shapley_output.pt",
                    )
                    # make dir
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    torch.save(obj=tracking_dict, f=save_path)
                else:
                    for subset_group_idx in range(
                        shapley_regression_key[dataset_key]
                        // other_args.target_subset_size
                    ):
                        _, tracking_dict, ratio = ShapleyRegression(
                            surrogate=func,
                            num_players=196,
                            num_subsets=other_args.target_subset_size,
                            detect_convergence=True,
                            paired_sampling=other_args.antithetical_sampling,
                            return_all=True,
                            variance_batches=2,
                            min_variance_samples=2,
                        )

                        save_path = os.path.join(
                            training_args.output_dir,
                            "extract_output",
                            dataset_key,
                            str(sample_idx),
                            f"shapley_output_{other_args.target_subset_size}_{subset_group_idx}.pt",
                        )

                        torch.save(obj=tracking_dict, f=save_path)

    if other_args.shapley_SGD:
        if isinstance(other_args.shapley_SGD, str) and "," in other_args.shapley_SGD:
            shapley_SGD_key = {
                "train": int(other_args.shapley_SGD.split(",")[0]),
                "validation": int(other_args.shapley_SGD.split(",")[1]),
                "test": int(other_args.shapley_SGD.split(",")[2]),
            }
        else:
            shapley_SGD_key = {
                "train": int(other_args.shapley_SGD),
                "validation": int(other_args.shapley_SGD),
                "test": int(other_args.shapley_SGD),
            }
        dataset_shapley_SGD = copy.deepcopy(dataset_original)
        dataset_shapley_SGD = configure_dataset(
            dataset=dataset_shapley_SGD,
            image_processor=surrogate_image_processor,
            training_args=training_args,
            data_args=data_args,
            train_augmentation=False,
            validation_augmentation=False,
            test_augmentation=False,
            logger=logger,
        )
        log_dataset(dataset=dataset_shapley_SGD, logger=logger)
        for dataset_key in dataset_shapley_SGD.keys():
            if shapley_SGD_key[dataset_key] == 0:
                continue
            from scipy.special import softmax

            # for sample_idx in tqdm(
            #     list(range(len(dataset_shapley_SGD[dataset_key])))
            # ):
            for sample_idx in tqdm(
                list(range(len(dataset_shapley_SGD[dataset_key])))[:]
            ):

                class SampleDataset:
                    def __init__(self, dataset, sample_idx, masks_list):
                        self.dataset = dataset
                        self.sample_idx = sample_idx
                        self.masks_list = masks_list

                    def __getitem__(self, idx):
                        item = self.dataset[self.sample_idx]
                        item["masks"] = self.masks_list[idx]
                        return item

                    def __len__(self):
                        return len(self.masks_list)

                func = lambda x: softmax(
                    surrogate_trainer.predict(
                        SampleDataset(
                            dataset=dataset_shapley_SGD[dataset_key],
                            sample_idx=sample_idx,
                            masks_list=x,
                        ),
                    ).predictions[0],
                    axis=2,
                )
                if other_args.target_subset_size is None:
                    tracking_dict = ShapleySGD(
                        surrogate=func,
                        d=196,
                        num_subsets=shapley_SGD_key[dataset_key],
                        mbsize=2,
                        step=0.0005,
                        step_type="constant",
                        sampling="paired"
                        if other_args.antithetical_sampling
                        else "default",
                        averaging="uniform",
                    )

                    save_path = os.path.join(
                        training_args.output_dir,
                        "extract_output",
                        dataset_key,
                        str(sample_idx),
                        f"shapley_output.pt",
                    )
                    # make dir
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    torch.save(obj=tracking_dict, f=save_path)
                else:
                    for subset_group_idx in range(
                        shapley_SGD_key[dataset_key] // other_args.target_subset_size
                    ):
                        tracking_dict = ShapleySGD(
                            surrogate=func,
                            d=196,
                            num_subsets=shapley_SGD_key[dataset_key],
                            mbsize=2,
                            step=0.0005,
                            step_type="constant",
                            sampling="paired"
                            if other_args.antithetical_sampling
                            else "default",
                            averaging="uniform",
                        )

                        save_path = os.path.join(
                            training_args.output_dir,
                            "extract_output",
                            dataset_key,
                            str(sample_idx),
                            f"shapley_output_{other_args.target_subset_size}_{subset_group_idx}.pt",
                        )

                        torch.save(obj=tracking_dict, f=save_path)

    if other_args.shapley_permutation:
        if (
            isinstance(other_args.shapley_permutation, str)
            and "," in other_args.shapley_permutation
        ):
            shapley_permutation_key = {
                "train": int(other_args.shapley_permutation.split(",")[0]),
                "validation": int(other_args.shapley_permutation.split(",")[1]),
                "test": int(other_args.shapley_permutation.split(",")[2]),
            }
        else:
            shapley_permutation_key = {
                "train": int(other_args.shapley_permutation),
                "validation": int(other_args.shapley_permutation),
                "test": int(other_args.shapley_permutation),
            }

        dataset_shapley_permutation = copy.deepcopy(dataset_original)
        dataset_shapley_permutation = configure_dataset(
            dataset=dataset_shapley_permutation,
            image_processor=surrogate_image_processor,
            training_args=training_args,
            data_args=data_args,
            train_augmentation=False,
            validation_augmentation=False,
            test_augmentation=False,
            logger=logger,
        )
        log_dataset(dataset=dataset_shapley_permutation, logger=logger)

        for dataset_key in dataset_shapley_permutation.keys():
            if shapley_permutation_key[dataset_key] == 0:
                continue
            from scipy.special import softmax

            for sample_idx in tqdm(
                list(range(len(dataset_shapley_permutation[dataset_key])))[::-1]
            ):

                class SampleDataset:
                    def __init__(self, dataset, sample_idx, masks_list):
                        self.dataset = dataset
                        self.sample_idx = sample_idx
                        self.masks_list = masks_list

                    def __getitem__(self, idx):
                        item = self.dataset[self.sample_idx]
                        item["masks"] = self.masks_list[idx]
                        return item

                    def __len__(self):
                        return len(self.masks_list)

                func = lambda x: softmax(
                    surrogate_trainer.predict(
                        SampleDataset(
                            dataset=dataset_shapley_permutation[dataset_key],
                            sample_idx=sample_idx,
                            masks_list=x,
                        ),
                    ).predictions[0],
                    axis=2,
                )
                if other_args.target_subset_size is None:
                    _, tracking_dict, ratio = ShapleySampling(
                        surrogate=func,
                        num_players=196,
                        total_samples=int(
                            np.ceil(shapley_permutation_key[dataset_key] / 196)
                        ),
                        detect_convergence=True,
                        antithetical=other_args.antithetical_sampling,
                        return_all=True,
                    )

                    save_path = os.path.join(
                        training_args.output_dir,
                        "extract_output",
                        dataset_key,
                        str(sample_idx),
                        f"shapley_output.pt",
                    )
                    # make dir
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    torch.save(obj=tracking_dict, f=save_path)
                else:
                    for subset_group_idx in range(
                        shapley_permutation_key[dataset_key]
                        // other_args.target_subset_size
                    ):
                        _, tracking_dict, ratio = ShapleySampling(
                            surrogate=func,
                            num_players=196,
                            total_samples=int(
                                np.ceil(other_args.target_subset_size / 196)
                            ),
                            detect_convergence=True,
                            antithetical=other_args.antithetical_sampling,
                            return_all=True,
                        )

                        save_path = os.path.join(
                            training_args.output_dir,
                            "extract_output",
                            dataset_key,
                            str(sample_idx),
                            f"shapley_output_{other_args.target_subset_size}_{subset_group_idx}.pt",
                        )

                        # make dir
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        torch.save(obj=tracking_dict, f=save_path)

    if other_args.banzhaf_regression:
        if (
            isinstance(other_args.banzhaf_regression, str)
            and "," in other_args.banzhaf_regression
        ):
            banzhaf_regression_key = {
                "train": int(other_args.banzhaf_regression.split(",")[0]),
                "validation": int(other_args.banzhaf_regression.split(",")[1]),
                "test": int(other_args.banzhaf_regression.split(",")[2]),
            }
        else:
            banzhaf_regression_key = {
                "train": int(other_args.banzhaf_regression),
                "validation": int(other_args.banzhaf_regression),
                "test": int(other_args.banzhaf_regression),
            }
        dataset_banzhaf_regression = copy.deepcopy(dataset_original)
        dataset_banzhaf_regression = configure_dataset(
            dataset=dataset_banzhaf_regression,
            image_processor=surrogate_image_processor,
            training_args=training_args,
            data_args=data_args,
            train_augmentation=False,
            validation_augmentation=False,
            test_augmentation=False,
            logger=logger,
        )
        log_dataset(dataset=dataset_banzhaf_regression, logger=logger)
        for dataset_key in dataset_banzhaf_regression.keys():
            if banzhaf_regression_key[dataset_key] == 0:
                continue
            from scipy.special import softmax

            # for sample_idx in tqdm(
            #     list(range(len(dataset_banzhaf_regression[dataset_key])))
            # ):
            for sample_idx in tqdm(
                np.random.RandomState(seed=42)
                .permutation(list(range(len(dataset_banzhaf_regression[dataset_key]))))
                .tolist()
            ):
                print(sample_idx)

                class SampleDataset:
                    def __init__(self, dataset, sample_idx, masks_list):
                        self.dataset = dataset
                        self.sample_idx = sample_idx
                        self.masks_list = masks_list

                    def __getitem__(self, idx):
                        item = self.dataset[self.sample_idx]
                        item["masks"] = self.masks_list[idx]
                        return item

                    def __len__(self):
                        return len(self.masks_list)

                func = lambda x: softmax(
                    surrogate_trainer.predict(
                        SampleDataset(
                            dataset=dataset_banzhaf_regression[dataset_key],
                            sample_idx=sample_idx,
                            masks_list=x,
                        ),
                    ).predictions[0],
                    axis=2,
                )

                _, tracking_dict, ratio = BanzhafRegression(
                    surrogate=func,
                    num_players=196,
                    num_subsets=banzhaf_regression_key[dataset_key],
                    detect_convergence=True,
                    paired_sampling=other_args.antithetical_sampling,
                    return_all=True,
                    variance_batches=2,
                    min_variance_samples=2,
                )

                save_path = os.path.join(
                    training_args.output_dir,
                    "extract_output",
                    dataset_key,
                    str(sample_idx),
                    f"banzhaf_output.pt",
                )
                # make dir
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save(obj=tracking_dict, f=save_path)

    if other_args.banzhaf_sampling:
        if (
            isinstance(other_args.banzhaf_sampling, str)
            and "," in other_args.banzhaf_sampling
        ):
            banzhaf_sampling_key = {
                "train": int(other_args.banzhaf_sampling.split(",")[0]),
                "validation": int(other_args.banzhaf_sampling.split(",")[1]),
                "test": int(other_args.banzhaf_sampling.split(",")[2]),
            }
        else:
            banzhaf_sampling_key = {
                "train": int(other_args.banzhaf_sampling),
                "validation": int(other_args.banzhaf_sampling),
                "test": int(other_args.banzhaf_sampling),
            }

        dataset_banzhaf_sampling = copy.deepcopy(dataset_original)
        dataset_banzhaf_sampling = configure_dataset(
            dataset=dataset_banzhaf_sampling,
            image_processor=surrogate_image_processor,
            training_args=training_args,
            data_args=data_args,
            train_augmentation=False,
            validation_augmentation=False,
            test_augmentation=False,
            logger=logger,
        )
        log_dataset(dataset=dataset_banzhaf_sampling, logger=logger)

        for dataset_key in dataset_banzhaf_sampling.keys():
            if banzhaf_sampling_key[dataset_key] == 0:
                continue

            from scipy.special import softmax

            idx_list = [
                sample_idx
                for sample_idx in range(len(dataset_banzhaf_sampling[dataset_key]))
                if not os.path.exists(
                    os.path.join(
                        training_args.output_dir,
                        "extract_output",
                        dataset_key,
                        str(sample_idx),
                        f"banzhaf_output.pt",
                    )
                )
            ]

            print(idx_list)
            # ddd

            # for sample_idx in tqdm(
            #     np.random.RandomState(seed=42)
            #     .permutation(list(range(len(dataset_banzhaf_sampling[dataset_key]))))
            #     .tolist()[idx : idx + 1]
            # ):
            for sample_idx in tqdm(idx_list):
                print(sample_idx)

                class SampleDataset:
                    def __init__(self, dataset, sample_idx, masks_list):
                        self.dataset = dataset
                        self.sample_idx = sample_idx
                        self.masks_list = masks_list

                    def __getitem__(self, idx):
                        item = self.dataset[self.sample_idx]
                        item["masks"] = self.masks_list[idx]
                        return item

                    def __len__(self):
                        return len(self.masks_list)

                func = lambda x: softmax(
                    surrogate_trainer.predict(
                        SampleDataset(
                            dataset=dataset_banzhaf_sampling[dataset_key],
                            sample_idx=sample_idx,
                            masks_list=x,
                        ),
                    ).predictions[0],
                    axis=2,
                )
                if other_args.target_subset_size is None:
                    _, tracking_dict = BanzhafSampling(
                        surrogate=func,
                        num_players=196,
                        num_subsets=banzhaf_sampling_key[dataset_key],
                        antithetical=other_args.antithetical_sampling,
                        return_all=True,
                    )

                    save_path = os.path.join(
                        training_args.output_dir,
                        "extract_output",
                        dataset_key,
                        str(sample_idx),
                        f"banzhaf_output.pt",
                    )
                    # make dir
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    torch.save(obj=tracking_dict, f=save_path)
                else:
                    for subset_group_idx in range(
                        banzhaf_sampling_key[dataset_key]
                        // other_args.target_subset_size
                    ):
                        (
                            _,
                            tracking_dict,
                        ) = BanzhafSampling(
                            surrogate=func,
                            num_players=196,
                            num_subsets=other_args.target_subset_size,
                            antithetical=other_args.antithetical_sampling,
                            return_all=True,
                        )

                        save_path = os.path.join(
                            training_args.output_dir,
                            "extract_output",
                            dataset_key,
                            str(sample_idx),
                            f"banzhaf_output_{other_args.target_subset_size}_{subset_group_idx}.pt",
                        )

                        # make dir
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        torch.save(obj=tracking_dict, f=save_path)


if __name__ == "__main__":
    main()
