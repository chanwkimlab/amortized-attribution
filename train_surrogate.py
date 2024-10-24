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
# Modifications Copyright 2023 Chanwoo Kim
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

# Import custom arguments, models, and utility functions.
from arguments import ClassifierArguments, DataTrainingArguments, SurrogateArguments
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

    extract_output_shapley: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    extract_output_binomial: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    num_mask_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of masks to use for extracting output."},
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
        train_augmentation=True,
        validation_augmentation=False,
        test_augmentation=False,
        logger=logger,
    )

    for dataset_key in dataset_surrogate.keys():
        if dataset_key == "train":
            dataset_surrogate[dataset_key] = MaskDataset(
                dataset_surrogate[dataset_key],
                num_features=196,
                num_mask_samples=1,
                paired_mask_samples=False,
                mode="uniform",
                random_seed=None,
            )
        elif dataset_key == "validation":
            dataset_surrogate[dataset_key] = MaskDataset(
                dataset_surrogate[dataset_key],
                num_features=196,
                num_mask_samples=1,
                paired_mask_samples=False,
                mode="uniform",
                random_seed=training_args.seed,
            )
        elif dataset_key == "test":
            dataset_surrogate[dataset_key] = MaskDataset(
                dataset_surrogate[dataset_key],
                num_features=196,
                num_mask_samples=1,
                paired_mask_samples=False,
                mode="uniform",
                random_seed=training_args.seed,
            )
        else:
            raise ValueError(
                f"Dataset key {dataset_key} not recognized. Choose from ['train', 'validation', 'test']"
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
        eval_dataset=(
            dataset_surrogate["validation"] if training_args.do_train else None
        ),
        compute_metrics=compute_metrics,
        tokenizer=surrogate_image_processor,
        data_collator=collate_fn,
    )

    if other_args.extract_output_shapley:
        if (
            isinstance(other_args.extract_output_shapley, str)
            and "," in other_args.extract_output_shapley
        ):
            extract_output_shapley_key = {
                "train": int(other_args.extract_output_shapley.split(",")[0]),
                "validation": int(other_args.extract_output_shapley.split(",")[1]),
                "test": int(other_args.extract_output_shapley.split(",")[2]),
            }
        else:
            extract_output_shapley_key = {
                "train": int(other_args.extract_output_shapley),
                "validation": int(other_args.extract_output_shapley),
                "test": int(other_args.extract_output_shapley),
            }

        dataset_extract = copy.deepcopy(dataset_original)
        dataset_extract = configure_dataset(
            dataset=dataset_extract,
            image_processor=surrogate_image_processor,
            training_args=training_args,
            data_args=data_args,
            train_augmentation=False,
            validation_augmentation=False,
            test_augmentation=False,
            logger=logger,
        )

        # save grand null
        logger.info("Calculating grand null output of the surrogate model")
        save_dict = {}
        for dataset_key in dataset_extract.keys():
            dataset_extract[dataset_key] = MaskDataset(
                dataset_extract[dataset_key],
                num_features=196,
                num_mask_samples=2,
                paired_mask_samples=True,
                mode="full",
                random_seed=None,
            )

        log_dataset(dataset=dataset_extract, logger=logger)
        for dataset_key in dataset_extract.keys():
            if extract_output_shapley_key[dataset_key] == 0:
                continue
            predict_output = surrogate_trainer.predict(dataset_extract[dataset_key])
            logger.info("Saving grand null output of the surrogate model")
            for sample_idx in tqdm(range(len(predict_output.predictions[0]))):
                save_path = os.path.join(
                    training_args.output_dir,
                    "extract_output",
                    dataset_key,
                    str(sample_idx),
                    "grand_null.pt",
                )
                # make dir
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "logits": predict_output.predictions[0][sample_idx],
                        "masks": predict_output.predictions[1][sample_idx],
                    },
                    save_path,
                )
        # save mask evaluation results
        logger.info("Calculating mask evaluation results")
        for dataset_key in dataset_extract.keys():
            dataset_extract[dataset_key].set_params(
                num_features=196,
                num_mask_samples=other_args.num_mask_samples,
                paired_mask_samples=other_args.antithetical_sampling,
                mode="shapley",
                random_seed=training_args.seed,
            )
        log_dataset(dataset=dataset_extract, logger=logger)
        for dataset_key in dataset_extract.keys():
            # continue
            for idx in tqdm(
                range(
                    (
                        extract_output_shapley_key[dataset_key]
                        + other_args.num_mask_samples
                        - 1
                    )
                    // other_args.num_mask_samples
                )
            ):
                dataset_extract[dataset_key].set_params(
                    num_features=196,
                    num_mask_samples=other_args.num_mask_samples,
                    paired_mask_samples=other_args.antithetical_sampling,
                    mode="shapley",
                    random_seed=training_args.seed + idx,
                )

                predict_output = surrogate_trainer.predict(dataset_extract[dataset_key])
                logger.info("Saving mask evaluation results")
                for sample_idx in tqdm(range(len(predict_output.predictions[0]))):
                    save_path = os.path.join(
                        training_args.output_dir,
                        "extract_output",
                        dataset_key,
                        str(sample_idx),
                        f"mask_eval_{idx*other_args.num_mask_samples}_{(idx+1)*other_args.num_mask_samples}.pt",
                    )
                    # make dir
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(
                        {
                            "logits": predict_output.predictions[0][sample_idx],
                            "masks": predict_output.predictions[1][sample_idx],
                        },
                        save_path,
                    )

    if other_args.extract_output_binomial:
        if (
            isinstance(other_args.extract_output_binomial, str)
            and "," in other_args.extract_output_binomial
        ):
            extract_output_binomial_key = {
                "train": int(other_args.extract_output_binomial.split(",")[0]),
                "validation": int(other_args.extract_output_binomial.split(",")[1]),
                "test": int(other_args.extract_output_binomial.split(",")[2]),
            }
        else:
            extract_output_binomial_key = {
                "train": int(other_args.extract_output_binomial),
                "validation": int(other_args.extract_output_binomial),
                "test": int(other_args.extract_output_binomial),
            }

        dataset_extract = copy.deepcopy(dataset_original)
        dataset_extract = configure_dataset(
            dataset=dataset_extract,
            image_processor=surrogate_image_processor,
            training_args=training_args,
            data_args=data_args,
            train_augmentation=False,
            validation_augmentation=False,
            test_augmentation=False,
            logger=logger,
        )

        # save grand null
        logger.info("Calculating grand null output of the surrogate model")
        save_dict = {}
        for dataset_key in dataset_extract.keys():
            dataset_extract[dataset_key] = MaskDataset(
                dataset_extract[dataset_key],
                num_features=196,
                num_mask_samples=2,
                paired_mask_samples=True,
                mode="full",
                random_seed=None,
            )

        log_dataset(dataset=dataset_extract, logger=logger)
        for dataset_key in dataset_extract.keys():
            if extract_output_binomial_key[dataset_key] == 0:
                continue
            predict_output = surrogate_trainer.predict(dataset_extract[dataset_key])
            logger.info("Saving grand null output of the surrogate model")
            for sample_idx in tqdm(range(len(predict_output.predictions[0]))):
                save_path = os.path.join(
                    training_args.output_dir,
                    "extract_output",
                    dataset_key,
                    str(sample_idx),
                    "grand_null.pt",
                )
                # make dir
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "logits": predict_output.predictions[0][sample_idx],
                        "masks": predict_output.predictions[1][sample_idx],
                    },
                    save_path,
                )
        # save mask evaluation results
        logger.info("Calculating mask evaluation results")
        for dataset_key in dataset_extract.keys():
            dataset_extract[dataset_key].set_params(
                num_features=196,
                num_mask_samples=other_args.num_mask_samples,
                paired_mask_samples=other_args.antithetical_sampling,
                mode="binomial",
                random_seed=training_args.seed,
            )
        log_dataset(dataset=dataset_extract, logger=logger)
        for dataset_key in dataset_extract.keys():
            # continue
            for idx in tqdm(
                range(
                    (
                        extract_output_binomial_key[dataset_key]
                        + other_args.num_mask_samples
                        - 1
                    )
                    // other_args.num_mask_samples
                )
            ):
                dataset_extract[dataset_key].set_params(
                    num_features=196,
                    num_mask_samples=other_args.num_mask_samples,
                    paired_mask_samples=other_args.antithetical_sampling,
                    mode="binomial",
                    random_seed=training_args.seed + idx,
                )

                predict_output = surrogate_trainer.predict(dataset_extract[dataset_key])
                logger.info("Saving mask evaluation results")
                for sample_idx in tqdm(range(len(predict_output.predictions[0]))):
                    save_path = os.path.join(
                        training_args.output_dir,
                        "extract_output",
                        dataset_key,
                        str(sample_idx),
                        f"mask_eval_{idx*other_args.num_mask_samples}_{(idx+1)*other_args.num_mask_samples}.pt",
                    )
                    # make dir
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(
                        {
                            "logits": predict_output.predictions[0][sample_idx],
                            "masks": predict_output.predictions[1][sample_idx],
                        },
                        save_path,
                    )

        # torch.save(dataset_extract, ("logs/extract_output_shapley.dataset.pt"))

        # torch.save(save_dict, os.path.join(training_args.output_dir, "extract_output_shapley.pt"))
        # torch.save(save_dict, os.path.join(training_args.output_dir, "extract_output_shapley.pt"))
        # # save dataset_extract
        # dataset_extract.save_to_disk(os.path.join(training_args.output_dir, "dataset_extract"))

    ########################################################
    # Training
    #######################################################
    if training_args.do_train:
        print("train loop")
        checkpoint = get_checkpoint(training_args=training_args, logger=logger)
        train_result = surrogate_trainer.train(resume_from_checkpoint=checkpoint)
        surrogate_trainer.save_model()
        surrogate_trainer.log_metrics("train", train_result.metrics)
        surrogate_trainer.save_metrics("train", train_result.metrics)
        surrogate_trainer.save_state()

    ########################################################
    # Evaluation
    #######################################################
    if training_args.do_eval:
        print("eval loop")
        metrics = surrogate_trainer.evaluate(dataset_surrogate["test"])
        surrogate_trainer.log_metrics("test", metrics)
        surrogate_trainer.save_metrics("test", metrics)

    ########################################################
    # Write model card and (optionally) push to hub
    #######################################################
    kwargs = {
        "finetuned_from": surrogate_args.surrogate_model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        surrogate_trainer.push_to_hub(**kwargs)
    else:
        surrogate_trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
