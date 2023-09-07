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
from PIL import Image
from torch.nn import functional as F
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

from models import (
    ExplainerForImageClassification,
    ExplainerForImageClassificationConfig,
    SurrogateForImageClassificationConfig,
)
from utils import generate_mask, get_image_transform

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
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )


@dataclass
class ExplainerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    explainer_model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    explainer_model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )

    explainer_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    explainer_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    explainer_model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    explainer_image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    explainer_ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
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
            SurrogateArguments,
            ExplainerArguments,
            DataTrainingArguments,
            TrainingArguments,
            OtherArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            surrogate_args,
            explainer_args,
            data_args,
            training_args,
            other_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            surrogate_args,
            explainer_args,
            data_args,
            training_args,
            other_args,
        ) = parser.parse_args_into_dataclasses()

    if other_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34.",
            FutureWarning,
        )
        if other_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        other_args.token = other_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification", explainer_args, data_args)

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
    # Set seed before initializing model.
    ########################################################
    set_seed(training_args.seed)

    ########################################################
    # Initialize our dataset and prepare it for the 'image-classification' task.
    ########################################################
    if data_args.dataset_name is not None:
        if data_args.dataset_name == "frgfm/imagenette":
            dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=data_args.dataset_cache_dir,
                task=None,
                token=other_args.token,
            )

            for split in dataset.keys():
                if "label" in dataset[split].features:
                    dataset[split] = dataset[split].rename_column("label", "labels")

        else:
            dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=data_args.dataset_cache_dir,
                task="image-classification",
                token=other_args.token,
            )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        if data_args.test_dir is not None:
            data_files["test"] = os.path.join(data_args.test_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=surrogate_args.surrogate_cache_dir,
            task="image-classification",
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_validation_split = (
        None if "validation" in dataset.keys() else data_args.train_validation_split
    )
    if (
        isinstance(data_args.train_validation_split, float)
        and data_args.train_validation_split > 0.0
    ):
        split = dataset["train"].train_test_split(data_args.train_validation_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    data_args.validation_test_split = (
        None if "test" in dataset.keys() else data_args.validation_test_split
    )

    if (
        isinstance(data_args.validation_test_split, float)
        and data_args.validation_test_split > 0.0
    ):
        split = dataset["validation"].train_test_split(data_args.validation_test_split)
        dataset["validation"] = split["train"]
        dataset["test"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["labels"].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    ########################################################
    # Initialize explainer model
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
    surrogate_for_image_classification_config = SurrogateForImageClassificationConfig(
        surrogate_pretrained_model_name_or_path=surrogate_args.surrogate_model_name_or_path,
        surrogate_config=surrogate_config,
        surrogate_from_tf=bool(".ckpt" in surrogate_args.surrogate_model_name_or_path),
        surrogate_cache_dir=surrogate_args.surrogate_cache_dir,
        surrogate_revision=surrogate_args.surrogate_model_revision,
        surrogate_token=other_args.token,
        surrogate_ignore_mismatched_sizes=surrogate_args.surrogate_ignore_mismatched_sizes,
    )
    explainer_config = AutoConfig.from_pretrained(
        explainer_args.explainer_config_name
        or explainer_args.explainer_model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=explainer_args.explainer_cache_dir,
        revision=explainer_args.explainer_model_revision,
        token=other_args.token,
    )
    explainer_for_image_classification_config = ExplainerForImageClassificationConfig(
        surrogate_pretrained_model_name_or_path=surrogate_args.surrogate_model_name_or_path,
        surrogate_config=surrogate_for_image_classification_config,
        surrogate_from_tf=bool(".ckpt" in surrogate_args.surrogate_model_name_or_path),
        surrogate_cache_dir=surrogate_args.surrogate_cache_dir,
        surrogate_revision=surrogate_args.surrogate_model_revision,
        surrogate_token=other_args.token,
        surrogate_ignore_mismatched_sizes=surrogate_args.surrogate_ignore_mismatched_sizes,
        explainer_pretrained_model_name_or_path=explainer_args.explainer_model_name_or_path,
        explainer_config=explainer_config,
        explainer_from_tf=bool(".ckpt" in explainer_args.explainer_model_name_or_path),
        explainer_cache_dir=explainer_args.explainer_cache_dir,
        explainer_revision=explainer_args.explainer_model_revision,
        explainer_token=other_args.token,
        explainer_ignore_mismatched_sizes=explainer_args.explainer_ignore_mismatched_sizes,
    )

    explainer = ExplainerForImageClassification(
        config=explainer_for_image_classification_config,
    )
    explainer_image_processor = AutoImageProcessor.from_pretrained(
        explainer_args.explainer_image_processor_name
        or explainer_args.explainer_model_name_or_path,
        cache_dir=explainer_args.explainer_cache_dir,
        revision=explainer_args.explainer_model_revision,
        token=other_args.token,
    )

    ########################################################
    # Align dataset to model settings
    ########################################################

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if "validation" not in dataset:
            raise ValueError("--do_train requires a validation dataset")

    if data_args.max_train_samples is not None:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=training_args.seed)
            .select(range(data_args.max_train_samples))
        )
    if data_args.max_val_samples is not None:
        dataset["validation"] = (
            dataset["validation"]
            .shuffle(seed=training_args.seed)
            .select(range(data_args.max_val_samples))
        )

    if training_args.do_eval:
        if "test" not in dataset:
            raise ValueError("--do_eval requires a test dataset")

    if data_args.max_test_samples is not None:
        dataset["test"] = (
            dataset["test"]
            .shuffle(seed=training_args.seed)
            .select(range(data_args.max_test_samples))
        )

    ########################################################
    # Add random generator
    ########################################################
    dataset["train_explainer"] = dataset["train"]
    dataset["validation_explainer"] = dataset["validation"]
    dataset["validation_explainer"] = dataset["validation_explainer"].add_column(
        "mask_random_seed",
        iter(
            np.random.RandomState(training_args.seed).randint(
                0,
                len(dataset["validation_explainer"]),
                size=len(dataset["validation_explainer"]),
            )
        ),
    )

    loaded = torch.load(
        "logs/vitbase_imagenette_surrogate_eval/extract_output_all.pt",
        map_location="cpu",
    )

    def tranform_mask(example_batch):
        """Add mask to example_batch"""

        mask_full_null = generate_mask(
            num_features=14 * 14,
            num_mask_samples=2,
            paired_mask_samples=True,
            mode="full",
            random_state=None,
        )
        if "mask_random_seed" in example_batch:
            example_batch["masks"] = [
                np.vstack(
                    [
                        generate_mask(
                            num_features=14 * 14,
                            num_mask_samples=32,
                            paired_mask_samples=True,
                            mode="shapley",
                            random_state=np.random.RandomState(
                                example_batch["mask_random_seed"][idx]
                            ),
                        ),
                    ]
                )
                for idx in range(len(example_batch["labels"]))
            ]
        else:
            example_batch["masks"] = [
                np.vstack(
                    [
                        generate_mask(
                            num_features=14 * 14,
                            num_mask_samples=32,
                            paired_mask_samples=True,
                            mode="shapley",
                            random_state=None,
                        ),
                    ]
                )
                for idx in range(len(example_batch["labels"]))
            ]
        return example_batch

    dataset["train_explainer"].set_transform(
        lambda x: tranform_mask(
            get_image_transform(explainer_image_processor)["eval_transform"](x)
        )
    )

    dataset["validation_explainer"].set_transform(
        lambda x: tranform_mask(
            get_image_transform(explainer_image_processor)["eval_transform"](x)
        )
    )
    # import ipdb

    # ipdb.set_trace()

    ########################################################
    # Initalize the explainer trainer
    ########################################################
    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        # import ipdb

        # ipdb.set_trace()
        # print(p.predictions.shape, p.label_ids.shape)
        # return metric.compute(
        #     predictions=np.argmax(p.predictions[:, 0, :], axis=1),
        #     references=p.label_ids,
        # )
        return {}

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        masks = torch.tensor(np.array([example["masks"] for example in examples]))

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "masks": masks,
        }

    explainer_trainer = Trainer(
        model=explainer,
        args=training_args,
        train_dataset=dataset["train_explainer"] if training_args.do_train else None,
        eval_dataset=dataset["validation_explainer"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=explainer_image_processor,
        data_collator=collate_fn,
    )

    # ipdb.set_trace()
    # print("explainer_trainer.label_names", explainer_trainer.label_names)
    # print(explainer_trainer.evaluate(dataset["validation_explainer"]))

    ########################################################
    # Detecting last checkpoint
    #######################################################
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ########################################################
    # Training
    #######################################################
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = explainer_trainer.train(resume_from_checkpoint=checkpoint)
        explainer_trainer.save_model()
        explainer_trainer.log_metrics("train", train_result.metrics)
        explainer_trainer.save_metrics("train", train_result.metrics)
        explainer_trainer.save_state()

    ########################################################
    # Evaluation
    #######################################################
    if training_args.do_eval:
        metrics = explainer_trainer.evaluate()
        explainer_trainer.log_metrics("eval", metrics)
        explainer_trainer.save_metrics("eval", metrics)

    ########################################################
    # Write model card and (optionally) push to hub
    #######################################################
    kwargs = {
        "finetuned_from": explainer_args.explainer_model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        explainer_trainer.push_to_hub(**kwargs)
    else:
        explainer_trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
