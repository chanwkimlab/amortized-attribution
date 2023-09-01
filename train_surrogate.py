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
import tqdm
import transformers
from datasets import load_dataset
from PIL import Image
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
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
    SurrogateForImageClassification,
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
print(MODEL_CONFIG_CLASSES)
print(MODEL_TYPES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the downloaded dataset."},
    )

    train_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the training data."}
    )
    validation_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the validation data."}
    )

    test_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the test data."}
    )

    train_validation_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )

    validation_test_split: Optional[float] = field(
        default=0.5, metadata={"help": "Percent to split off of val for test."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (
            self.train_dir is None and self.validation_dir is None
        ):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class OtherArguments:
    extract_output: Optional[str] = field(
        default=None,
        metadata={
            "help": "Extract output from the model. If None, will not extract output with N masks."
        },
    )

    num_mask_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of masks to use for extracting output."},
    )

    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
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


@dataclass
class ClassifierArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    classifier_model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    classifier_model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )

    classifier_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    classifier_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    classifier_model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    classifier_image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )

    classifier_ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


@dataclass
class SurrogateArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    surrogate_model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    surrogate_model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )

    surrogate_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    surrogate_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    surrogate_model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    surrogate_image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    surrogate_ignore_mismatched_sizes: bool = field(
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
    send_example_telemetry("run_image_classification", surrogate_args, data_args)

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
    classifier = AutoModelForImageClassification.from_pretrained(
        classifier_args.classifier_model_name_or_path,
        from_tf=bool(".ckpt" in classifier_args.classifier_model_name_or_path),
        config=classifier_config,
        cache_dir=classifier_args.classifier_cache_dir,
        revision=classifier_args.classifier_model_revision,
        token=other_args.token,
        ignore_mismatched_sizes=classifier_args.classifier_ignore_mismatched_sizes,
    )
    classifier_image_processor = AutoImageProcessor.from_pretrained(
        classifier_args.classifier_image_processor_name
        or classifier_args.classifier_model_name_or_path,
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
    # Align dataset to model settings
    ########################################################
    # Set the training transforms
    # if training_args.do_train:
    #     dataset["train_classifier"] = dataset["train"]
    #     dataset["train_classifier"].set_transform(
    #         get_image_transform(classifier_image_processor)["train_transform"]
    #     )
    #     dataset["validation_classifier"] = dataset["validation"]
    #     dataset["validation_classifier"].set_transform(
    #         get_image_transform(classifier_image_processor)["eval_transform"]
    #     )

    # # Set the validation transforms
    # if training_args.do_eval:
    #     dataset["test_classifier"] = dataset["test"]
    #     dataset["test_classifier"].set_transform(
    #         get_image_transform(classifier_image_processor)["eval_transform"]
    #     )

    ########################################################
    # Evaluate the original model
    ########################################################

    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     labels = torch.tensor([example["labels"] for example in examples])
    #     return {"pixel_values": pixel_values, "labels": labels}

    # classifier_trainer = Trainer(
    #     model=classifier,
    #     args=training_args,
    #     train_dataset=None,
    #     eval_dataset=None,
    #     compute_metrics=None,
    #     tokenizer=classifier_image_processor,
    #     data_collator=collate_fn,
    # )
    # print("classifier_trainer.label_names", classifier_trainer.label_names)
    # print(classifier_trainer.evaluate(dataset["validation_classifier"]))
    ########################################################
    # Add random generator
    ########################################################

    def transform_mask(example_batch):
        """Add mask to example_batch"""
        if "mask_random_seed" in example_batch:
            example_batch["masks"] = [
                generate_mask(
                    num_features=14 * 14,
                    num_mask_samples=other_args.num_mask_samples,
                    paired_mask_samples=False,
                    mode="uniform",
                    random_state=np.random.RandomState(
                        example_batch["mask_random_seed"][idx]
                    ),
                )
                for idx in range(len(example_batch["labels"]))
            ]
        else:
            example_batch["masks"] = [
                generate_mask(
                    num_features=14 * 14,
                    num_mask_samples=other_args.num_mask_samples,
                    paired_mask_samples=False,
                    mode="uniform",
                    random_state=None,
                )
                for idx in range(len(example_batch["labels"]))
            ]
        return example_batch

    dataset_surrogate = dataset.copy()
    dataset_surrogate["validation"] = dataset_surrogate["validation"].add_column(
        "mask_random_seed",
        iter(
            np.random.RandomState(training_args.seed).randint(
                0,
                len(dataset_surrogate["validation"]),
                size=len(dataset_surrogate["validation"]),
            )
        ),
    )

    dataset_surrogate["train"].set_transform(
        lambda x: transform_mask(
            get_image_transform(surrogate_image_processor)["train_transform"](x)
        )
    )

    dataset_surrogate["validation"].set_transform(
        lambda x: transform_mask(
            get_image_transform(surrogate_image_processor)["eval_transform"](x)
        )
    )

    ########################################################
    # Initalize the surrogate trainer
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
        # print(p)
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
        eval_dataset=dataset_surrogate["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=surrogate_image_processor,
        data_collator=collate_fn,
    )

    # ipdb.set_trace()
    # print("surrogate_trainer.label_names", surrogate_trainer.label_names)
    # print(surrogate_trainer.evaluate(dataset["validation_surrogate"]))

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

    if other_args.extract_output:
        if (
            isinstance(other_args.extract_output, str)
            and "," in other_args.extract_output
        ):
            extract_output_key = {
                "train": int(other_args.extract_output.split(",")[0]),
                "validation": int(other_args.extract_output.split(",")[1]),
                "test": int(other_args.extract_output.split(",")[2]),
            }
        else:
            extract_output_key = {
                "train": int(other_args.extract_output),
                "validation": int(other_args.extract_output),
                "test": int(other_args.extract_output),
            }

        def transform_mask_grand_null(example_batch):
            """Add mask to example_batch"""
            example_batch["masks"] = [
                generate_mask(
                    num_features=14 * 14,
                    num_mask_samples=2,
                    paired_mask_samples=True,
                    mode="full",
                    random_state=None,
                )
                for idx in range(len(example_batch["labels"]))
            ]
            return example_batch

        def transform_mask(example_batch):
            """Add mask to example_batch"""
            if "mask_random_seed" in example_batch:
                example_batch["masks"] = [
                    generate_mask(
                        num_features=14 * 14,
                        num_mask_samples=other_args.num_mask_samples,
                        paired_mask_samples=False,
                        mode="shapley",
                        random_state=np.random.RandomState(
                            example_batch["mask_random_seed"][idx]
                        ),
                    )
                    for idx in range(len(example_batch["labels"]))
                ]
            else:
                example_batch["masks"] = [
                    generate_mask(
                        num_features=14 * 14,
                        num_mask_samples=other_args.num_mask_samples,
                        paired_mask_samples=False,
                        mode="shapley",
                        random_state=None,
                    )
                    for idx in range(len(example_batch["labels"]))
                ]
            return example_batch

        # for dataset.keys()
        import copy

        dataset_extract = copy.deepcopy(dataset)
        # dataset_extract = dataset.copy()
        # for key in dataset_extract.keys():
        save_dict = {}
        for key in dataset_extract.keys():
            dataset_extract[key].set_transform(
                lambda x: transform_mask_grand_null(
                    get_image_transform(surrogate_image_processor)["eval_transform"](x)
                )
            )
            predict_output = surrogate_trainer.predict(dataset_extract[key])
            assert all(
                predict_output.label_ids
                == dataset_extract[key].with_transform(lambda x: x)["labels"]
            )
            save_dict.setdefault(key + "_grand_null_logits", []).append(
                predict_output.predictions[0]
            )
            save_dict.setdefault(key + "_grand_null_masks", []).append(
                predict_output.predictions[1]
            )
            # continue

            dataset_extract[key].set_transform(
                lambda x: transform_mask(
                    get_image_transform(surrogate_image_processor)["eval_transform"](x)
                )
            )
            for idx in tqdm.tqdm(
                range(
                    (extract_output_key[key] + other_args.num_mask_samples - 1)
                    // other_args.num_mask_samples
                )
            ):
                predict_output = surrogate_trainer.predict(dataset_extract[key])
                assert all(
                    predict_output.label_ids
                    == dataset_extract[key].with_transform(lambda x: x)["labels"]
                )
                save_dict.setdefault(key + "_logits", []).append(
                    predict_output.predictions[0]
                )
                save_dict.setdefault(key + "_masks", []).append(
                    predict_output.predictions[1]
                )
        ipdb.set_trace()
        # save to file
        # np.concatenate(save_dict["train_logits"])
        torch.save(
            save_dict, os.path.join(training_args.output_dir, "extract_output.pt")
        )
        # torch.save(dataset_extract, ("logs/extract_output.dataset.pt"))

        # torch.save(save_dict, os.path.join(training_args.output_dir, "extract_output.pt"))
        # torch.save(save_dict, os.path.join(training_args.output_dir, "extract_output.pt"))
        # # save dataset_extract
        # dataset_extract.save_to_disk(os.path.join(training_args.output_dir, "dataset_extract"))

    ########################################################
    # Training
    #######################################################
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = surrogate_trainer.train(resume_from_checkpoint=checkpoint)
        surrogate_trainer.save_model()
        surrogate_trainer.log_metrics("train", train_result.metrics)
        surrogate_trainer.save_metrics("train", train_result.metrics)
        surrogate_trainer.save_state()

    ########################################################
    # Evaluation
    #######################################################
    if training_args.do_eval:
        metrics = surrogate_trainer.evaluate()
        surrogate_trainer.log_metrics("eval", metrics)
        surrogate_trainer.save_metrics("eval", metrics)

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
