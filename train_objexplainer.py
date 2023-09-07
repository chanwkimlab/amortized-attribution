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

from arguments import DataTrainingArguments, ExplainerArguments, SurrogateArguments
from models import (
    ObjExplainerForImageClassification,
    ObjExplainerForImageClassificationConfig,
    SurrogateForImageClassificationConfig,
)
from utils import (
    MaskDataset,
    configure_dataset,
    generate_mask,
    get_checkpoint,
    get_image_transform,
    load_shapley,
    log_dataset,
    read_eval_results,
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
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )

    train_subsets_cache_path: str = field(
        default=None,
        metadata={
            "help": "Where to load the downloaded dataset.",
        },
    )
    validation_subsets_cache_path: str = field(
        default=None,
        metadata={
            "help": "Where to load the downloaded dataset.",
        },
    )
    test_subsets_cache_path: str = field(
        default=None,
        metadata={
            "help": "Where to load the downloaded dataset.",
        },
    )
    train_mask_mode: str = field(
        default="incremental,1",
        metadata={
            "help": "mask mode for train",
        },
    )

    validation_mask_mode: str = field(
        default="incremental,1",
        metadata={
            "help": "mask mode for validation",
        },
    )

    test_mask_mode: str = field(
        default="incremental,1",
        metadata={
            "help": "mask mode for test",
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
    explainer_for_image_classification_config = ObjExplainerForImageClassificationConfig(
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

    explainer = ObjExplainerForImageClassification(
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
    # Configure dataset (set max samples, transforms, etc.)
    ########################################################
    dataset_explainer = copy.deepcopy(dataset_original)
    dataset_explainer = configure_dataset(
        dataset=dataset_explainer,
        image_processor=explainer_image_processor,
        training_args=training_args,
        data_args=data_args,
        train_augmentation=False,
        validation_augmentation=False,
        test_augmentation=False,
        logger=logger,
    )

    for dataset_key in dataset_explainer.keys():
        # mask num samples
        # batch size
        if dataset_key == "train":
            cache_path = other_args.train_subsets_cache_path
            mask_mode = other_args.train_mask_mode
        elif dataset_key == "validation":
            cache_path = other_args.validation_subsets_cache_path
            mask_mode = other_args.validation_mask_mode
        elif dataset_key == "test":
            continue
            # cache_path = other_args.test_subsets_cache_path
            # mask_mode = other_args.test_mask_mode
        else:
            raise ValueError(
                f"Dataset key {dataset_key} not recognized. Choose from ['train', 'validation', 'test']"
            )

        grand_all = []
        null_all = []
        subsets_logits_all = []
        subsets_masks_all = []
        assert len(dataset_explainer[dataset_key]) == len(
            os.listdir(cache_path)
        ), f"eval list: {len(dataset_explainer[dataset_key])} != {len(os.listdir(cache_path))}"
        for sample_idx in tqdm(range(len(dataset_explainer[dataset_key]))):
            eval_results = read_eval_results(f"{cache_path}/{sample_idx}")
            # ipdb.set_trace()
            # x=explainer.surrogate(
            #     pixel_values=dataset_explainer[dataset_key][0][
            #         "pixel_values"
            #     ].unsqueeze(0),
            #     masks=torch.ones(1, 1, 196),
            #     return_loss=False
            # )

            grand_all.append(eval_results["grand"]["logits"])
            null_all.append(eval_results["null"]["logits"])
            subsets_logits_all.append(
                eval_results["subsets"]["logits"]
            )  # (num_subsets, num_classes)
            subsets_masks_all.append(
                eval_results["subsets"]["masks"]
            )  # (num_subsets, num_players)

        num_subsets = np.unique(
            [subsets_logits.shape[0] for subsets_logits in subsets_logits_all]
        )
        assert len(num_subsets) == 1
        num_subsets = num_subsets[0]

        dataset_explainer[dataset_key] = MaskDataset(
            dataset_explainer[dataset_key],
        )
        # import ipdb

        # ipdb.set_trace()
        if mask_mode.startswith("new_samples"):
            masks_param = int(mask_mode.split(",")[1])

            dataset_explainer[dataset_key].set_cache(
                masks=subsets_masks_all,
                model_outputs=subsets_logits_all,
                grand_values=grand_all,
                null_values=null_all,
                cache_start_idx=lambda x: masks_param * x,
                cache_mask_size=masks_param,
            )
        elif mask_mode.startswith("accumulated"):
            masks_param = int(mask_mode.split(",")[1])

            dataset_explainer[dataset_key].set_cache(
                masks=subsets_masks_all,
                model_outputs=subsets_logits_all,
                grand_values=grand_all,
                null_values=null_all,
                cache_start_idx=0,
                cache_mask_size=lambda x: masks_param * (x + 1),
            )
        elif mask_mode.startswith("upfront"):
            masks_param = int(mask_mode.split(",")[1])
            dataset_explainer[dataset_key].set_cache(
                masks=subsets_masks_all,
                model_outputs=subsets_logits_all,
                grand_values=grand_all,
                null_values=null_all,
                cache_start_idx=0,
                cache_mask_size=masks_param,
            )
        else:
            raise ValueError(
                f"mask mode {mask_mode} not recognized. Choose from ['new_samples', 'accumulated', 'upfront']"
            )

    log_dataset(dataset=dataset_explainer, logger=logger)
    dataset_explainer["train"].reset_cache_counter()
    dataset_explainer["validation"].reset_cache_counter()

    ########################################################
    # Initalize the explainer trainer
    ########################################################

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        # import ipdb
        return {}

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        masks = torch.tensor(np.array([example["masks"] for example in examples]))
        model_outputs = torch.tensor(
            np.array([example["model_outputs"] for example in examples])
        )
        grand_values = torch.tensor(
            np.array([example["grand_values"] for example in examples])
        )
        null_values = torch.tensor(
            np.array([example["null_values"] for example in examples])
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "masks": masks,
            "model_outputs": model_outputs,
            "grand_values": grand_values,
            "null_values": null_values,
        }

    explainer_trainer = Trainer(
        model=explainer,
        args=training_args,
        train_dataset=dataset_explainer["train"] if training_args.do_train else None,
        eval_dataset=dataset_explainer["validation"]
        if training_args.do_train
        else None,
        compute_metrics=compute_metrics,
        tokenizer=explainer_image_processor,
        data_collator=collate_fn,
    )

    ########################################################
    # Training
    #######################################################
    if training_args.do_train:
        checkpoint = get_checkpoint(training_args=training_args, logger=logger)
        train_result = explainer_trainer.train(resume_from_checkpoint=checkpoint)
        explainer_trainer.save_model()
        explainer_trainer.log_metrics("train", train_result.metrics)
        explainer_trainer.save_metrics("train", train_result.metrics)
        explainer_trainer.save_state()

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

    ########################################################
    # Evaluation
    #######################################################
    if training_args.do_eval:
        metrics = explainer_trainer.evaluate(dataset_explainer["test"])
        explainer_trainer.log_metrics("test", metrics)
        explainer_trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
