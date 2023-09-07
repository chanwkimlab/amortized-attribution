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
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from arguments import ClassifierArguments, DataTrainingArguments
from utils import configure_dataset, get_checkpoint, log_dataset, setup_dataset

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.32.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/image-classification/requirements.txt",
)


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


def main():
    ########################################################
    # Parse arguments
    #######################################################
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ClassifierArguments, DataTrainingArguments, TrainingArguments, OtherArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        classifier_args, data_args, training_args, other_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            classifier_args,
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
    # Configure dataset (set max samples, transforms, etc.)
    ########################################################
    dataset = copy.deepcopy(dataset_original)
    dataset = configure_dataset(
        dataset=dataset,
        image_processor=classifier_image_processor,
        training_args=training_args,
        data_args=data_args,
        train_augmentation=True,
        validation_augmentation=False,
        test_augmentation=False,
        logger=logger,
    )

    log_dataset(dataset=dataset, logger=logger)
    ########################################################
    # Initalize the classifier trainer
    ########################################################
    import copy

    dataset_classifier = copy.deepcopy(dataset)
    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # Initalize our trainer
    classifier_trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=dataset_classifier["train"] if training_args.do_train else None,
        eval_dataset=dataset_classifier["validation"]
        if training_args.do_train
        else None,
        compute_metrics=compute_metrics,
        tokenizer=classifier_image_processor,
        data_collator=collate_fn,
    )

    ########################################################
    # Training
    #######################################################
    if training_args.do_train:
        # Detecting last checkpoint
        checkpoint = get_checkpoint(training_args=training_args, logger=logger)
        train_result = classifier_trainer.train(resume_from_checkpoint=checkpoint)
        classifier_trainer.save_model()
        classifier_trainer.log_metrics("train", train_result.metrics)
        classifier_trainer.save_metrics("train", train_result.metrics)
        classifier_trainer.save_state()

    ########################################################
    # Evaluation
    #######################################################
    if training_args.do_eval:
        metrics = classifier_trainer.evaluate(dataset_classifier["test"])
        classifier_trainer.log_metrics("test", metrics)
        classifier_trainer.save_metrics("test", metrics)

    ########################################################
    # Write model card and (optionally) push to hub
    #######################################################
    kwargs = {
        "finetuned_from": classifier_args.classifier_model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        classifier_trainer.push_to_hub(**kwargs)
    else:
        classifier_trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
