import glob
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm import tqdm
from transformers.trainer_utils import get_last_checkpoint


def setup_dataset(data_args, other_args):
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
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=data_args.dataset_cache_dir,
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
    return dataset, labels, label2id, id2label


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def get_image_transform(image_processor):
    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (
            image_processor.size["height"],
            image_processor.size["width"],
        )
    normalize = Normalize(
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    )
    _augment_transform = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _static_transform = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def augment_transform(example_batch):
        """Apply _augment_transform across a batch."""
        example_batch["pixel_values"] = [
            _augment_transform(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    def static_transform(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _static_transform(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    return {
        "augment_transform": augment_transform,
        "static_transform": static_transform,
    }


def configure_dataset(
    dataset,
    image_processor,
    training_args,
    data_args,
    train_augmentation,
    validation_augmentation,
    test_augmentation,
    logger,
):
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if "validation" not in dataset:
            raise ValueError("--do_train requires a validation dataset")

    if training_args.do_eval:
        if "test" not in dataset:
            raise ValueError("--do_eval requires a test dataset")

    # Set max samples
    if data_args.max_train_samples is not None:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=training_args.seed)
            .select(range(data_args.max_train_samples))
        )
    if data_args.max_validation_samples is not None:
        dataset["validation"] = (
            dataset["validation"]
            .shuffle(seed=training_args.seed)
            .select(range(data_args.max_validation_samples))
        )
    if data_args.max_test_samples is not None:
        dataset["test"] = (
            dataset["test"]
            .shuffle(seed=training_args.seed)
            .select(range(data_args.max_test_samples))
        )

    # Set the training transforms
    for dataset_key in dataset.keys():
        if dataset_key == "train":
            dataset[dataset_key].set_transform(
                lambda x: get_image_transform(image_processor)[
                    "augment_transform" if train_augmentation else "static_transform"
                ](x)
            )
        elif dataset_key == "validation":
            dataset[dataset_key].set_transform(
                lambda x: get_image_transform(image_processor)[
                    "augment_transform"
                    if validation_augmentation
                    else "static_transform"
                ](x)
            )
        elif dataset_key == "test":
            dataset[dataset_key].set_transform(
                lambda x: get_image_transform(image_processor)[
                    "augment_transform" if test_augmentation else "static_transform"
                ](x)
            )
        else:
            raise ValueError(
                f"Dataset key {dataset_key} not recognized. Choose from ['train', 'validation', 'test']"
            )

    # logger.info(f"Dataset configured: ")

    return dataset


def log_dataset(dataset, logger):
    logger.info(f"Dataset statistics: ")
    for dataset_key in dataset.keys():
        logger.info(f"*{dataset_key}: {len(dataset[dataset_key]):,} samples")
        item_0 = dataset[dataset_key][0]
        item_1 = dataset[dataset_key][0]
        for item_key in item_0.keys():
            # compare if two values are equal
            is_item_static = None
            if isinstance(item_0[item_key], np.ndarray):
                is_item_static = (item_0[item_key] == item_1[item_key]).all()
            elif isinstance(item_0[item_key], torch.Tensor):
                is_item_static = (item_0[item_key] == item_1[item_key]).all()
            elif (
                isinstance(item_0[item_key], int)
                or isinstance(item_0[item_key], str)
                or isinstance(item_0[item_key], float)
            ):
                is_item_static = item_0[item_key] == item_1[item_key]
            if is_item_static is None:
                logger.info(f"    {item_key}: unknown")
            else:
                logger.info(
                    f"    {item_key}: {'static' if is_item_static else 'dynamic'}"
                )


def generate_mask(
    num_features: int,
    num_mask_samples: int = 1,
    paired_mask_samples: bool = True,
    mode: str = "uniform",
    random_state: np.random.RandomState or None = None,
) -> np.array:
    """
    Args:
        num_features: the number of features
        num_mask_samples: the number of masks to generate
        paired_mask_samples: if True, the generated masks are pairs of x and 1-x.
        mode: the distribution that the number of masked features follows. ('uniform' or 'shapley')
        random_state: random generator

    Returns:
        torch.Tensor of shape
        (num_masks, num_features) if num_masks is int
        (num_features) if num_masks is None

    """
    random_state = random_state or np.random

    num_samples_ = num_mask_samples

    if paired_mask_samples:
        assert (
            num_samples_ % 2 == 0
        ), "'num_samples' must be a multiple of 2 if 'paired_mask_samples' is True"
        num_samples_ = num_samples_ // 2

    if mode == "uniform":
        masks = (
            random_state.rand(num_samples_, num_features)
            > random_state.rand(num_samples_, 1)
        ).astype("int")
    elif mode == "banzhaf" or mode == "binomial":
        masks = (np.random.rand(num_samples_, num_features) > 0.5).astype("int")
    elif mode == "shapley":
        probs = 1 / (
            np.arange(1, num_features) * (num_features - np.arange(1, num_features))
        )
        probs = probs / probs.sum()
        num_included = random_state.choice(
            np.arange(1, num_features), size=num_samples_, p=probs, replace=True
        )
        tril = np.tril(np.ones((num_features - 1, num_features)), k=0)
        masks = tril[num_included - 1].astype(int)
        for i in range(num_samples_):
            masks[i] = masks[i, random_state.permutation(num_features)]
    elif mode == "full":
        masks = np.ones((num_samples_, num_features)).astype("int")
    elif mode == "empty":
        masks = np.zeros((num_samples_, num_features)).astype("int")
    else:
        raise ValueError("'mode' must be 'random' or 'shapley'")

    if paired_mask_samples:
        masks = np.stack([masks, 1 - masks], axis=1).reshape(
            num_samples_ * 2, num_features
        )

    return masks  # (num_samples, num_masks)


class MaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        num_features=None,
        num_mask_samples=None,
        paired_mask_samples=None,
        mode=None,
        random_seed=None,
    ):
        self.dataset = dataset

        self.use_cache = False

        # for not using cache
        self.num_features = num_features
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self.mode = mode
        self.random_seed = random_seed

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.use_cache == True:
            if (
                self.masks is not None
                or self.model_outputs is not None
                or self.shapley_values is not None
            ):
                if isinstance(self.cache_start_idx, int):
                    start_idx = self.cache_start_idx
                else:
                    start_idx = self.cache_start_idx(self.cache_counter_list[idx])

                if isinstance(self.cache_mask_size, int):
                    step_size = self.cache_mask_size
                else:
                    step_size = self.cache_mask_size(self.cache_counter_list[idx])

                self.cache_counter_list[idx] = self.cache_counter_list[idx] + 1

                if self.masks is not None:
                    assert (start_idx + step_size) <= len(
                        self.masks[idx]
                    ), f"start_idx: {start_idx}, step_size: {step_size}, len(self.masks[idx]): {len(self.masks[idx])}"
                    item["masks"] = self.masks[idx][start_idx : start_idx + step_size]
                if self.model_outputs is not None:
                    assert (start_idx + step_size) <= len(
                        self.model_outputs[idx]
                    ), f"start_idx: {start_idx}, step_size: {step_size}, len(self.model_outputs[idx]): {len(self.model_outputs[idx])}"
                    item["model_outputs"] = self.model_outputs[idx][
                        start_idx : start_idx + step_size
                    ]

                if self.shapley_values is not None:
                    item["shapley_values"] = self.shapley_values[idx][start_idx]

            if self.grand_values is not None:
                item["grand_values"] = self.grand_values[idx]
            if self.null_values is not None:
                item["null_values"] = self.null_values[idx]

        elif self.use_cache == False:
            if self.random_seed is None:
                random_state = None
            elif isinstance(self.random_seed, int):
                random_state = np.random.RandomState(self.random_seed + idx)
            else:
                raise ValueError(
                    "random_seed must be an integer, an iterable of integers or None"
                )

            item["masks"] = generate_mask(
                num_features=self.num_features,
                num_mask_samples=self.num_mask_samples,
                paired_mask_samples=self.paired_mask_samples,
                mode=self.mode,
                random_state=random_state,
            )
        return item

    def set_params(
        self,
        num_features=None,
        num_mask_samples=None,
        paired_mask_samples=None,
        mode=None,
        random_seed=None,
    ):
        self.use_cache = False
        self.num_features = num_features
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self.mode = mode
        self.random_seed = random_seed

    def set_cache(
        self,
        masks=None,
        model_outputs=None,
        grand_values=None,
        null_values=None,
        shapley_values=None,
        cache_start_idx=0,
        cache_mask_size=1,
    ):
        self.use_cache = True
        self.cache_start_idx = cache_start_idx
        self.cache_mask_size = cache_mask_size

        # assert not (
        #     masks is None and model_outputs is None
        # ), "masks and model_outputs cannot both be None"

        if masks is None:
            self.masks = None
        else:
            assert len(self.dataset) == len(masks)
            self.masks = masks

        if model_outputs is None:
            self.model_outputs = None
        else:
            assert len(self.dataset) == len(model_outputs)
            self.model_outputs = model_outputs

        if grand_values is None:
            self.grand_values = None
        else:
            assert len(self.dataset) == len(grand_values)
            self.grand_values = grand_values

        if null_values is None:
            self.null_values = None
        else:
            assert len(self.dataset) == len(null_values)
            self.null_values = null_values

        if shapley_values is None:
            self.shapley_values = None
        else:
            assert len(self.dataset) == len(shapley_values)
            self.shapley_values = shapley_values

        self.reset_cache_counter()

    def reset_cache_counter(self):
        self.cache_counter_list = [0 for _ in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)


def get_checkpoint(training_args, logger):
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
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    return checkpoint


def load_shapley(path, target_subset_size=None):
    file_list = glob.glob(str(Path(path) / "[0-9]*"))
    output_dict = {}
    if target_subset_size is None:
        for file in tqdm(file_list):
            loaded = torch.load(Path(file) / "shapley_output.pt")

            output_dict[int(file.split("/")[-1])] = loaded
    else:
        for file in tqdm(file_list):
            subset_file_list = glob.glob(
                str(Path(file) / f"shapley_output_{target_subset_size}_*.pt")
            )
            loaded_list = []
            for subset_file in sorted(
                subset_file_list,
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            ):
                loaded = torch.load(subset_file)
                loaded_list.append(loaded)
            output_dict[int(file.split("/")[-1])] = loaded_list

    return output_dict


def load_attribution(
    path, attribution_name="shapley", target_subset_size=None, sample_select=None
):
    file_list = glob.glob(str(Path(path) / "[0-9]*"))
    if sample_select is not None:
        file_list = [
            file for file in file_list if int(file.split("/")[-1]) in sample_select
        ]
    output_dict = {}
    if target_subset_size is None:
        for file in tqdm(file_list):
            loaded = torch.load(Path(file) / f"{attribution_name}_output.pt")

            output_dict[int(file.split("/")[-1])] = loaded
    else:
        for file in tqdm(file_list):
            subset_file_list = glob.glob(
                str(Path(file) / f"{attribution_name}_output_{target_subset_size}_*.pt")
            )
            loaded_list = []
            for subset_file in sorted(
                subset_file_list,
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            ):
                loaded = torch.load(subset_file)
                loaded_list.append(loaded)
            output_dict[int(file.split("/")[-1])] = loaded_list

    return output_dict


def read_eval_results(path):
    file_set = set(
        [
            p
            for p in glob.glob(str(Path(path) / "*.pt"))
            if not (
                (p.split("/")[-1].startswith("shapley_output"))
                or (p.split("/")[-1].startswith("lime_output"))
            )
        ]
    )

    path_grand_null = str(Path(path) / "grand_null.pt")
    file_set.remove(path_grand_null)

    file_list = sorted(list(file_set), key=lambda x: x.split("_")[-2])
    begin_idx = int(file_list[0].split("_")[-2])
    end_idx = int(file_list[0].split("_")[-1].replace(".pt", ""))
    step_size = end_idx - begin_idx

    idx = begin_idx

    path_eval_list = []
    while True:
        path_eval = str(Path(path) / f"mask_eval_{idx}_{idx+step_size}.pt")
        if path_eval in file_set:
            file_set.remove(path_eval)
            path_eval_list.append(path_eval)
        else:
            break
        idx += step_size

    assert len(file_set) == 0, file_set

    grand_null = torch.load(path_grand_null)
    eval_list = [torch.load(path_eval) for path_eval in path_eval_list]

    grand_logits = grand_null["logits"][0]
    grand_masks = grand_null["masks"][0]
    null_logits = grand_null["logits"][1]
    nulll_masks = grand_null["masks"][1]

    eval_logits = np.concatenate(
        [eval_value["logits"] for eval_value in eval_list], axis=0
    )
    eval_masks = np.concatenate(
        [eval_value["masks"] for eval_value in eval_list], axis=0
    )

    return {
        "grand": {"logits": grand_logits, "masks": grand_masks},
        "null": {"logits": null_logits, "masks": nulll_masks},
        "subsets": {"logits": eval_logits, "masks": eval_masks},
    }
