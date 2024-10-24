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
    """
    Set up a dataset for image classification based on the provided arguments.

    Parameters
    ----------
    data_args : DataTrainingArguments
        Arguments related to data configuration, including dataset name,
        cache directory, and train/validation split ratios.
    other_args : OtherArguments
        Additional arguments, including authentication tokens.

    Returns
    -------
    tuple
        A tuple containing:
        - dataset (DatasetDict): The loaded dataset.
        - labels (List[str]): List of class labels.
        - label2id (dict): Mapping from label names to IDs.
        - id2label (dict): Mapping from IDs to label names.

    Notes
    -----
    - If a dataset name is provided, it is loaded from the hub. If the dataset
      is "frgfm/imagenette", the label column is renamed to "labels".
    - If no dataset name is provided, the dataset is loaded from local image folders.
    - If a validation split is not present, a portion of the training data can
      be split off for validation based on the specified ratio.
    - If a test split is not present, a portion of the validation data can be
      split off for testing based on the specified ratio.
    - The function prepares label mappings for human-readable labels in the
      model's configuration.
    """
    if data_args.dataset_name is not None:
        # Load dataset from HuggingFace hub
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
            # Load dataset from local directories
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
    """
    Load an image from the given path using PIL.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    Image.Image
        The loaded image in RGB mode.
    """
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def get_image_transform(image_processor):
    # Define torchvision transforms to be applied to each image.
    """
    Create and return a dictionary of image transformation functions.

    This function defines two sets of torchvision transformations for image processing:
    an augmentation transform for training and a static transform for validation/testing.
    The transformations include resizing, cropping, normalization, and conversion to tensor.

    Parameters
    ----------
    image_processor : object
        An object containing image processing configurations, including size, mean, and std.

    Returns
    -------
    dict
        A dictionary containing two functions:
        - 'augment_transform': a function to apply augmentation transforms on a batch of images.
        - 'static_transform': a function to apply static transforms on a batch of images.
    """
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
    """
    Configure dataset splits and assign transformations.

    Args
    ----
    dataset : DatasetDict
        The dataset to configure.
    image_processor : object
        Image processor object with size and normalization details.
    training_args : TrainingArguments
        Training configuration.
    data_args : DataTrainingArguments
        Data-related configuration such as sample limits.
    train_augmentation : bool
        Whether to apply augmentation to the training split.
    validation_augmentation : bool
        Whether to apply augmentation to the validation split.
    test_augmentation : bool
        Whether to apply augmentation to the test split.
    logger : Logger
        Logger instance for logging messages.

    Returns
    -------
    DatasetDict
        The configured dataset.
    """
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
                    (
                        "augment_transform"
                        if validation_augmentation
                        else "static_transform"
                    )
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
    """
    Log dataset statistics.

    Parameters
    ----------
    dataset : dict
        Dictionary containing dataset splits (e.g., 'train', 'validation', 'test').
    logger : Logger
        Logger instance to log messages.
    """
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
    random_state=None,
) -> np.array:
    """
    Generate binary masks for feature inclusion/exclusion.

    Parameters
    ----------
    num_features : int
        Total number of features.
    num_mask_samples : int, optional
        Number of masks to generate (default: 1).
    paired_mask_samples : bool, optional
        Whether to generate pairs of complementary masks (default: True).
    mode : str, optional
        Sampling mode ('uniform', 'shapley', 'binomial', 'full', 'empty').
    random_state : RandomState, optional
        Random state generator for reproducibility.

    Returns
    -------
    np.array
        Array of generated masks.
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
        # masks = np.stack([masks, 1 - masks], axis=1).reshape(
        #     num_samples_ * 2, num_features
        # )
        masks = np.hstack([masks, 1 - masks]).reshape(num_samples_ * 2, num_features)

    return masks  # (num_samples, num_masks)


class MaskDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset wrapper that supports generating masks and caching outputs
    for use.

    Attributes
    ----------
    dataset : Dataset
        The underlying dataset.
    num_features : int, optional
        Number of features to generate masks for.
    num_mask_samples : int, optional
        Number of masks to generate per sample.
    paired_mask_samples : bool, optional
        Whether to generate complementary mask pairs.
    mode : str, optional
        Mask generation mode ('uniform', 'shapley', etc.).
    random_seed : int, optional
        Seed for reproducibility.

    Methods
    -------
    __getitem__(idx)
        Returns an item from the dataset with masks and optional cached values.
    set_params(**kwargs)
        Set parameters for mask generation.
    set_cache(**kwargs)
        Enable caching with provided masks and outputs.
    reset_cache_counter()
        Reset the cache access counter.
    __len__()
        Return the size of the dataset.
    """

    def __init__(
        self,
        dataset,
        num_features=None,
        num_mask_samples=None,
        paired_mask_samples=None,
        mode=None,
        random_seed=None,
    ):
        """
        Initialize the MaskDataset.

        Parameters
        ----------
        dataset : Dataset
            The underlying dataset to wrap.
        num_features : int, optional
            Number of features to use for mask generation.
        num_mask_samples : int, optional
            Number of masks to generate per sample.
        paired_mask_samples : bool, optional
            Whether to generate pairs of complementary masks.
        mode : str, optional
            The mode for mask generation ('uniform', 'shapley', etc.).
        random_seed : int, optional
            Seed for reproducibility.
        """
        self.dataset = dataset

        self.use_cache = False

        # for not using cache
        self.num_features = num_features
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self.mode = mode
        self.random_seed = random_seed

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset with masks and optional cached values.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing the sample with optional masks and model outputs.
        """
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
        """
        Set the parameters for mask generation.

        Parameters
        ----------
        kwargs : dict
            Parameters to set (num_features, num_mask_samples, etc.).
        """
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
        """
        Enable caching with provided masks and outputs.

        Parameters
        ----------
        masks : list, optional
            Pre-generated masks.
        model_outputs : list, optional
            Cached model outputs.
        grand_values : list, optional
            Cached grand coalition values.
        null_values : list, optional
            Cached null coalition values.
        shapley_values : list, optional
            Cached Shapley values.
        cache_start_idx : int, optional
            Start index for cached access.
        cache_mask_size : int, optional
            Number of elements to retrieve from cache.
        """
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
        """Reset the cache access counters."""
        self.cache_counter_list = [0 for _ in range(len(self.dataset))]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)


def get_checkpoint(training_args, logger):
    """
    Retrieve the latest checkpoint if available.

    Parameters
    ----------
    training_args : TrainingArguments
        Training configuration containing the output directory and other options.
    logger : Logger
        Logger to record messages.

    Returns
    -------
    str or None
        Path to the latest checkpoint or None if no checkpoint is available.
    """
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
    """
    Load Shapley values from saved files.

    Parameters
    ----------
    path : str
        Path to the directory containing Shapley outputs.
    target_subset_size : int, optional
        Subset size to filter specific outputs.

    Returns
    -------
    dict
        Dictionary of Shapley values indexed by sample ID.
    """
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
    """
    Load attribution values from saved files.

    Parameters
    ----------
    path : str
        Directory containing attribution outputs.
    attribution_name : str, optional
        Name of the attribution method (default: "shapley").
    target_subset_size : int, optional
        Subset size to filter outputs.
    sample_select : list, optional
        List of specific sample IDs to load.

    Returns
    -------
    dict
        Dictionary of attribution values indexed by sample ID.
    """
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
