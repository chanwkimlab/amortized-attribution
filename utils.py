import numpy as np
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
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
    elif mode == "shapley":
        probs = 1 / (
            np.arange(1, num_features) * (num_features - np.arange(1, num_features))
        )
        probs = probs / probs.sum()
        masks = (
            random_state.rand(num_samples_, num_features)
            > 1
            / num_features
            * random_state.choice(
                np.arange(num_features - 1), p=probs, size=[num_samples_, 1]
            )
        ).astype("int")
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
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _eval_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    def eval_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _eval_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    return {"train_transform": train_transforms, "eval_transform": eval_transforms}
