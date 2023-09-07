from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
    max_validation_samples: Optional[int] = field(
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
