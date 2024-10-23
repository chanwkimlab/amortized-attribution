import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ipdb
import torch
from torch import nn
from torch.nn import KLDivLoss
from torch.nn import functional as F
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ImageClassifierOutput, SemanticSegmenterOutput
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.utils import ModelOutput


@dataclass
class ImageSurrogateOutput(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    masks: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SurrogateForImageClassificationConfig(PretrainedConfig):
    def __init__(
        self,
        classifier_pretrained_model_name_or_path=None,
        classifier_config=None,
        classifier_from_tf=False,
        classifier_cache_dir=None,
        classifier_revision=None,
        classifier_token=None,
        classifier_ignore_mismatched_sizes=None,
        surrogate_pretrained_model_name_or_path="google/vit-base-patch16-224",
        surrogate_config=None,
        surrogate_from_tf=False,
        surrogate_cache_dir=None,
        surrogate_revision=None,
        surrogate_token=None,
        surrogate_ignore_mismatched_sizes=None,
        **kwargs,
    ):
        assert classifier_pretrained_model_name_or_path is None or isinstance(
            classifier_pretrained_model_name_or_path, str
        )
        assert isinstance(surrogate_pretrained_model_name_or_path, str)

        self.classifier_pretrained_model_name_or_path = (
            classifier_pretrained_model_name_or_path
        )
        self.classifier_config = classifier_config
        self.classifier_from_tf = classifier_from_tf
        self.classifier_cache_dir = classifier_cache_dir
        self.classifier_model_revision = classifier_revision
        self.classifier_token = classifier_token
        self.classifier_ignore_mismatched_sizes = classifier_ignore_mismatched_sizes

        self.surrogate_pretrained_model_name_or_path = (
            surrogate_pretrained_model_name_or_path
        )
        self.surrogate_config = surrogate_config
        self.surrogate_from_tf = surrogate_from_tf
        self.surrogate_cache_dir = surrogate_cache_dir
        self.surrogate_model_revision = surrogate_revision
        self.surrogate_token = surrogate_token
        self.surrogate_ignore_mismatched_sizes = surrogate_ignore_mismatched_sizes

        super().__init__(**kwargs)


# https://huggingface.co/docs/transformers/custom_models
class SurrogateForImageClassification(PreTrainedModel):
    config_class = SurrogateForImageClassificationConfig

    def __init__(
        self,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    ):
        super().__init__(config)

        self.surrogate = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=config.surrogate_pretrained_model_name_or_path,
            config=config.surrogate_config,
            from_tf=config.surrogate_from_tf,
            cache_dir=config.surrogate_cache_dir,
            revision=config.surrogate_model_revision,
            token=config.surrogate_token,
            ignore_mismatched_sizes=config.surrogate_ignore_mismatched_sizes,
        )

        if config.classifier_pretrained_model_name_or_path is not None:
            self.classifier = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name_or_path=config.classifier_pretrained_model_name_or_path,
                config=config.classifier_config,
                from_tf=config.classifier_from_tf,
                cache_dir=config.classifier_cache_dir,
                revision=config.classifier_model_revision,
                token=config.classifier_token,
                ignore_mismatched_sizes=config.classifier_ignore_mismatched_sizes,
            )
            assert self.surrogate.num_labels == self.classifier.num_labels

            # freeze classifier
            for param in self.classifier.parameters():
                param.requires_grad = False

        else:
            self.classifier = None

    def forward(self, pixel_values, masks, labels=None, return_loss=True, **kwargs):
        # infer patch size
        assert (
            len(pixel_values.shape) == 4
        ), f"The pixel values must be a 4D tensor of shape (num_batches, channels, width, height) but got {pixel_values.shape}"
        assert (
            len(masks.shape) == 3
        ), f"The masks must be a 3D tensor of shape (num_batches, num_mask_samples, num_patches) but got {masks.shape}"
        patch_size = (
            pixel_values.shape[2] * pixel_values.shape[3] / masks.shape[2]
        ) ** (0.5)
        num_mask_samples = masks.shape[1]
        assert patch_size.is_integer(), "The patch size is not an integer"
        patch_size = int(patch_size)

        # resize masks
        masks_resize = masks.reshape(
            -1, masks.shape[2]
        )  # (num_batches, num_mask_samples, num_patches) -> (num_batches * num_mask_samples, num_patches)
        masks_resize = masks_resize.reshape(
            -1,
            int(pixel_values.shape[2] // patch_size),
            int(pixel_values.shape[3] / patch_size),
        )  # (num_batches * num_mask_samples, num_patches) -> (num_batches * num_mask_samples, num_patches_width, num_patches_height)
        masks_resize = masks_resize.repeat_interleave(
            patch_size, dim=1
        ).repeat_interleave(
            patch_size, dim=2
        )  # (num_batches * num_mask_samples, num_patches_width, num_patches_height) -> (num_batches * num_mask_samples, width, height)

        surrogate_output = self.surrogate(
            pixel_values=pixel_values.repeat_interleave(num_mask_samples, dim=0)
            * (
                masks_resize.unsqueeze(1)
            ),  # (num_batches * num_mask_samples, num_channels, width, height) x (num_batches * num_mask_samples, 1, width, height)
        )

        loss = None

        if return_loss:
            self.classifier.eval()
            with torch.no_grad():
                classifier_output = self.classifier(pixel_values=pixel_values)

            if self.surrogate.config.problem_type == "regression":
                raise NotImplementedError
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), classifier_logits.squeeze())
                else:
                    loss = loss_fct(logits, classifier_logits)
            elif self.surrogate.config.problem_type == "single_label_classification":
                loss_fct = KLDivLoss(reduction="batchmean", log_target=False)
                loss = loss_fct(
                    input=torch.log_softmax(
                        surrogate_output["logits"].view(-1, self.surrogate.num_labels),
                        dim=1,
                    ),
                    target=torch.softmax(
                        classifier_output["logits"]
                        .repeat_interleave(num_mask_samples, dim=0)
                        .view(-1, self.surrogate.num_labels),
                        dim=1,
                    ),
                )
            elif self.surrogate.config.problem_type == "multi_label_classification":
                raise NotImplementedError
            else:
                raise RuntimeError(
                    f"Unknown problem type: {self.surrogate.config.problem_type}"
                )

        return ImageSurrogateOutput(
            loss=loss,
            logits=surrogate_output.logits.reshape(
                pixel_values.shape[0],
                num_mask_samples,
                *surrogate_output.logits.shape[1:],
            ),
            masks=masks,
        )
