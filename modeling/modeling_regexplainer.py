import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.vit.modeling_vit import ViTLayer

from modeling.modeling_surrogate import SurrogateForImageClassification


class RegExplainerForImageClassificationConfig(PretrainedConfig):
    """
    Configuration class for the explainer and surrogate models.

    Attributes:
        surrogate_* (str, Optional): Configurations for the surrogate model.
        explainer_* (str, Optional): Configurations for the explainer model.
    """

    def __init__(
        self,
        surrogate_pretrained_model_name_or_path=None,
        surrogate_config=None,
        surrogate_from_tf=False,
        surrogate_cache_dir=None,
        surrogate_revision=None,
        surrogate_token=None,
        surrogate_ignore_mismatched_sizes=None,
        explainer_pretrained_model_name_or_path="google/vit-base-patch16-224",
        explainer_config=None,
        explainer_from_tf=False,
        explainer_cache_dir=None,
        explainer_revision=None,
        explainer_token=None,
        explainer_ignore_mismatched_sizes=None,
        **kwargs,
    ):
        """
        Initialize configurations for both surrogate and explainer models.
        """
        assert surrogate_pretrained_model_name_or_path is None or isinstance(
            surrogate_pretrained_model_name_or_path, str
        )
        assert isinstance(explainer_pretrained_model_name_or_path, str)

        self.surrogate_pretrained_model_name_or_path = (
            surrogate_pretrained_model_name_or_path
        )
        self.surrogate_config = surrogate_config
        self.surrogate_from_tf = surrogate_from_tf
        self.surrogate_cache_dir = surrogate_cache_dir
        self.surrogate_model_revision = surrogate_revision
        self.surrogate_token = surrogate_token
        self.surrogate_ignore_mismatched_sizes = surrogate_ignore_mismatched_sizes

        self.explainer_pretrained_model_name_or_path = (
            explainer_pretrained_model_name_or_path
        )
        self.explainer_config = explainer_config
        self.explainer_from_tf = explainer_from_tf
        self.explainer_cache_dir = explainer_cache_dir
        self.explainer_model_revision = explainer_revision
        self.explainer_token = explainer_token
        self.explainer_ignore_mismatched_sizes = explainer_ignore_mismatched_sizes

        super().__init__(**kwargs)


class RegExplainerForImageClassification(PreTrainedModel):
    """
    Custom explainer model that uses both a surrogate and explainer model.

    Attributes:
        surrogate (SurrogateForImageClassification): The surrogate model.
        explainer (SurrogateForImageClassification): The explainer model.
    """

    config_class = RegExplainerForImageClassificationConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    ):
        """
        Initialize the explainer model with surrogate and explainer components.
        """
        super().__init__(config)

        # Load surrogate model if provided.
        if config.surrogate_pretrained_model_name_or_path is not None:
            self.surrogate = SurrogateForImageClassification.from_pretrained(
                pretrained_model_name_or_path=config.surrogate_pretrained_model_name_or_path,
                from_tf=config.surrogate_from_tf,
                cache_dir=config.surrogate_cache_dir,
                revision=config.surrogate_model_revision,
                token=config.surrogate_token,
                ignore_mismatched_sizes=config.surrogate_ignore_mismatched_sizes,
            )
            self.surrogate.classifier = None

            # Freeze surrogate parameters to avoid updating them during training.
            for param in self.surrogate.parameters():
                param.requires_grad = False
        else:
            self.surrogate = None

        # Load explainer model.
        self.explainer = SurrogateForImageClassification.from_pretrained(
            pretrained_model_name_or_path=config.explainer_pretrained_model_name_or_path,
            from_tf=config.explainer_from_tf,
            cache_dir=config.explainer_cache_dir,
            revision=config.explainer_model_revision,
            token=config.explainer_token,
            ignore_mismatched_sizes=config.explainer_ignore_mismatched_sizes,
        )

        # Initialize attention and MLP layers.
        self.attention_blocks = nn.ModuleList(
            [ViTLayer(config=self.explainer.surrogate.config)]
        )
        self.mlp_blocks = nn.Sequential(
            *[
                nn.LayerNorm(
                    self.explainer.surrogate.config.hidden_size,
                    eps=self.explainer.surrogate.config.layer_norm_eps,
                ),
                nn.Linear(
                    in_features=self.explainer.surrogate.config.hidden_size,
                    out_features=4 * self.explainer.surrogate.config.hidden_size,
                ),
                ACT2FN["gelu"],
                nn.Linear(
                    in_features=4 * self.explainer.surrogate.config.hidden_size,
                    out_features=4 * self.explainer.surrogate.config.hidden_size,
                ),
                ACT2FN["gelu"],
                nn.Linear(
                    in_features=4 * self.explainer.surrogate.config.hidden_size,
                    out_features=self.surrogate.surrogate.config.num_labels,
                ),
            ]
        )

        # Normalization and output link functions.
        self.normalization = (
            lambda pred, grand, null: pred
            + ((grand - null) - torch.sum(pred, dim=1)).unsqueeze(1) / pred.shape[1]
        )

        self.link = nn.Softmax(dim=2)

        # Ensure label consistency between models.
        assert (
            self.explainer.surrogate.num_labels == self.surrogate.surrogate.num_labels
        )

    def grand(self, pixel_values):
        """
        Compute predictions for the grand coalition (all features included).

        Args:
            pixel_values (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predictions for the grand coalition.
        """
        self.surrogate.eval()
        with torch.no_grad():
            grand = self.link(
                self.surrogate(
                    pixel_values=pixel_values,
                    # (batch, channel, height, weight)
                    masks=torch.ones(
                        (pixel_values.shape[0], 1, 196),
                        device=pixel_values.device,
                    ),
                    return_loss=False,
                    # (batch, num_players)
                )["logits"]
            )[
                :, 0, :
            ]  # (batch, num_classes)

        return grand

    def null(self, pixel_values):
        """
        Compute predictions for the null coalition (no features included).

        Args:
            pixel_values (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predictions for the null coalition.
        """
        if hasattr(self, "surrogate_null"):
            return self.surrogate_null
        else:
            self.surrogate.eval()
            with torch.no_grad():
                self.surrogate_null = self.link(
                    self.surrogate(
                        pixel_values=pixel_values[0:1],
                        masks=torch.zeros(
                            (1, 1, 196),
                            device=pixel_values.device,
                        ),
                        return_loss=False,
                    )["logits"]
                )[
                    :, 0, :
                ]  # (batch, channel, height, weight) -> (1, num_classes)
            return self.surrogate_null

    def forward(
        self, pixel_values, shapley_values=None, labels=None, return_loss=True, **kwargs
    ):
        """
        Forward pass through the explainer model.

        Args:
            pixel_values (torch.Tensor): Input images.
            shapley_values (Optional[torch.Tensor]): Target Shapley values for loss computation.
            labels (Optional[torch.Tensor]): Ground truth labels.
            return_loss (bool): Whether to compute and return the loss.

        Returns:
            SemanticSegmenterOutput: Model output containing predictions and loss (if applicable).
        """
        output = self.explainer.surrogate(
            pixel_values=pixel_values, output_hidden_states=True
        )
        hidden_states = output["hidden_states"][-1]

        for _, attention_layer in enumerate(self.attention_blocks):
            hidden_states = attention_layer(hidden_states=hidden_states)
            hidden_states = hidden_states[0]  # (batch, 1+num_players, hidden_size)

        pred = self.mlp_blocks(
            hidden_states[:, 1:, :]
        ).tanh()  # (batch, num_players, num_classes)

        loss = None

        if return_loss:
            value_diff = 196 * F.mse_loss(
                input=pred, target=shapley_values.type(pred.dtype), reduction="mean"
            )  # (batch, num_players, num_classes), (batch, num_players, num_classes)

            loss = value_diff
        return SemanticSegmenterOutput(
            loss=loss,
            logits=pred.transpose(
                1, 2
            ),  # (batch, num_players, num_classes) -> (batch, num_classes, num_players)
        )
