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


class SegExplainerForImageClassificationConfig(PretrainedConfig):
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


# https://huggingface.co/docs/transformers/custom_models
class SegExplainerForImageClassification(PreTrainedModel):
    config_class = ExplainerForImageClassificationConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    ):
        super().__init__(config)
        # import ipdb

        # ipdb.set_trace()
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

            # freeze surrogate
            for param in self.surrogate.parameters():
                param.requires_grad = False
        else:
            self.surrogate = None

        self.explainer = AutoModelForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=config.explainer_pretrained_model_name_or_path,
            config=config.explainer_config,
            from_tf=config.explainer_from_tf,
            cache_dir=config.explainer_cache_dir,
            revision=config.explainer_model_revision,
            token=config.explainer_token,
            ignore_mismatched_sizes=config.explainer_ignore_mismatched_sizes,
        )
        # freeze backbone of explainer
        for param in self.explainer.segformer.parameters():
            param.requires_grad = False

        # self.attention_blocks = nn.ModuleList(
        #     [ViTLayer(config=self.explainer.surrogate.config)]
        # )
        # self.mlp_blocks = nn.Sequential(
        #     *[
        #         nn.LayerNorm(
        #             self.explainer.surrogate.config.hidden_size,
        #             eps=self.explainer.surrogate.config.layer_norm_eps,
        #         ),
        #         nn.Linear(
        #             in_features=self.explainer.surrogate.config.hidden_size,
        #             out_features=4 * self.explainer.surrogate.config.hidden_size,
        #         ),
        #         ACT2FN["gelu"],
        #         nn.Linear(
        #             in_features=4 * self.explainer.surrogate.config.hidden_size,
        #             out_features=4 * self.explainer.surrogate.config.hidden_size,
        #         ),
        #         ACT2FN["gelu"],
        #         nn.Linear(
        #             in_features=4 * self.explainer.surrogate.config.hidden_size,
        #             out_features=self.surrogate.surrogate.config.num_labels,
        #         ),
        #     ]
        # )

        self.normalization = (
            lambda pred, grand, null: pred
            + ((grand - null) - torch.sum(pred, dim=1)).unsqueeze(1) / pred.shape[1]
        )

        self.link = nn.Softmax(dim=2)
        assert self.explainer.config.num_labels == self.surrogate.surrogate.num_labels

    def grand(self, pixel_values):
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
        self, pixel_values, masks=None, labels=None, return_loss=True, **kwargs
    ):
        # ipdb.set_trace()
        output = self.explainer(pixel_values=pixel_values)
        # output["logits"].shape
        # x=F.interpolate(output["logits"], size=(14, 14))
        output_logits = (
            output["logits"]
            .reshape(
                (output["logits"].shape[0], output["logits"].shape[1], 4, 14, 4, 14)
            )
            .transpose(3, 4)
            .mean(dim=(2, 3))
        )
        # (batch, num_classes, height_palyer*4, width_player*4) -> (batch, num_classes, 4, height_palyer, 4, width_player) ->  (batch, num_classes, height_palyer, width_player)
        pred = output_logits.reshape(
            (output_logits.shape[0], output_logits.shape[1], -1)
        )
        pred = pred.transpose(1, 2)
        pred = pred.tanh()
        # ipdb.set_trace()

        surrogate_grand = None
        surrogate_null = None

        surrogate_grand = self.grand(
            pixel_values
        )  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(
            pixel_values
        )  # (batch, channel, height, weight) -> (1, num_classes)
        values_pred = self.normalization(
            pred=pred, grand=surrogate_grand, null=surrogate_null
        )  # (batch, num_players, num_classes)

        value_pred_beforenorm_sum = values_pred.sum(
            dim=1
        )  # (batch, num_players, num_classes) -> (batch, num_classes)
        value_pred_beforenorm_sum_class = values_pred.sum(
            dim=2
        )  # (batch, num_players, num_classes) -> (batch, num_players)
        # print(
        #     "value_pred_beforenorm_sum",
        #     value_pred_beforenorm_sum[0],
        #     "value_pred_beforenorm_sum_class",
        #     value_pred_beforenorm_sum_class[0],
        # )
        # F.mse_loss(
        #     input=value_pred_beforenorm_sum_class,
        #     target=torch.zeros_like(value_pred_beforenorm_sum_class),
        #     reduction="mean",
        # )
        # 196 * F.mse_loss(
        #     input=value_pred_beforenorm_sum,
        #     target=surrogate_grand - surrogate_null,
        #     reduction="mean",
        # )
        # 196 * F.mse_loss(
        #     input=value_pred_for_regression, target=surrogate_prob, reduction="mean"
        # )

        loss = 999
        if return_loss:
            # infer patch size
            patch_size = (
                pixel_values.shape[2] * pixel_values.shape[3] / masks.shape[2]
            ) ** (0.5)
            num_mask_samples = masks.shape[1]
            assert patch_size.is_integer(), "The patch size is not an integer"
            patch_size = int(patch_size)

            self.surrogate.eval()
            with torch.no_grad():
                surrogate_output = self.surrogate(
                    pixel_values=pixel_values,
                    masks=masks,
                    return_loss=False,
                )  # (batch, channel, height, width), (batch, num_mask_samples, num_players) -> (batch, num_mask_samples, num_classes)
                surrogate_prob = self.link(surrogate_output["logits"])
            value_pred_for_regression = (
                surrogate_null + masks.float() @ values_pred
            )  # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) -> (batch, num_mask_samples, num_classes)
            # import ipdb

            # ipdb.set_trace()
            value_diff = 196 * F.mse_loss(
                input=value_pred_for_regression, target=surrogate_prob, reduction="mean"
            )  # (batch, num_mask_samples, num_classes), (batch, num_mask_samples, num_classes)

            loss = value_diff
        return SemanticSegmenterOutput(
            loss=loss,
            logits=values_pred.transpose(
                1, 2
            ),  # (batch, num_players, num_classes) -> (batch, num_classes, num_players)
        )


# https://huggingface.co/docs/transformers/custom_models
class SegExplainerForImageClassification2(PreTrainedModel):
    config_class = ExplainerForImageClassificationConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    ):
        super().__init__(config)
        # import ipdb

        # ipdb.set_trace()
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

            # freeze surrogate
            for param in self.surrogate.parameters():
                param.requires_grad = False
        else:
            self.surrogate = None

        # self.explainer = AutoModelForSemanticSegmentation.from_pretrained(
        #     pretrained_model_name_or_path=config.explainer_pretrained_model_name_or_path,
        #     config=config.explainer_config,
        #     from_tf=config.explainer_from_tf,
        #     cache_dir=config.explainer_cache_dir,
        #     revision=config.explainer_model_revision,
        #     token=config.explainer_token,
        #     ignore_mismatched_sizes=config.explainer_ignore_mismatched_sizes,
        # )
        # # freeze backbone of explainer
        # for param in self.explainer.segformer.parameters():
        #     param.requires_grad = False

        config.explainer_config.strides = [16, 4, 2, 2]
        self.explainer = AutoModelForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=config.explainer_pretrained_model_name_or_path,
            config=config.explainer_config,
            from_tf=config.explainer_from_tf,
            cache_dir=config.explainer_cache_dir,
            revision=config.explainer_model_revision,
            token=config.explainer_token,
            ignore_mismatched_sizes=config.explainer_ignore_mismatched_sizes,
        )
        # # freeze backbone of explainer
        # for param in self.explainer.segformer.parameters():
        #     param.requires_grad = False

        # self.attention_blocks = nn.ModuleList(
        #     [ViTLayer(config=self.explainer.surrogate.config)]
        # )
        # self.mlp_blocks = nn.Sequential(
        #     *[
        #         nn.LayerNorm(
        #             self.explainer.surrogate.config.hidden_size,
        #             eps=self.explainer.surrogate.config.layer_norm_eps,
        #         ),
        #         nn.Linear(
        #             in_features=self.explainer.surrogate.config.hidden_size,
        #             out_features=4 * self.explainer.surrogate.config.hidden_size,
        #         ),
        #         ACT2FN["gelu"],
        #         nn.Linear(
        #             in_features=4 * self.explainer.surrogate.config.hidden_size,
        #             out_features=4 * self.explainer.surrogate.config.hidden_size,
        #         ),
        #         ACT2FN["gelu"],
        #         nn.Linear(
        #             in_features=4 * self.explainer.surrogate.config.hidden_size,
        #             out_features=self.surrogate.surrogate.config.num_labels,
        #         ),
        #     ]
        # )

        self.normalization = (
            lambda pred, grand, null: pred
            + ((grand - null) - torch.sum(pred, dim=1)).unsqueeze(1) / pred.shape[1]
        )

        self.link = nn.Softmax(dim=2)
        assert self.explainer.config.num_labels == self.surrogate.surrogate.num_labels

    def grand(self, pixel_values):
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
        self, pixel_values, masks=None, labels=None, return_loss=True, **kwargs
    ):
        # ipdb.set_trace()
        output = self.explainer(pixel_values=pixel_values)
        # output["logits"].shape
        # x=F.interpolate(output["logits"], size=(14, 14))
        # output_logits = (
        #     output["logits"]
        #     .reshape(
        #         (output["logits"].shape[0], output["logits"].shape[1], 4, 14, 4, 14)
        #     )
        #     .transpose(3, 4)
        #     .mean(dim=(2, 3))
        # )
        output_logits = (
            output["logits"]
            .reshape(
                (output["logits"].shape[0], output["logits"].shape[1], 1, 14, 1, 14)
            )
            .transpose(3, 4)
            .mean(dim=(2, 3))
        )

        # (batch, num_classes, height_palyer*4, width_player*4) -> (batch, num_classes, 4, height_palyer, 4, width_player) ->  (batch, num_classes, height_palyer, width_player)
        pred = output_logits.reshape(
            (output_logits.shape[0], output_logits.shape[1], -1)
        )
        pred = pred.transpose(1, 2)
        pred = pred.tanh()
        # ipdb.set_trace()

        surrogate_grand = None
        surrogate_null = None

        surrogate_grand = self.grand(
            pixel_values
        )  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(
            pixel_values
        )  # (batch, channel, height, weight) -> (1, num_classes)
        values_pred = self.normalization(
            pred=pred, grand=surrogate_grand, null=surrogate_null
        )  # (batch, num_players, num_classes)

        value_pred_beforenorm_sum = values_pred.sum(
            dim=1
        )  # (batch, num_players, num_classes) -> (batch, num_classes)
        value_pred_beforenorm_sum_class = values_pred.sum(
            dim=2
        )  # (batch, num_players, num_classes) -> (batch, num_players)
        # print(
        #     "value_pred_beforenorm_sum",
        #     value_pred_beforenorm_sum[0],
        #     "value_pred_beforenorm_sum_class",
        #     value_pred_beforenorm_sum_class[0],
        # )
        # F.mse_loss(
        #     input=value_pred_beforenorm_sum_class,
        #     target=torch.zeros_like(value_pred_beforenorm_sum_class),
        #     reduction="mean",
        # )
        # 196 * F.mse_loss(
        #     input=value_pred_beforenorm_sum,
        #     target=surrogate_grand - surrogate_null,
        #     reduction="mean",
        # )
        # 196 * F.mse_loss(
        #     input=value_pred_for_regression, target=surrogate_prob, reduction="mean"
        # )

        loss = 999
        if return_loss:
            # infer patch size
            patch_size = (
                pixel_values.shape[2] * pixel_values.shape[3] / masks.shape[2]
            ) ** (0.5)
            num_mask_samples = masks.shape[1]
            assert patch_size.is_integer(), "The patch size is not an integer"
            patch_size = int(patch_size)

            self.surrogate.eval()
            with torch.no_grad():
                surrogate_output = self.surrogate(
                    pixel_values=pixel_values,
                    masks=masks,
                    return_loss=False,
                )  # (batch, channel, height, width), (batch, num_mask_samples, num_players) -> (batch, num_mask_samples, num_classes)
                surrogate_prob = self.link(surrogate_output["logits"])
            value_pred_for_regression = (
                surrogate_null + masks.float() @ values_pred
            )  # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) -> (batch, num_mask_samples, num_classes)
            # import ipdb

            # ipdb.set_trace()
            value_diff = 196 * F.mse_loss(
                input=value_pred_for_regression, target=surrogate_prob, reduction="mean"
            )  # (batch, num_mask_samples, num_classes), (batch, num_mask_samples, num_classes)

            loss = value_diff
        return SemanticSegmenterOutput(
            loss=loss,
            logits=values_pred.transpose(
                1, 2
            ),  # (batch, num_players, num_classes) -> (batch, num_classes, num_players)
        )
