import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import KLDivLoss
from torch.nn import functional as F
from transformers import (
    AutoModelForImageClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ImageClassifierOutput, SemanticSegmenterOutput
from transformers.models.vit.modeling_vit import ViTLayer


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
            # labels=labels.repeat_interleave(num_mask_samples, dim=0),
            # **kwargs,
            # **{i:  for i in kwargs if i != "labels"},
        )

        loss = None

        if return_loss:
            self.classifier.eval()
            with torch.no_grad():
                classifier_output = self.classifier(pixel_values=pixel_values)

            # print(
            #     "classifier",
            #     classifier_output["logits"]
            #     .view(-1, self.surrogate.num_labels)
            #     .softmax(dim=1)
            #     .max(),
            # )
            # print(
            #     "surrogate",
            #     surrogate_output["logits"]
            #     .view(-1, self.surrogate.num_labels)
            #     .softmax(dim=1)
            #     .max(),
            # )

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

        return ImageClassifierOutput(
            loss=loss,
            logits=surrogate_output.logits.reshape(
                pixel_values.shape[0],
                num_mask_samples,
                *surrogate_output.logits.shape[1:],
            ),
        )


import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import KLDivLoss
from transformers import (
    AutoModelForImageClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig


class ExplainerForImageClassificationConfig(PretrainedConfig):
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
class ExplainerForImageClassification(PreTrainedModel):
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

        self.explainer = SurrogateForImageClassification.from_pretrained(
            pretrained_model_name_or_path=config.explainer_pretrained_model_name_or_path,
            from_tf=config.explainer_from_tf,
            cache_dir=config.explainer_cache_dir,
            revision=config.explainer_model_revision,
            token=config.explainer_token,
            ignore_mismatched_sizes=config.explainer_ignore_mismatched_sizes,
        )
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

        self.normalization = (
            lambda pred, grand, null: pred
            + ((grand - null) - torch.sum(pred, dim=1)).unsqueeze(1) / pred.shape[1]
        )

        self.link = nn.Softmax(dim=2)

        assert (
            self.explainer.surrogate.num_labels == self.surrogate.surrogate.num_labels
        )

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
        output = self.explainer.surrogate(
            pixel_values=pixel_values, output_hidden_states=True
        )
        hidden_states = output["hidden_states"][-1]

        # output = self.backbone(x=pixel_values)
        # embedding_cls, embedding_tokens = output["x"], output["x_others"]

        # if self.hparams.explainer_head_include_cls:
        #     embedding_all = torch.cat(
        #         [embedding_cls.unsqueeze(dim=1), embedding_tokens], dim=1
        #     )
        # else:
        #     embedding_all = embedding_tokens

        for _, attention_layer in enumerate(self.attention_blocks):
            hidden_states = attention_layer(hidden_states=hidden_states)
            hidden_states = hidden_states[0]  # (batch, 1+num_players, hidden_size)

        # import ipdb

        pred = self.mlp_blocks(
            hidden_states[:, 1:, :]
        ).tanh()  # (batch, num_players, num_classes)
        # if self.hparams.explainer_head_include_cls:
        #     pred = self.mlps(last_hidden_state)[:, 1:]
        # else:
        #     pred = self.mlps(last_hidden_state)

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

        loss = None
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
