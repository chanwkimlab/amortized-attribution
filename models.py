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
from transformers.modeling_outputs import ImageClassifierOutput


class SurrogateForImageClassificationConfig(PretrainedConfig):
    def __init__(
        self,
        classifier_pretrained_model_name_or_path=None,
        classifier_config=None,
        classifier_from_tf=None,
        classifier_cache_dir=None,
        classifier_revision=None,
        classifier_token=None,
        classifier_ignore_mismatched_sizes=None,
        surrogate_pretrained_model_name_or_path="google/vit-base-patch16-224",
        surrogate_config=None,
        surrogate_from_tf=None,
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
        else:
            self.classifier = None

        self.surrogate = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=config.surrogate_pretrained_model_name_or_path,
            config=config.surrogate_config,
            from_tf=config.surrogate_from_tf,
            cache_dir=config.surrogate_cache_dir,
            revision=config.surrogate_model_revision,
            token=config.surrogate_token,
            ignore_mismatched_sizes=config.surrogate_ignore_mismatched_sizes,
        )

        assert self.surrogate.num_labels == self.classifier.num_labels

    def forward(self, pixel_values, masks, labels=None, return_loss=True, **kwargs):
        # infer patch size
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
            labels=labels.repeat_interleave(num_mask_samples, dim=0),
            # **kwargs,
            # **{i:  for i in kwargs if i != "labels"},
        )

        loss = 5
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
                loss_fct = KLDivLoss(reduction="batchmean", log_target=True)
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
        # output["loss_label"] = output.loss
        # print("output", output.keys())  #'loss', 'logits'
        # import ipdb

        # ipdb.set_trace()
        # output["loss"] = loss
        # import ipdb

        # output =
        # ipdb.set_trace()
        # print("output", surrogate_output)
        # surrogate_output.loss = loss
        # surrogate_output.logits = surrogate_output.logits.reshape(
        #     pixel_values.shape[0],
        #     num_mask_samples,
        #     *(surrogate_output.logits.shape[1:]),
        # )
        # return surrogate_output

        return ImageClassifierOutput(
            loss=loss,
            logits=surrogate_output.logits.reshape(
                pixel_values.shape[0],
                num_mask_samples,
                *surrogate_output.logits.shape[1:],
            ),
        )


class AutoModelForImageClassificationSurrogate(AutoModelForImageClassification):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        def surrogate_forward(
            self,
            pixel_values,
            masks,
            labels=None,
            classifier_logits=None,
            **kwargs,
        ):
            # # unpack masks
            # if len(masks.shape) == 2:
            #     masks = masks.unsqueeze(1)
            # infer patch size
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

            # pixel_values_masked =
            output = model._forward(
                pixel_values=pixel_values.repeat_interleave(num_mask_samples, dim=0)
                * (
                    masks_resize.unsqueeze(1)
                ),  # (num_batches * num_mask_samples, num_channels, width, height) x (num_batches * num_mask_samples, 1, width, height)
                labels=labels.repeat_interleave(num_mask_samples, dim=0),
                **kwargs,
            )

            loss = None
            if classifier_logits is not None:
                # if self.config.problem_type is None:
                #     if self.num_labels == 1:
                #         raise NotImplementedError
                #         self.config.problem_type = "regression"
                #     elif self.num_labels > 1 and (
                #         labels.dtype == torch.long or labels.dtype == torch.int
                #     ):
                #         self.config.problem_type = "single_label_classification"
                #     else:
                #         raise NotImplementedError
                #         self.config.problem_type = "multi_label_classification"
                if self.config.problem_type == "regression":
                    raise NotImplementedError
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), classifier_logits.squeeze())
                    else:
                        loss = loss_fct(logits, classifier_logits)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = KLDivLoss(reduction="batchmean", log_target=True)
                    loss = loss_fct(
                        input=torch.log_softmax(
                            output["logits"].view(-1, self.num_labels), dim=1
                        ),
                        target=torch.softmax(
                            classifier_logits.repeat_interleave(
                                num_mask_samples, dim=0
                            ).view(-1, self.num_labels),
                            dim=1,
                        ),
                    )
                elif self.config.problem_type == "multi_label_classification":
                    raise NotImplementedError
            # output["loss_label"] = output.loss
            output["loss"] = loss
            output["logits"] = output["logits"].reshape(
                pixel_values.shape[0], num_mask_samples, *output["logits"].shape[1:]
            )

            # ipdb.set_trace()
            # for key in output.keys():
            #     if (
            #         isinstance(output[key], torch.Tensor)
            #         and output[key].shape[0] != num_mask_samples
            #     ):
            #         output[key] = output[key][
            #             [i % num_mask_samples == 0 for i in range(len(output[key]))]
            #         ]
            # # logits = output.logits
            # import ipdb

            # ipdb.set_trace()

            return output

        model._forward = model.forward
        model.forward = surrogate_forward.__get__(model, model.__class__)
        # import types; model.forward = types.MethodType(surrogate_forward, model)

        return model
