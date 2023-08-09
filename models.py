import torch
from torch.nn import KLDivLoss
from transformers import AutoModelForImageClassification


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
