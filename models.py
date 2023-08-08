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
            patch_size = (
                pixel_values.shape[2] * pixel_values.shape[3] / masks.shape[1]
            ) ** (0.5)
            assert patch_size.is_integer(), "The patch size is not an integer"
            patch_size = int(patch_size)

            masks_resize = masks.reshape(
                -1,
                int(pixel_values.shape[2] // patch_size),
                int(pixel_values.shape[3] / patch_size),
            )
            masks_resize = masks_resize.repeat_interleave(
                patch_size, dim=1
            ).repeat_interleave(patch_size, dim=2)
            pixel_values_masked = pixel_values * (masks_resize.unsqueeze(1))
            original_output = model._forward(
                pixel_values=pixel_values_masked, labels=labels, **kwargs
            )
            logits = original_output.logits
            # ipdb.set_trace()

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
                            logits.view(-1, self.num_labels), dim=1
                        ),
                        target=torch.softmax(
                            classifier_logits.view(-1, self.num_labels), dim=1
                        ),
                    )
                elif self.config.problem_type == "multi_label_classification":
                    raise NotImplementedError
            original_output.loss_label = original_output.loss
            original_output.loss = loss

            return original_output

        model._forward = model.forward
        model.forward = surrogate_forward.__get__(model, model.__class__)
        # import types; model.forward = types.MethodType(surrogate_forward, model)

        return model
