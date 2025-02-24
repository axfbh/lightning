import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import MLP
from torchvision.ops.boxes import box_convert


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DetrHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_classes):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        hidden_channels = [k for k in h + [output_dim]]

        self.bbox_embed = MLP(input_dim, hidden_channels)
        self.class_embed = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, orig_target_sizes):
        outputs_coord, outputs_class = self.bbox_embed(x).sigmoid(), self.class_embed(x)
        outputs = {'pred_logits': outputs_class[-1],
                   'pred_boxes': outputs_coord[-1],
                   'aux_outputs': self._set_aux_loss(outputs_class, outputs_coord)}

        if self.training:
            return outputs

        z = self._inference(outputs, orig_target_sizes)
        return z, outputs

    def _inference(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_convert(out_bbox, 'cxcywh', 'xyxy')

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DeformableDetrHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_classes, num_pred):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        hidden_channels = [k for k in h + [output_dim]]

        bbox_embed = MLP(input_dim, hidden_channels)
        class_embed = nn.Linear(hidden_dim, num_classes)

        self.class_embed = nn.ModuleList([class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_pred)])

    def forward(self, x, init_reference, inter_references, orig_target_sizes):

        outputs_classes = []
        outputs_coords = []

        for lvl in range(x.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](x[lvl])
            tmp = self.bbox_embed[lvl](x[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        outputs = {'pred_logits': outputs_class[-1],
                   'pred_boxes': outputs_coord[-1],
                   'aux_outputs': self._set_aux_loss(outputs_class, outputs_coord)}

        if self.training:
            return outputs

        z = self._inference(outputs, orig_target_sizes)
        return z, outputs

    def _inference(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values

        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_convert(out_bbox, 'cxcywh', 'xyxy')
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
