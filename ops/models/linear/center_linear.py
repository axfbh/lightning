import torch
import torch.nn as nn


class CenterLinear(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLinear, self).__init__()

        self.centers = nn.Parameter(torch.FloatTensor(feat_dim, num_classes).to(device))
        nn.init.xavier_uniform_(self.centers.data)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim).
            target: ground truth labels with shape (batch_size).
        """
        b = inputs.size(0)
        cyi = self.centers[targets]
        return torch.pow(torch.linalg.norm(inputs - cyi), 2) / b


def center_loss(inputs, targets, centers, num_classes):
    batch_size = inputs.size(0)
    distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + torch.pow(centers, 2).sum(
        dim=1, keepdim=True).expand(num_classes, batch_size).t()
    distmat.addmm_(1, -2, inputs, centers.t())

    classes = torch.arange(num_classes).long()
    targets = targets.unsqueeze(1).expand(batch_size, num_classes)
    mask = targets.eq(classes.expand(batch_size, num_classes))

    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
    return loss


# if __name__ == '__main__':
#     x = torch.tensor([[-8, 6, 0, -15],
#                       [-6, 5, 7, 0],
#                       [-6, 4, 7, 2]], dtype=torch.float32)
#     labels = torch.tensor([0, 1, 1])
#     num_classes = 2
#     dim = 4
#
#     critiern = CenterLinear(num_classes, dim, 'cpu')
#     print(critiern(x, labels))
#
#     loss = center_loss(x, labels, critiern.centers, num_classes)
#     print(loss)
