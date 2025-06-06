import torch
import warnings
import torch.nn as nn

class BinaryFocalLoss(torch.nn.modules.loss._Loss):
    """
    Inherits from torch.nn.modules.loss._Loss. Finds the binary focal loss between each element
    in the input and target tensors.

    Parameters
    -----------
        gamma: float (optional)
            power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean

    Attributes
    -----------
        gamma: float (optional)
            focusing parameter -- power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean
    """
    def __init__(self, gamma=2, reduction='mean'):
        if reduction not in ("sum", "mean", "none"):
            raise AttributeError("Invalid reduction type. Please use 'mean', 'sum', or 'none'.")
        super().__init__(None, None, reduction)
        self.gamma = gamma
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input_tensor, target):
        """
        Compute binary focal loss for an input prediction map and target mask.

        Arguments
        ----------
            input_tensor: torch.Tensor
                input prediction map
            target: torch.Tensor
                target mask

        Returns
        --------
            loss_tensor: torch.Tensor
                binary focal loss, summed, averaged, or raw depending on self.reduction
        """
        #Warn that if sizes don't match errors may occur
        if not target.size() == input_tensor.size():
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size"\
                "({input_tensor.size()}). \n This will likely lead to incorrect results"\
                "due to broadcasting.\n Please ensure they have the same size.",
                stacklevel=2,
            )
        #Broadcast to get sizes/shapes to match
        input_tensor, target = torch.broadcast_tensors(input_tensor, target)
        assert input_tensor.shape == target.shape, "Input and target tensor shapes don't match"

        #Vectorized computation of binary focal loss
        pt_tensor = (target == 0)*(1-input_tensor) + target*input_tensor
        pt_tensor = torch.clamp(pt_tensor, min=self.eps, max=1.0) #Avoid vanishing gradient
        loss_tensor = -(1-pt_tensor)**self.gamma*torch.log(pt_tensor)

        #Apply reduction
        if self.reduction =='none':
            return loss_tensor
        if self.reduction=='mean':
            return torch.mean(loss_tensor)
        #If not none or mean, sum
        return torch.sum(loss_tensor)

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """""
        # print("调查VS的城市",logits.size(), labels.size())
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)
 
        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
 
        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        # print(x,x_sigmoid)

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # print(x, y)
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        # print("-self.loss.sum():",-self.loss.sum())
        return -self.loss.sum()


class MultiLabelContrastiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(MultiLabelContrastiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        focal_loss = focal_weight * bce_loss
        contrastive_loss = self.contrastive_loss(probs, targets)
        total_loss = focal_loss.mean() + contrastive_loss.mean()
        return total_loss

    def contrastive_loss(self, probs, targets):
        batch_size, num_labels = targets.size()
        loss = 0.0
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    similarity = torch.dot(probs[i], probs[j])
                    target_similarity = torch.dot(targets[i], targets[j])
                    loss += (1 - target_similarity) * similarity
        loss /= (batch_size * (batch_size - 1))
        return loss


# outputs=torch.tensor([[ 0.0316,  0.0590, -0.4111, -0.2717],
#         [-0.1602,  0.3017, -0.0032, -0.2134],
#         [-0.3455, -0.0858, -0.4613, -0.4502],
#         [ 0.1486,  0.2357,  0.0338, -0.5084]], device='cuda:0') 

# targets=torch.tensor([[1., 1., 0., 1.],
#         [0., 0., 0., 0.],
#         [0., 0., 1., 1.],
#         [0., 0., 0., 0.]], device='cuda:0')    

# # criterion = BinaryFocalLoss(gamma=2, reduction='mean')
# criterion =AsymmetricLoss()

# loss = criterion(outputs, targets)
# print(loss)

