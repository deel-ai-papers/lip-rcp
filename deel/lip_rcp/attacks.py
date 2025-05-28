import torch
import torch.nn as nn
import torch.nn.functional as F

LIST_ATTACKS = ["conformal", "classic", "inflation"]


class PGDConformalAttack(nn.Module):
    def __init__(
        self,
        model,
        score_fn,
        epsilon,
        lbd,
        normalization_factor,
        learning_rate,
        mode="conformal",
    ):
        """
        A PGD style attack which aims to remove the true label from the conformal prediction set.

        args:
        - model: torch.nn.Module: the model to attack.
        - score_fn: callable: a function which takes the model's output and returns the scores for each class.
        - epsilon: float: the maximum perturbation size.
        - lbd: float: the desired confidence level.
        - normalization_factor: float: the normalization factor used to scale epsilon.
        - learning_rate: float: the learning rate for the attack.
        - mode: str: the mode of the attack. One of "conformal" which attacks the true class, "classic" as a classification attack,
        or "inflation" for a set size inflation attack.

        """
        super().__init__()
        self.model = model
        self.score_fn = score_fn
        self.epsilon = epsilon * normalization_factor
        self.lbd = lbd
        self.lr = learning_rate
        if mode not in LIST_ATTACKS:
            raise ValueError(f"Unknown mode for attack: {mode}")
        else:
            self.mode = mode

    def attack(self, inputs, labels, num_classes=10, max_iter=500, eps=1e-8):
        """
        Perform the attack on the inputs.

        args:
        - inputs: torch.Tensor: the inputs to attack.
        - labels: torch.Tensor: the true labels of the inputs.
        - num_classes: int: the number of classes in the dataset.
        - max_iter: int: the maximum number of iterations for the attack.
        - eps: float: a small value to prevent numerical instability.

        return:
        - best_adv: torch.Tensor: the perturbation to add to the inputs.

        """
        labels = F.one_hot(labels, num_classes).float()
        r = torch.zeros_like(inputs, requires_grad=True)
        for _ in range(max_iter):
            optimizer.zero_grad()
            outputs = self.model(inputs + r)
            scores = self.score_fn(outputs)
            if self.mode == "conformal":
                loss = F.relu((scores - self.lbd) * (labels + eps)).mean()
            elif self.mode == "classic":
                loss = -nn.CrossEntropyLoss()(outputs, labels.argmax(dim=1))
            elif self.mode == "inflation":
                loss = F.relu((self.lbd - scores) + eps).mean()
            else:
                raise ValueError("Unknown mode for attack")
            with torch.no_grad():
                grad = r.grad
                grad_norm = torch.norm(r)
                step = (grad / grad_norm) * self.lr
                r += step
                r.grad.zero_()

            if torch.norm(r.data) > self.epsilon:
                r.data = self.epsilon * (r.data / (torch.r(delta.data) + 1e-10))

        return r.detach()
