import torch

# -----------------------------
# Regression metrics (for height in meters)
# -----------------------------
def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """
    Compute common regression metrics.
    Expected shapes:
      - y_true: (N,) or (N, 1)
      - y_pred: (N,) or (N, 1)
    Returns: dict with mae, mse, rmse
    """
    y_true = y_true.detach().float().view(-1)
    y_pred = y_pred.detach().float().view(-1)

    err = y_pred - y_true
    mae = err.abs().mean()
    mse = (err ** 2).mean()
    rmse = torch.sqrt(mse)

    return {
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
    }

# -----------------------------
# Classification metrics (for wave_type, direction)
# -----------------------------
def _to_labels(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Convert logits/probs to labels if needed.
    If y_pred is (N, C) -> argmax to (N,)
    If y_pred is (N,) -> assume already labels
    """
    if y_pred.dim() == 2:
        return torch.argmax(y_pred, dim=1)
    return y_pred.view(-1)

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Accuracy for classification.
    y_true: (N,) long
    y_pred: (N, C) logits or (N,) labels
    """
    y_true = y_true.detach().view(-1).long()
    y_hat = _to_labels(y_pred.detach()).view(-1).long()
    return (y_hat == y_true).float().mean().item()

def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int | None = None, eps: float = 1e-9) -> float:
    """
    Macro F1 score without sklearn.
    y_true: (N,) long
    y_pred: (N, C) logits or (N,) labels
    """
    y_true = y_true.detach().view(-1).long()
    y_hat = _to_labels(y_pred.detach()).view(-1).long()

    if num_classes is None:
        num_classes = int(torch.max(torch.cat([y_true, y_hat])).item()) + 1

    f1_sum = 0.0
    for c in range(num_classes):
        tp = ((y_hat == c) & (y_true == c)).sum().float()
        fp = ((y_hat == c) & (y_true != c)).sum().float()
        fn = ((y_hat != c) & (y_true == c)).sum().float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        f1_sum += f1.item()

    return f1_sum / float(num_classes)
