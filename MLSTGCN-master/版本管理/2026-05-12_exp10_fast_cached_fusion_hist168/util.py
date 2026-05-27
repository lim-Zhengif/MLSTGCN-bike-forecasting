from typing import Any, Callable, Optional
import torch
import numpy as np
try:
    from torchmetrics import Metric
except ImportError:
    from pytorch_lightning.metrics.metric import Metric
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj


def get_L(W):
    edge_index, edge_weight = dense_to_sparse(W)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight)
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    return adj


def get_L_ASTGCN(W):
    edge_index, edge_weight = dense_to_sparse(W)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization="rw")
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    return adj


def _build_mask(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mse(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val=null_val)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val=null_val)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=np.nan, delta=3.0):
    mask = _build_mask(labels, null_val=null_val)
    abs_error = torch.abs(preds - labels)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan, eps=1.0):
    mask = _build_mask(labels, null_val=null_val)
    denominator = torch.clamp(torch.abs(labels), min=eps)
    loss = torch.abs(preds-labels) / denominator
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real, mape_eps=1.0):
    pred = pred.float()
    real = real.float()
    mae = masked_mae(pred, real, np.nan).item()
    mape = masked_mape(pred, real, np.nan, eps=mape_eps).item()
    rmse = masked_rmse(pred, real, np.nan).item()
    return np.round(mae, 4), np.round(mape, 4), np.round(rmse, 4)


class LightningMetric(Metric):

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        mape_eps: float = 1.0,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("y_true", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)
        self.mape_eps = mape_eps

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.y_pred.append(preds)
        self.y_true.append(target)

    def compute(self):
        """
        Computes explained variance over state.
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        feature_dim = y_pred.shape[-1]
        pred_len = y_pred.shape[1]
        # (16, 12, 38, 1)

        y_pred = torch.reshape(y_pred.permute((0, 2, 1, 3)), (-1, pred_len, feature_dim))
        y_true = torch.reshape(y_true.permute((0, 2, 1, 3)), (-1, pred_len, feature_dim))

        metric_dict = {}
        rmse_avg = []
        mae_avg = []
        mape_avg = []
        for i in range(pred_len):
            mae, mape, rmse = metric(
                y_pred[:, i, :].reshape(-1),
                y_true[:, i, :].reshape(-1),
                mape_eps=self.mape_eps,
            )
            idx = i + 1

            metric_dict.update({'rmse_%s' % idx: rmse})
            metric_dict.update({'mae_%s' % idx: mae})
            metric_dict.update({'mape_%s' % idx: mape})

            rmse_avg.append(rmse)
            mae_avg.append(mae)
            mape_avg.append(mape)

            if feature_dim > 1:
                for feature_idx in range(feature_dim):
                    feature_mae, feature_mape, feature_rmse = metric(
                        y_pred[:, i, feature_idx],
                        y_true[:, i, feature_idx],
                        mape_eps=self.mape_eps,
                    )
                    name_suffix = 'f%s_%s' % (feature_idx + 1, idx)
                    metric_dict.update({'rmse_%s' % name_suffix: feature_rmse})
                    metric_dict.update({'mae_%s' % name_suffix: feature_mae})
                    metric_dict.update({'mape_%s' % name_suffix: feature_mape})

        metric_dict.update({'rmse_avg': np.round(np.mean(rmse_avg), 4)})
        metric_dict.update({'mae_avg': np.round(np.mean(mae_avg), 4)})
        metric_dict.update({'mape_avg': np.round(np.mean(mape_avg), 4)})

        return metric_dict


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _match_type(self, data, value):
        if isinstance(data, torch.Tensor) and not isinstance(value, torch.Tensor):
            return torch.as_tensor(value, dtype=data.dtype, device=data.device)
        return value

    def transform(self, data):
        mean = self._match_type(data, self.mean)
        std = self._match_type(data, self.std)
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = self._match_type(data, self.mean)
        std = self._match_type(data, self.std)
        return (data * std) + mean


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


if __name__ == '__main__':

    lightning_metric = LightningMetric()
    batches = 10
    for i in range(batches):
        preds = torch.randn(32, 24, 38, 1)
        target = preds + 0.15

        rmse_batch = lightning_metric(preds, target)
        print(f"Metrics on batch {i}: {rmse_batch}")

    rmse_epoch = lightning_metric.compute()
    print(f"Metrics on all data: {rmse_epoch}")

    lightning_metric.reset()
