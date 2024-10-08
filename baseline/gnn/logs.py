from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    PearsonCorrCoef,
)


def log(pl_module, target, pred, prefix: str, batch_size: int, task_type="regression"):
    device = target.device

    mae_metric = MeanAbsoluteError().to(device)
    mse_metric = MeanSquaredError().to(device)
    r2score_metric = R2Score().to(device)
    pearson_metric = PearsonCorrCoef().to(device)

    mae = mae_metric(target, pred)
    mse = mse_metric(target, pred)
    pearson = pearson_metric(target, pred)

    pl_module.log(f"{prefix}_mae", mae, batch_size=batch_size, sync_dist=True)
    pl_module.log(f"{prefix}_mse", mse, batch_size=batch_size, sync_dist=True)
    pl_module.log(f"{prefix}_pearson", pearson, batch_size=batch_size, sync_dist=True)

    if len(target.shape) == 1 and target.shape[0] == 1: # batch_size == 1
        pass
    else:
        r2score = r2score_metric(target, pred)
        pl_module.log(f"{prefix}_r2", r2score, batch_size=batch_size, sync_dist=True)
