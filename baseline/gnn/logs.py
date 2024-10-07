from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    PearsonCorrCoef,
)


def log(pl_module, target, preds, prefix: str, batch_size: int, task_type="regression"):
    mae_metric = MeanAbsoluteError()
    mse_metric = MeanSquaredError()
    r2score_metric = R2Score()
    pearson_metric = PearsonCorrCoef()

    mae = mae_metric(target, preds)
    mse = mse_metric(target, preds)
    r2score = r2score_metric(target, preds)
    pearson = pearson_metric(target, preds)

    pl_module.log(f"{prefix}_mae", mae, batch_size=batch_size, sync_dist=True)
    pl_module.log(f"{prefix}_mse", mse, batch_size=batch_size, sync_dist=True)
    pl_module.log(f"{prefix}_r2", r2score, batch_size=batch_size, sync_dist=True)
    pl_module.log(f"{prefix}_pearson", pearson, batch_size=batch_size, sync_dist=True)
