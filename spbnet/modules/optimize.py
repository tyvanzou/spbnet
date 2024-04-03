import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)


def set_scheduler(pl_module):
    optim_config = pl_module.optimizer_config
    lr = optim_config["lr"]
    wd = optim_config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
        "potential_mapper",
    ]
    head_names = ["head"]
    end_lr = optim_config["end_lr"]
    lr_mult = optim_config["lr_mult"]
    decay_power = optim_config["decay_power"]
    optim_type = optim_config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)  # within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and any(bb in n for bb in head_names)  # within head_names
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                # within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps == -1:
        max_steps = pl_module.trainer.estimated_stepping_batches
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = optim_config["warmup_steps"]
    if isinstance(optim_config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


# def get_scheduler(pl_module, optimizer):
#     if pl_module.trainer.max_steps == -1:
#         max_steps = pl_module.trainer.estimated_stepping_batches
#     else:
#         max_steps = pl_module.trainer.max_steps
#     warmup_steps = pl_module.optimizer_config["warmup_steps"]
#     if isinstance(pl_module.optimizer_config["warmup_steps"], float):
#         warmup_steps = int(max_steps * warmup_steps)

#     decay_power = pl_module.optimizer_config["decay_power"]
#     end_lr = pl_module.optimizer_config["end_lr"]

#     if decay_power == "cosine":
#         scheduler = get_cosine_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=warmup_steps,
#             num_training_steps=max_steps,
#         )
#     elif decay_power == "constant":
#         scheduler = get_constant_schedule(
#             optimizer,
#         )
#     elif decay_power == "constant_with_warmup":
#         scheduler = get_constant_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=warmup_steps,
#         )
#     else:
#         scheduler = get_polynomial_decay_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=warmup_steps,
#             num_training_steps=max_steps,
#             lr_end=end_lr,
#             power=decay_power,
#         )

#     return scheduler
