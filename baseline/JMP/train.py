"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from jmp.configs.finetune.jmp_l import jmp_l_ft_config_

from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase

import click

# We create a list of all configurations that we want to run.
configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []

# from jmp.configs.finetune.qmof import jmp_l_qmof_config_
# from jmp.configs.finetune.coremof import jmp_l_qmof_config_
# from jmp.tasks.finetune.qmof import QMOFConfig, QMOFModel
# from jmp.tasks.finetune.coremof import QMOFConfig, QMOFModel

# ckpt_path = Path("./ckpt/jmp-l.pt")
# base_path = Path("./data/qmof")
# # base_path = Path("./data/coremof")

# config = QMOFConfig.draft()
# jmp_l_ft_config_(config, ckpt_path)  # This loads the base JMP-L fine-tuning config
# # This loads the rMD17-specific configuration
# jmp_l_qmof_config_(config, base_path, target="y")
# config = config.finalize()  # Actually construct the config object
# # print(config)

# configs.append((config, QMOFModel))

from all_configs import (
    qmof_configs,
    coremof_configs,
    hmof_configs,
    ch4n2_configs,
    tsd_configs,
)


# print(configs)

from jmp.lightning import Runner, Trainer
from jmp.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    if (ckpt_path := config.meta.get("ckpt_path")) is None:
        raise ValueError("No checkpoint path provided")

    model = model_cls(config)

    # Load the checkpoint
    state_dict = retreive_state_dict_for_finetuning(
        ckpt_path, load_emas=config.meta.get("ema_backbone", False)
    )
    embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
    backbone = filter_state_dict(state_dict, "backbone.")
    model.load_backbone_state_dict(backbone=backbone, embedding=embedding, strict=True)

    trainer = Trainer(config, devices=[0, 1, 2, 3])
    trainer.fit(model)
    # trainer.test(
    #     model,
    #     ckpt_path="./lightning_logs/coremof/4_11_ft_lg_jmp_testing/4wrjzhid/checkpoints/epoch=72-step=22849.ckpt",
    # )


def test(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    model = model_cls(config)

    trainer = Trainer(config, devices=[0])
    # trainer.fit(model)
    trainer.test(
        model,
        ckpt_path='./ckpt/qmof.ckpt',
    )


@click.command()
@click.option("--train-type", "-T", type=str)
@click.option("--dataset", "-D", type=str)
@click.option("--size", "-S", type=str)
def main(train_type: str, dataset: str, size: str):
    assert train_type == "train" or train_type == "test"
    assert dataset in ["coremof", "qmof", "tsd", "ch4n2", "hmof"]
    assert size in ["small", "large"]

    if dataset == "coremof":
        configs.append(coremof_configs(size))
    elif dataset == "qmof":
        configs.append(qmof_configs(size))
    elif dataset == "tsd":
        configs.append(tsd_configs(size))
    elif dataset == "ch4n2":
        configs.append(ch4n2_configs(size))
    elif dataset == "hmof":
        configs.append(hmof_configs(size))

    if train_type == "train":
        # runner.fast_dev_run(configs)
        runner = Runner(run)
    elif train_type == "test":
        runner = Runner(test)

    runner(configs)


if __name__ == "__main__":
    main()
