"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from jmp.configs.finetune.jmp_l import jmp_l_ft_config_
# from jmp.configs.finetune.qmof import jmp_l_qmof_config_
from jmp.configs.finetune.coremof import jmp_l_qmof_config_
from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
# from jmp.tasks.finetune.qmof import QMOFConfig, QMOFModel
from jmp.tasks.finetune.coremof import QMOFConfig, QMOFModel

ckpt_path = Path("./ckpt/jmp-l.pt")
base_path = Path("./data/coremof")

# We create a list of all configurations that we want to run.
configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []

from all_configs import qmof_configs

# config, _ = qmof_configs()
config = QMOFConfig.draft()
jmp_l_ft_config_(config, ckpt_path)  # This loads the base JMP-L fine-tuning config
# This loads the rMD17-specific configuration
jmp_l_qmof_config_(config, base_path, target="y")
config = config.finalize()  # Actually construct the config object
# print(config)

configs.append((config, QMOFModel))


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

    trainer = Trainer(config, max_epochs=300, min_epochs=0, devices=[0, 1, 2, 3, 4, 5, 6, 7])
    trainer.fit(model)


runner = Runner(run)
# runner.fast_dev_run(configs)
runner(configs)
