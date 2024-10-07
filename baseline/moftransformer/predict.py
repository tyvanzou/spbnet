from pathlib import Path
import moftransformer
from moftransformer.examples import example_path
import click


@cli.command()
@click.option(
    "--root-dir", "-R", type=click.Path(exists=True, file_okay=False, path_type=Path)
)  # eg. ./data/coremof/moftransformer
@click.option("--task", "-T", type=str)
@click.option(
    "--load-path", "-L", type=click.Path(exists=True, file_okay=False, path_type=Path)
)  # eg. ./lightning_logs/coremof/n2/version_0
def main(root_dir, task: str, load_path):
    ckpts = [
        fp for fp in load_path.iterdir() if fp.stem.startswith("epoch")
    ]  # ckpts in the folder should like ['epoch=45-step=16100.ckpt', 'last.ckpt']
    assert len(ckpts) == 1
    load_path = ckpts[0]

    hparams = yaml.full_load((load_path / "hparams.yaml").open("r"))
    mean = hparams["config"]["dicitems"]["mean"]
    std = hparams["config"]["dicitems"]["std"]

    moftransformer.predict(
        root_dir,
        load_path,
        downstream=task,
        split="test",
        save_dir=f"./pred/{root_dir.stem}_{task}",
        mean=mean,
        std=std,
    )


if __name__ == "__main__":
    main()
