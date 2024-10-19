import moftransformer
from pathlib import Path
from multiprocessing import Process
import click
import yaml


def run(
    root_dataset,
    downstream,
    mean,
    std,
    save_dir,
    device,
    load_path,
    max_epochs=60,
    batch_size=8,
):

    moftransformer.run(
        root_dataset,
        downstream,
        mean=mean,
        std=std,
        log_dir=save_dir,
        max_epochs=max_epochs,
        batch_size=batch_size,
        devices=device,
        load_path=load_path,
    )


def main():
    config = yaml.full_load(open("config.yaml", 'r'))

    root_dataset = config["root_dataset"]
    downstreams = config["downstreams"]
    means = config["means"]
    stds = config["stds"]
    devices = config["devices"]
    ckpt = config["ckpt"]
    save_dir = config["save_dir"]

    processes = []
    for device, downstream, mean, std in zip(devices, downstreams, means, stds):
        print(f"starting downstream: {downstream}, gpu_idx: {device}")
        process = Process(
            target=run,
            kwargs={
                "root_dataset": root_dataset,
                "downstream": downstream,
                "save_dir": f"{save_dir}/{downstream}",
                "device": device,
                "mean": mean,
                "std": std,
                "load_path": ckpt,
            },
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
