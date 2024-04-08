# SpbNet

SpbNet is the official implementation of the paper.

## The demo to fine-tune SpbNet

A example to finetune SpbNet has been uploaded to [Figshare](https://figshare.com/projects/spbnet/200692). To run this demo, download the `demo.tar.gz` file and extract it. The directory should be look like

```txt
Root
├── demo
|  ├── ckpt
|  ├── config.example.yaml
|  ├── data
|  |  ├── benchmark.csv
|  |  ├── benchmark.filter.csv
|  |  ├── benchmark.test.csv
|  |  ├── benchmark.train.csv
|  |  ├── benchmark.validate.csv
|  |  ├── cif
|  |  └── spbnet
|  ├── logs
|  |  └── hmof
|  └── main.py
└── demo.tar.gz
```

To finetune, you should first download the pretrained weight from [Figshare](https://figshare.com/projects/spbnet/200692). Put the weight to `demo/ckpt` directory. Then you can install spbnet and finetune spbnet.

```sh
cd demo
# optional: conda create -n spbtest python=3.10
# optional: conda activate spbtest
pip install spbnet
python ./finetune.py
python ./predict.py
python ./feat.py
```

The log will be put in the `logs` directory. We have provided the expected result in the `logs/CO2-298-2.5/version_0` direcotry.

After fine-tuned for `30` epochs, the result should look like:

```sh
---------------------------------------
	Test metric		DataLoader 0
---------------------------------------
	test_mae		1.5358973344167073
	test_mse		3.877110533444727
	test_r2			0.3504098369900308
---------------------------------------
```

The predicted result can be found in the log directory, which should look like `logs/hmof/CO2-298-2.5/version_0/test_result.csv`.

## Prepare

To finetune spbnet, we recommend to make a new directory.

```sh
mkdir test
cd test
```

## Install

### Depedencies

All the code is tested on Linux. We recommend using Linux to reproduce or use SpbNet.

SpbNet depends on

```python
python>=3.8
```

### Installation

SpbNet is uploaded to [Pypi](https://pypi.org). Simply use

```shell
pip install spbnet
```

### Download weight

The weight has been uploaded in [Figshare](https://figshare.com/projects/spbnet/200692).

Save the weight to your directory, such as `./ckpt/spbnet.180k.ckpt`

Your directory should look like:

```txt
- test
    - ckpt
        spbnet.180k.ckpt
```

## Data Preprocessing

SpbNet need to preprocess the `cif` format files to obtain structure and potential energy embedding.

### Install GRIDAY

The code to generate energy grid depend on `make` and `g++`. Use the following command to install.

```shell
spbnet install-make
spbnet make-griday
```

The first command will install `make` and C++ compiler tools via conda.
The second command will compile the code to generate energy grid.

### Build Data

SpbNet has provided the command to preprocess data.

To use SpbNet, please provide your dataset first. Your dataset should look like

```txt
- test
    - ckpt
        spbnet.180k.ckpt
    - data
        - cif
            mof1.cif
            mof2.cif
            ...
        benchmark.csv
```

The `benchmark.csv` contains the label data, which should look like

```txt
cifid,CO2
mof1,1.29
mof2,3.81
...
```

The `cifid` colume is needed, while other colums represent the tasks needed to be predicted.

Then use the following command to build data:

```sh
spbnet build-data --root-dir ./data
```

SpbNet will produce the following files:

```txt
- test
    - ckpt
        spbnet.180k.ckpt
    - data
        - cif
            - mof1.cif
            - mof2.cif
            ...
        - spbnet
            - graphdata
                - mof1.graphdata
                - mof2.graphdata
                ...
            - grid
                - mof1.grid
                - mof2.grid
                ...
            - griddata8
                - mof1.npy
                - mof2.npy
                ...
        benchmark.csv
```

### Split Labels

You can split the label data to test the SpbNet's performance. We have provide a command to do this.

```sh
spbnet filter-data --root-dir ./data
```

This command will look up the `benchmark.csv` to check if all the cif files are correctly preprocessed.
In addtion, `filter-data` will filter all the outlier point according to `Q1 - outlier * IQR` and `Q3 + outlier * IQR`.
By default, `outlier` is set to `5`. You can use `spbnet filter-data --root-dir PATH/TO/YOUR/ROOT_DIR --outlier -1` to cancel this behavior.

If correctly preprocessed, the directory should look like

```txt
- test
    - ckpt
        spbnet.180k.ckpt
    - data
        - cif
            xxx.cif
        - spbnet
            - graphdata
                xxx.graphdata
            - grid
                xxx.grid
            - griddata8
                xxx.npy
        benchmark.csv
        benchmark.filter.csv
        benchmark.train.csv
        benchmark.validate.csv
        benchmark.test.csv
```

You can check if the data is correctly preprocessed by

```sh
spbnet check-data --root-dir ./data
```

This command will automatically check the `benchmark.train.csv`, `benchmark.validate.csv` and `benchmark.test.csv`.

## Finetune

After data preprocessing, you can finetune SpbNet.

To configure spbnet, provide a configuration file, such as `config.example.yaml`. The file should look like

```yaml
ckpt: './ckpt/spbnet.18w.ckpt'
data_dir: './data'
id_prop: './data/benchmark.csv'
task: 'CO2-298-2.5'
log_dir: './logs/hmof'
```

- ckpt: pPath to the checkpoint to finetune
- data_dir: The root directory of data
- id_prop: The label data. If it is set to `benchmark.csv`, spbnet will automatically find `benchmark.train.csv`, `benchmark.validate.csv` and `benchmark.test.csv`
- task: The task to train. Should be one of the colums in the `benchmark.csv`
- log_dir: The logger directory. Used by `pytorch-lightning`

NOTE: More configuration can be found in github repository. An important configuration is `max_graph_len`, which is max length of tokens, since SpbNet is based on Transformer architecture. To choose `max_graph_len`, you can estimate the average atom number using this command first.

```sh
spbnet calc-atomnum --root-dir ./data/cif
```

We recommend to choose the number most close to the average number of atoms of your dataset from `512`, `768`, `1024`. For `hMOF` and `CoREMOF` dataset, we recommend `512` (default). For tobacco dataset, we recommend to use `1024`.

Thus, your directory should look like:

```txt
- test
    - ckpt
        spbnet.180k.ckpt
    - data
        - cif
            xxx.cif
        - spbnet
            - graphdata
                xxx.graphdata
            - grid
                xxx.grid
            - griddata8
                xxx.npy
        benchmark.csv
        benchmark.filter.csv
        benchmark.train.csv
        benchmark.validate.csv
        benchmark.test.csv
    config.example.yaml
    main.py
```

The `main.py` should contain the code to finetune spbnet, which should like:

```python
import spbnet

spbnet.finetune("./config.example.yaml")
```

After finetuning, the result (checkpoint and test result) should be saved in the directory specified by `log_dir`.
The path may look like: `./logs/hmof/CO2-298-2.5/version_0`.

Your directory should look like:

```txt
- test
    - ckpt
        spbnet.180k.ckpt
    - data
        ...
    - logs
        ...
    config.example.yaml
    main.py
```

## Predict

After finetune, the checkpoints and hyperparamters should be found in directory like `./tests/logs/hmof/CO2-298-2.5/version_0`. Due to that SpbNet will automatically normalize the training data during training to increase training stability. SpbNet uses mean-variance normalization. The mean and std used can be found in `.../version_0/hparams.yaml`.

Based on the finetuned checkpoint and mean, std. You can predict the target property. First provide the `data_dir` and the `id_prop` csv file. The `id_prop` file should looks like

```txt
cifid
mof1
mof2
...
```

Then prepare a configuration `yaml` file like

```yaml
ckpt: './logs/hmof/CO2-298-2.5/version_0/checkpoints/last.ckpt'
data_dir: './data/spbnet'
id_prop: './data/benchmark.test.csv' # the id_prop file
log_dir: './predict'

mean: 5.325830404166666
std: 2.6947958848152913
```

Then predict

```python
import spbnet

spbnet.predict("./config.predict.yaml")
```

The `test_results.csv` will be saved in directory like `predict/version_0`. It shoud look like

```csv
cifid,predict
hMOF-4000155,12.504342079162598
hMOF-5024342,2.8985860347747803
hMOF-25731,4.6314473152160645
```

## Visualize

SpbNet provide visualization of attention score and atom grid.

To visualize cif file, make a new directory and prepare a `cif` format file. Such as

```txt
- test
    ...
    - vis
        - cif
            mof1.cif
```

Change to the `vis` directory and build modal data:

```sh
cd vis
spbnet build-modal-data --cif-path ./cif/mof1.cif
```

By default, SpbNet will make a `modal` directory under `vis` directory. The directory should look like:

```txt
- test
    ...
    - vis
        - cif
            mof1.cif
        - modal
            - attn
            - energycell
                mof1.cif
            - graphdata
                mof1.graphdata
            - grid
                mof1.grid
            - griddata
                mof1.griddata
            - griddata8
                mof1.griddata8
            - mol
            - supercell
                mof1.cif
            - total
```

Now you can get attention score.

```sh
spbnet attn --cif-dir ./cif/mof1.cif --modal-dir ./modal --ckpt PATH/TO/YOUR/CKPT
```

SpbNet will make an `attn` directory under your current directory, with the `mof1.html` file.

Now open the `mof1.html`. You should see `3dmol` like the following.

![Cooperative](./doc/img/cooperative.png)
