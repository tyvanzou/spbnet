# SpbNet

The repository is the official implementation of paper: "A foundation model for porous materials via computational chemistry knowledge distillation".

Code in this repository includes:

- finetune SpbNet in your dataset
- visualize the attention score
- predict the benchmark dataset to reproduce results in the paper

## Depedencies

SpbNet depends on the following:

- Linux server. We strongly recommend to use linux OS to run our code. All the code is **ONLY** tested in Linux (Ubuntu).
- `python > 3.8`. we recommend to use `python = 3.10`
- `conda`. `conda` is recommended to use to install compile tools and manage environment.
- `gcc, gxx, make`. The code to calculate the energy grid is modifed based on the implementation of [MOFTransformer](https://github.com/hspark1212/MOFTransformer) and relies on these compile tools. We recommend to install these tools using command `python -m spbnet.datamodule.install_griday install-make`. The command will install compile tools based on `conda`.
- GPU. setting `batch_size=8` will require about memory of `5000 MB` (normal MOF, such as structrures in CoREMOF and hMOF).

## Install

First clone the repository and `cd` to the root directory.

run the following shell commands

```sh
# new env
conda create -n spbnet python=3.10
conda activate spbnet

# install requirements
pip install -r ./requirements.txt
# install neccessary make tools, if you don't have `gcc, make, gxx`
python -m spbnet.datamodule.install_griday install-make
# compile code to generate energy grid
pyhotn -m spbnet.datamodule.install_griday make-griday
```

## Prepare data for fine-tuning

We have provided a demo in `test/demo` with a toy dataset in `test/demo/data/toydataset` to test the code and environment.

The demo's directory should look like:

```sh
test/demo
└── data
    └── toydataset
        ├── benchmark.csv
        └── cif
```

Where `cif` folder includes all the cif file of materials, `benchmark.csv` includes `cifid` (corresponding with the name of cif files in the `cif` folder) and target properties to fit.

```csv
cifid,CO2-298-2.5,CH4-298-2.5,N2-298-0.9,H2-77-2,Kr-273-10,Xe-273-10
hMOF-2000924,3.54278,1.31845,0.282032,5.52298,2.74018,2.303
hMOF-7001649,2.79473,1.00696,0.199241,10.642,4.54596,4.03322
hMOF-5068963,1.64653,0.847794,0.181647,5.62688,4.01195,2.56567
...
```

The demo dataset comes from [hMOF(MOFXDB)](https://mof.tech.northwestern.edu).

To prepare data. run

```sh
# note the option `--n-process`, choose the number of process to prepare data.
python -m spbnet.datamodule.buildModal -R ./test/demo/data/toydataset --n-process 8
```

The procedure may take some time. You can use larger `--n-process` to use multiprocessing, default is `1`. `-R` option refer to the `root_dir` containing `cif`, `benchmark.csv`, etc..

After preprocess end, the directory should look like

```sh
test/demo
└── data
    └── toydataset
        ├── benchmark.csv
        ├── cif
        ├── graphdata
        ├── grid
        ├── griddata
        ├── griddata8
        └── total
```

## Filter dataset

Sometimes there are some problems to handle the cif file, e.g. format error, outlier points, etc.. It's helpful to filter dataset. SpbNet provide the following command to do this.

```sh
python -m spbnet.datamodule.filterData -R ./test/demo/data/toydataset
```

SpbNet will automatically read the `benchmark.csv` file in the root_dir specified by `-R` option and remove all the structure unsuccessfully preprocessed. After filtering, the dataset should look like:

```sh
test/demo
└── data
    └── toydataset
        ├── benchmark.csv
        ├── benchmark.filter.csv
        ├── cif
        ├── graphdata
        ├── grid
        ├── griddata
        ├── griddata8
        └── total
```

## Download pre-trained weights

To finetune, you should first download the pretrained weight from [Figshare](https://figshare.com/projects/spbnet/200692) and unextracted. Put the weight to `ckpt` directory. The checkpoint should include `spbnet.160k.ckpt`, etc..

## Finetune spbnet on target task

You can fine-tune SpbNet based on a single `config.yaml` file.

The config file of the demo has been provided in `config.finetune.yaml`. The main configuration adjustable includes:

```yaml
epochs: 10
ckpt: ./ckpt/spbnet.160k.ckpt

root_dir: ./test/demo/data/toydataset
id_prop: benchmark.filter
task: CO2-298-2.5
task_type: regression
log_dir: ./test/demo/logs
device: [0]

batch_size: 8
accumulate_grad_batches: 4
```

Note the final option. SpbNet should be trained on GPU.

Finally, run

```sh
python -m spbnet.finetune --config-path ./config.finetune.yaml
```

SpbNet will automatically read `test/demo/data/toydataset/benchmark.filter.csv` and split it into `5:1:1` to train.

After training for `10` epoches, the result should look like:

```sh
──────────────────────────────────────────────────────────
       Test metric             DataLoader 0
──────────────────────────────────────────────────────────
        test_mae            1.2196654031674066
        test_mse             2.301900389589154
         test_p              0.862175703048706
         test_r2            0.5239504770570731
──────────────────────────────────────────────────────────
```
And the directory should looks like:

```sh
test/demo
├── data
│   └── toydataset
│       ├── benchmark.csv
│       ├── benchmark.filter.csv
│       ├── benchmark.filter.test.csv
│       ├── benchmark.filter.train.csv
│       ├── benchmark.filter.val.csv
│       ├── cif
│       ├── graphdata
│       ├── grid
│       ├── griddata
│       ├── griddata8
│       └── total
└── logs
    └── CO2-298-2.5
        └── version_0
```

The test_result can be found in `test/demo/logs/CO2-298-2.5/version_0/test_result.csv`. It looks like:

```csv
cifid,target,predict
hMOF-5010350,9.906319618225098,7.0625
hMOF-5079604,4.121389865875244,5.796875
hMOF-5059119,4.968820095062256,6.9140625
...
```

## Predict

To predict properties using the fine-tuned weight. See the configuration file in `config.predict.yaml`. It looks like

```yaml
ckpt: ./test/demo/logs/CO2-298-2.5/version_0/checkpoints/epoch=7-step=40.ckpt
root_dir: ./test/demo/data/toydataset
modal_folder: "."
id_prop: benchmark.filter.test
log_dir: ./test/demo/predict
device: [0]
```

There should be a result in `./test/demo/predict/version_0/test_result.csv`, which looks like

```csv
cifid,predict
hMOF-5010350,7.0625
hMOF-5079604,5.796875
hMOF-5059119,6.9140625
...
```

## Visulization

SpbNet provide code for visualization the attention score. Attention is the mechanism designed by the famous Transformer model and used by SpbNet. 

To visualize, first install [openbabel](https://openbabel.org)

We provide an example of visualizing attention score of SpbNet in `test/vis`. The directory should look like

```sh
test/vis
├── data
│   └── cif
│       └── cooperative.cif
└── out
    ├── agc
    │   └── cooperative.npy
    └── attn
        └── cooperative.html
```

You can simple download the `cooperative.html` file and open it in the browser. If you need to visualize by yourself. First prepare your `cif` file and put it in `test/vis/data/cif` folder such as `cooperative.cif` provided by SpbNet.

Next, prepare data.

```sh
python -m spbnet.vis.buildModal -C test/vis/data/cif/cooperative.cif -T test/vis/data
```

The new directory should look like:

```sh
test/vis
├── data
│   ├── attn
│   ├── cif
│   ├── energycell
│   ├── graphdata
│   ├── grid
│   ├── griddata
│   ├── griddata8
│   ├── mol
│   ├── supercell
│   └── total
└── out
    ├── agc
    └── attn
```

Then prepare your ckpt file to visualize, such as `CO2.ckpt` provided in the [Figshare](https://figshare.com/projects/spbnet/200692).

Finally, visualize the attention score of the corresponding checkpoint file.

```sh
python -m spbnet.vis.attn -C test/vis/data/cif/cooperative.cif -M test/vis/data --ckpt ./ckpt/CO2.ckpt -O test/vis/out/attn
```

Result should be saved in `test/vis/out/attn/cooperative.html`. Open the html file and you will look something like:

![Attn](./doc/img/cooperative.png)

Due to that each structure's attention score is different, the visualization need to be adjusted. We recommend you to modify the code in `spbnet/vis/template/attn.html` to adjust contrast and size.

```js
attn_weight = attn_weight.map((val) => (val > threshold ? val : 0))

attn_weight = nj.array(attn_weight)
attn_weight = attn_weight.divide(mean)
attn_weight = attn_weight.pow(10)
```

NOTE: currently, SpbNet is only supported to visualize on the basic mode, i.e. SpbNet without charge, grad, (as shown in the Supplementary Information), etc..

## BREAKING CHNAGE

The code for SpbNet has been modified to make the code easier to maintain. We now recommend to run code of SpbNet directly in the cloned root directory instead of install package from pypi. In addition, the format of configuration file has been modified.

- `correction` (control whether to use basis functions) has been changed to `useBasis`
- `id_prop` has been removed the `.csv` suffix
- add `task_type` option to control task type, including `regression` and `classification`
- other options like `useCharge` and `useGrad` are for experiments in the Supplementary Information. We recommend to use the default version of SpbNet for the best and unbiased performance.
