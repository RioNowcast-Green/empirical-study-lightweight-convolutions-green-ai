# An empirical study using lightweight convolutions to achieve Green AI

This repository contains the source code used in the paper _An Empirical Study Using Lightweight Convolutions to Achieve Green AI_, published at the _Encontro Nacional de Inteligência Artificial e Computacional_ (ENIAC 2026). A link to the paper will be added once it becomes publicly available.

The Deep Learning model used in the experiments was STMixGAN, proposed in _A spatiotemporal mixed-enhanced generative adversarial network for radar-based precipitation nowcasting_.
We also used the authors' publicly available implementation (https://github.com/Helomin/STMixGAN) as the baseline implementation upon which the extensions and modifications of this research were developed.

The Moving MNIST dataset is automatically downloaded by the code and can be used directly.
The TAASRAD19 dataset used in the experiments will be made publicly available after the official publication of the paper, with a link provided in this repository.

## Installing the project dependencies

The project was developed in **Python 3.12.12**.

To install the dependencies, create a Python virtual environment of your choice. Then, run:

```shell
make setup
```

If you want to install the dependencies related to the project's development, run:

```shell
make setup-dev
```

Currently, the only development dependency is a Python code formatter, used to enforce PEP 8 style guidelines.

To apply the formatter, run:

```shell
make format
```

To install the dependencies required to run the experiments, run:

```shell
make setup-experiment
```

## How to run the project

We have two executable scripts: `trainer.py` and `predictor.py`.
For more details on all available parameters to run the scripts above, execute one of the commands below, depending on the desired step:

```shell
python trainer.py --help
python predictor.py --help
```


### Training

To run a training process, use the following command:

```shell
python trainer.py \
    --num_epochs 4 \
    --batch_size 10 \
    --working_dir my_space/save_here \
    --start_epoch_saving 3 \
    --normalize_values \
    --light_conv standard \
    --dataset moving_mnist \
    --result_folder_suffix my_suffix
```

- num_epochs: Number of training epochs
- batch_size: Batch size for each training step
- working_dir: Path to the directory where training outputs will be saved
- start_epoch_saving: Initial epoch from which the Generator and Discriminator weights will be saved
- normalize_values: Normalize data values to the range [0, 1] before training
- light_conv: Name that identifies which lightweight convolution to use. For all available options, refer to the light_conv/conv_module.py script
- dataset: Name that identifies the dataset to be used. Options: moving_mnist and taasrad19
- result_folder_suffix: Suffix name to be appended to the output folder created by the project. If not specified, the suffix will be the execution start datetime, in the format `YYYYMMDD_HHMMSS`

After training, the following file structure will be generated:

```
my_space
├──save_here
│    ├──training_moving_mnist_standard_my_suffix/
│    │    ├── emissions/
│    │    │ └── emissions.csv
│    │    ├── metrics/
│    │    │ ├── execution_metrics_training_standard_moving_mnist.csv
│    │    │ ├── loss_epoch_standard_moving_mnist.csv
│    │    │ └── loss_step_standard_moving_mnist.csv
│    │    ├── weights/
│    │    │ ├── discriminator/
│    │    │ │ └── moving_mnist_dcnet_standard_3.pth
│    │    │ │ └── moving_mnist_dcnet_standard_4.pth
│    │    │ └── generator/
│    │    │   └── moving_mnist_stminet_standard_3.pth
│    │    │   └── moving_mnist_stminet_standard_4.pth
```

The root folder will always follow the naming pattern: `training_[dataset]_[light_conv]_[result_folder_suffix]`.
Inside it, three subfolders are created: CodeCarbon metrics (emissions), training and execution metrics (metrics), and the model weights saved per epoch (weights).

### Prediction (Inference)

To run a prediction, use the following command:

```shell
python predictor.py \ 
    --model_weight_file_path my_space/save_here/training_moving_mnist_standard_my_suffix/weights/generator/moving_mnist_stmixnet_standard_4.pth \ 
    --working_dir my_space/save_here \ 
    --light_conv standard \ 
    --normalize_values \ 
    --dataset moving_mnist \
    --result_folder_suffix my_suffix
```

- model_weight_file_path: Path to the model weights file used to load the model
- working_dir: Path to the directory where prediction outputs will be saved
- normalize_values: Normalize data values to the range [0, 1] before prediction
- light_conv: Name that identifies which lightweight convolution to use. For all available options, refer to the `light_conv/conv_module.py`
- dataset: Name that identifies the dataset to be used. Options: moving_mnist and taasrad19
- result_folder_suffix: Suffix name to be appended to the output folder created by the project. If not specified, the suffix will be the execution start datetime, in the format `YYYYMMDD_HHMMSS`

After prediction, the following file structure will be generated:

```
my_space
├──save_here
│    ├──predict_moving_mnist_standard_my_suffix/
│    │    ├── emissions/
│    │    │   └── emissions.csv
│    │    ├── images/
│    │    │   ├── generated/
│    │    │   └── real/
│    │    ├── metrics/
│    │    │   ├── execution_metrics_predict_standard_moving_mnist.csv
│    │    │   ├── standard_moving_mnist_fss.csv
│    │    │   ├── standard_moving_mnist_image_metrics.csv
│    │    │   ├── standard_moving_mnist_pooled_crps.csv
│    │    │   └── standard_moving_mnist_threshold_metrics.csv
│    │    ├── npy_files/
│    │    │   ├── generated/
│    │    │   └── real/
```

The root folder will always follow the naming pattern: `predict_[dataset]_[light_conv]_YYYYMMDD_HHMMSS`.

Inside it, four subfolders are created: CodeCarbon metrics (`emissions`), predictive performance and execution metrics (`metrics`), images in .png format of the real and generated sequences (`images`), and a folder containing the real and generated images in .npy format (`npy_files`).

The images and npy_files folders are generated optionally. To enable their creation in addition to the other folders, you must pass the flags `--save_sequence_images` and `--save_sequence_npy_files`. This design choice was made to save disk space for the files generated during prediction.

If we want to skip the evaluation step in the prediction process, we can pass the flag `--skip_run_evaluation`.


## Reproducing the experiments with IOPS

The full parametric study (all lightweight convolutions across the datasets) is orchestrated with [IOPS](https://iops-benchmark.com), a benchmark orchestration framework. IOPS sweeps the parameters, runs the training and prediction steps, collects the metrics, and aggregates the results automatically.

Install IOPS with:

```shell
pip install iops-benchmark
```

The `requirements-experiment.txt` file contains the dependencies required to run the experiments with IOPS. If you have already installed the dependencies with `make setup-experiment`, you can skip installing the IOPS package, as it is included in the `requirements-experiment.txt` file.

The GPU usage probe relies on `nvidia-smi`, so an NVIDIA GPU with the NVIDIA drivers installed is required for the `gpu_sampling` metrics. Make sure `nvidia-smi` is available on the compute nodes.

Two IOPS configuration files are provided:

- `iops_train.yaml`: sweeps the lightweight convolutions and datasets to run the training step (`trainer.py`).
- `iops_predict.yaml`: runs the prediction step (`predictor.py`) over the weights produced by training.
- `iops_predict_evaluation.yaml`: runs the prediction step (`predictor.py`) over the weights produced by training and evaluates the results.

Before running, adjust `project_dir` in each file to point to the directory holding this code, and review the SLURM directives in `script_template` (node list, partition, modules) to match your cluster.

To launch the experiments:

```shell
iops run iops_train.yaml
iops run iops_predict.yaml
iops run iops_predict_evaluation.yaml
```

See the documentation at [iops-benchmark.com](https://iops-benchmark.com) for more details.

### Get IOPS metrics to reproduce the paper's results

When running IOPS experiments, two output files are generated in the experiment's root directory: `results.csv` and `__iops_resource_summary.csv`.
The `results.csv` file contains the predictive performance metrics of the model, including SSIM, MSE, and RMSE.
The `__iops_resource_summary.csv` file contains the GPU resource utilization metrics collected during the experiment execution.
For a full explanation of the IOPS resource tracing layer that generates this file, see the [IOPS Resource Sampling page](https://iops.gitlabpages.inria.fr/user-guide/resource-tracing/).

Therefore, the evaluated metrics can be mapped to the generated files as follows:

- results.csv (iops_train.yaml):
  - **metrics.total_training_time_sec**: Execution time in seconds
  - **metadata.sysinfo.gpu_memory_mib**: GPU total memory in MiB
- results.csv (iops_predict.yaml):
  - **metrics.total_predict_time_sec**: Execution time in seconds
  - **metadata.sysinfo.gpu_memory_mib**: GPU total memory in MiB
- results.csv (iops_predict_evaluation.yaml):
  - **metrics.MAE**
  - **metrics.RMSE**
  - **metrics.SSIM**
- __iops_resource_summary.csv:
  - **gpu0_energy_j**: GPU energy consumption in Joules
  - **gpu0_mem_peak_mib**: GPU peak memory usage in MiB
  - **gpu0_avg_mem_utilization_pct**: GPU memory utilization percentage

The columns with the prefix of `metrics.` are user-defined metrics parsed by IOPS.

To obtain the average GPU memory usage over the entire execution, we apply the following formula:

$$
avg\_mem\_uti = \frac{gpu0\_avg\_mem\_utilization\_pct}{100} * gpu\_memory\_mib
$$

To convert in kWh, we apply the formula:

$$
kWh = \frac{gpu0\_energy\_j}{3600000}
$$

## Deep Learning model reference

```
@article{he2025spatiotemporal,
  title={A spatiotemporal mixed-enhanced generative adversarial network for radar-based precipitation nowcasting},
  author={He, Long and Zheng, Kun and Ruan, Huihua and Yang, Shuo and Zhang, Jinbiao and Luo, Cong and Tang, Siyu and Yi, Yunlei and Tian, Yugang and Cheng, Jianmei},
  journal={Computers \& Geosciences},
  volume={200},
  pages={105919},
  year={2025},
  publisher={Elsevier}
}
```