# Video Model Benchmark



## Installation

Please find installation instructions in [INSTALL.md](INSTALL.md).

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md).

### Dataset
The dataset is 800MB and can be found in *data/kinetics20* folder. 
For further information regarding the dataset (such as license), please refer to [Kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset).

## Benchmark Score
The code should not be changed other than some settings regarding resources such as the num of cpus (DATA_LOADER.NUM_WORKERS) and number of gpus (NUM_GPUS).
Since we consider two types of models, we will use the following benchmark scores for CNN and Transformer based models.

Score for the CNN model with config **configs/X3D_M.yaml**:

    cnn_score = t_ref / t_bench * (n_gpu_ref / n_gpu_bench)

- **t_ref**: time to top1 accuracy 16.75 on our reference machines with **n_gpu_ref=4** gpus 
- **t_bench**: time to top1 accuracy  >= 16.75 on test machine with **n_gpu_bench** gpus

Score for the Transformer model with config **configs/MVIT_B_16x4.yaml**:

    transformer_score = t_ref / t_bench * (n_gpu_ref / n_gpu_bench)

- **t_ref**: time to top1 accuracy 34.70 on our reference machines with **n_gpu_ref=4** gpus 
- **t_bench**: time to top1 accuracy >= 34.70 on test machine with **n_gpu_bench** gpus

Final score is the average of the two scores.

    benchmark_score = (cnn_score + transformer_score) / 2

The **t_ref** time can be found in the [Results](#resultid) section.

<h2 id="resultid"> Results </h2> 

All the experiments were done on NVIDIA TITAN RTX gpus.

|val acc@1 | NUM_GPUs | BATCH_SIZE | EPOCHS | NUM_WORKERS | CONFIG_FILE | Elapsed Time (sec) |
|:---:|:---:| :---:| :---:|:---:| :---:| :---:|
| 16.75 | 4 | 64 | 20 | 32 | Kinetics/X3D_M.yaml | 3657.96 | 
| 34.70 | 4 | 48 | 20 | 32 | Kinetics/MVIT_B_16x4.yaml | 6682.53 |


Results with varying NUM_WORKERS using a single gpu.

| GPUs |val acc@1 | NUM_GPUs | BATCH_SIZE | EPOCHS | NUM_WORKERS | CONFIG_FILE | Elapsed Time (sec) |
|:---:|:---:|:---:| :---:| :---:|:---:| :---:| :---:|
| 1 X NVIDIA TITAN RTX | 22.70 | 1 | 16 | 20 | 8 | Kinetics/X3D_M.yaml | 7970.64 | 
| 1 X NVIDIA TITAN RTX | 22.65 | 1 | 16 | 20 | 16 | Kinetics/X3D_M.yaml | 7602.86 | 
| 1 X NVIDIA TITAN RTX | 22.85 | 1 | 16 | 20 | 32 | Kinetics/X3D_M.yaml | 7654.86 | 



## License
The majority of this work is licensed under [Apache 2.0 license](LICENSE). Portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).


## References
The code is adapted from the following repositories:

[https://github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast )

[https://github.com/kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
