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

- **t_ref**: time to top1 accuracy 20.11 on our reference machines with **n_gpu_ref=4** gpus 
- **t_bench**: time to top1 accuracy greater than or equal to 20.11 on test machine with **n_gpu_bench** gpus

Score for the Transformer model with config **configs/MVIT_B_16x4.yaml**:

    transformer_score = t_ref / t_bench * (n_gpu_ref / n_gpu_bench)

- **t_ref**: time to top1 accuracy 20.59 on our reference machines with **n_gpu_ref=4** gpus 
- **t_bench**: time to top1 accuracy greater than or equal to 20.59 on test machine with **n_gpu_bench** gpus

Final score is the average of the two scores.

    benchmark_score = (cnn_score + transformer_score) / 2

## License
The majority of this work is licensed under [Apache 2.0 license](LICENSE). Portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).


## References
The code is adapted from the following repositories:

[https://github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast )

[https://github.com/kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
