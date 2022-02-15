# Video Model Benchmark



## Installation

Please find installation instructions in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](sgs/datasets/DATASET.md) to prepare the datasets.

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md).

## Benchmark Score
The code should not be changed other than some settings regarding resources such as the num of cpus (DATA_LOADER.NUM_WORKERS) and number of gpus (NUM_GPUS).

score = t_ref / t_bench * (n_gpu_ref / n_gpu_bench)

- **t_ref**: time to top1 accuracy 5.16 on our reference machine with **n_gpu_ref** gpus
- **t_bench**: time to top1 accuracy 5.16 on test machine with **n_gpu_bench** gpus



## License
The majority of this work is licensed under [Apache 2.0 license](LICENSE). Portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).


## References
The code is adapted from the following repositories:

[https://github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast )

[https://github.com/kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
