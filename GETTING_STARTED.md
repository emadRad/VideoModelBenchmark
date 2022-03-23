# Getting Started

To run with a single gpu use the following command from the root of the project (where *run.py* located):
```bash
export DATA_DIR="$(pwd)/data/kinetics20"
docker run --gpus '"device=0"' --rm --user $(id -u):$(id -g) \
            --ipc=host -d --name run_vmb -v "$PWD":/workspace \
            -v "$DATA_DIR":"/data" -t video_benchmark \
            python run.py --cfg configs/Kinetics/X3D_M.yaml \
            DATA.PATH_TO_DATA_DIR /data \
            DATA_LOADER.NUM_WORKERS 8 \
            TRAIN.BATCH_SIZE 16 \
            NUM_GPUS 1
 ```

To run with multiple gpus change *--gpus '"device=[gpu_ids]"'* to add gpu ids and *NUM_GPUS*. For example running with 4 gpus

```bash
docker run --gpus '"device=0,1,2,3"' --rm --user $(id -u):$(id -g) \
            --ipc=host -d --name run_vmb -v "$PWD":/workspace \
            -v "$DATA_DIR":"/data" -t video_benchmark \
            python run.py --cfg configs/Kinetics/X3D_M.yaml \
            DATA.PATH_TO_DATA_DIR /data \
            DATA_LOADER.NUM_WORKERS 32 \
            TRAIN.BATCH_SIZE 64 \
            NUM_GPUS 4
            
```

You will find the results in the results directory.
