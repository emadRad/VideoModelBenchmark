# Getting Started

To run with a single gpu use the following command from the root of the project (where *run.py* located):
```bash
export DATA_DIR=[path to data directory]
docker run --gpus '"device=0"' --rm --user $(id -u):$(id -g) \
            --ipc=host -d --name run_vmb -v "$PWD":/workspace \
            -v "$DATA_DIR":"$DATA_DIR" -t video_benchmark \
            python run.py --cfg configs/Kinetics/X3D_M.yaml \
            DATA.PATH_TO_DATA_DIR $DATA_DIR \
            DATA_LOADER.NUM_WORKERS 8 \
            TRAIN.BATCH_SIZE 16 \
            NUM_GPUS 1
 ```

To run with multiple gpus change *--gpus '"device=[gpu_ids]"'* to add gpu ids and *NUM_GPUS*. For example running with 4 gpus

```bash
docker run --gpus '"device=0,1,2,3"' --rm --user $(id -u):$(id -g) \
            --ipc=host -d --name run_vmb -v "$PWD":/workspace \
            -v "$DATA_DIR":"$DATA_DIR" -t video_benchmark \
            python run.py --cfg configs/Kinetics/X3D_M.yaml \
            DATA.PATH_TO_DATA_DIR $DATA_DIR \
            DATA_LOADER.NUM_WORKERS 32 \
            TRAIN.BATCH_SIZE 64 \
            NUM_GPUS 4
            
```

You will find the results in the results directory.
            
## Results

| Machine | GPUs |val acc@1 | NUM_GPUs | BATCH_SIZE | Epochs | NUM_WORKERS | Experiment | Elapsed Time |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:| :---:| :---:|
| CVG-SRV05 | 4 X NVIDIA TITAN RTX | 20.11 | 4 | 16 | 20 | 8 | Kinetics/X3D_M.yaml | 7456.88 | 
| CVG-SRV05 | 4 X NVIDIA TITAN RTX | 20.59 | 4 | 16 | 20 | 8 | Kinetics/MVIT_B_16x4.yaml | 7725.12 |
