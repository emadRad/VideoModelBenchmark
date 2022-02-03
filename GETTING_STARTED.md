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
            TRAIN.BATCH_SIZE 16  
 ```
