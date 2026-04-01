# KD-UVAD
The paper link and code will be made publicly available upon acceptance.

## Results

| Model      | Dataset   | Source Paper | kduvad.py |
|------------|-----------|--------------|-----------|
| KD-UVAD    | UCSD Ped2 | 97.1         | 97.1      |
| KD-UVAD    | Avenue    | 89.3         | 89.3      |
| KD-UVAD    | STC       | 78.0         | 78.0      |
| KD-UVAD-ST | UCSD Ped2 | -            | 91.4      |
| KD-UVAD-ST | Avenue    | -            | 84.8      |
| KD-UVAD-ST | STC       | -            | -         |
| KD-UVAD-DS | UCSD Ped2 | -            | 84.2      |
| KD-UVAD-DS | Avenue    | -            | 77.8      |
| KD-UVAD-DS | STC       | -            | -         |


## Setup

### Packages

```
Python==3.10
torch==1.13.0+cu117
torchvision==0.14.0+cu117
torchaudio==0.13.0+cu117
tifffile==2025.5.10
tqdm==4.67.1
scikit-learn==1.2.2
einops==0.8.1
fvcore==0.1.5.post20221221
numpy==1.26.4
opencv-python==4.11.0.86
pillow==11.3.0
```


Download dataset: (Ped2 dataset is derived from http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm.).

## ped2_processing.py

Building training data from UCSD Ped2 dataset.

```
from util.ped2_motion_config import update_config
from util.ped2_motion_dataset import dataset_path_for_memory, np_img_load_frame, Dataset_Path_For_Memory_Train_Infinite, Dataset_Path_For_Memory_Test_Infinite, InfiniteDataloader
```

## kduvad_ped2.py

Training and inference:

```
python kduvad_ped2.py --cfg /home/user/codes/KD-VAD/exp/k400/k400+k710_b16_f8x224/config_mlkd.yaml --num_shards 1 NUM_GPUS 1 NUM_SHARDS 1 --dataset UCSDped2 --output_dir output/kduvad_on_ped2 --consecutive 8 --prev_clip_num 1 --train_steps 100000
```

## Avenue Dataset

Download dataset: (Avenue dataset is derived from https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html.).

## avenue_processing.py

Building training data from Avenue dataset.

```
from util.avenue_motion_config import update_config
from util.avenue_motion_dataset import dataset_path_for_memory, np_img_load_frame, Dataset_Path_For_Memory_Train_Infinite, Dataset_Path_For_Memory_Test_Infinite, InfiniteDataloader
```

## kduvad_avenue.py

Training and inference:

```
python kduvad_avenue.py --cfg /home/user/codes/KD-VAD/exp/k400/k400+k710_b16_f8x224/config_mlkd.yaml --num_shards 1 NUM_GPUS 1 NUM_SHARDS 1 --dataset avenue --output_dir output/kduvad_on_avenue --consecutive 8 --prev_clip_num 1 --train_steps 100000
```

## STC Dataset

Download dataset: (STC dataset is derived from https://svip-lab.github.io/dataset/campus_dataset.html.).

## stc_processing.py

Building training data from Shanghai Campus Tech dataset.

```
from util.stc_motion_config import update_config
from util.stc_motion_dataset import dataset_path_for_memory, np_img_load_frame, Dataset_Path_For_Memory_Train_Infinite, Dataset_Path_For_Memory_Test_Infinite, InfiniteDataloader
```

## kduvad_stc.py

Training and inference:

```
python kduvad_stc.py --cfg /home/user/codes/KD-VAD/exp/k400/k400+k710_b16_f8x224/config_mlkd.yaml --num_shards 1 NUM_GPUS 1 NUM_SHARDS 1 --dataset shanghai --output_dir output/kduvad_on_stc --consecutive 8 --prev_clip_num 1 --train_steps 1000000
```

## Reproducibility Details

This section includes:
1. Model configuration
2. Hyperparameter settings
3. Training strategy

```
Detailed information will be made available upon paper acceptance.
```
