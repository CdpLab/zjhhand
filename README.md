## Install
Requires: Linux cuda11.8 python3.9 CUDA11.3

## packages
- pytorch2.0.0
- pytorch3d
- numpy
- OpenCV
- yacs
- cmake
- imageio
- matplotlib

## MANO
download MANO data in [here](https://mano.is.tue.mpg.de/) and put MANO_LEFT.PKL and MANO_RIGHT.PKL in misc/mano

## Dataset processing
download InterHand2.6M dataset in [here](https://mks0601.github.io/InterHand2.6M/) and unzip it.(We use the v1.0 5FPS version and H+M subset)
run the code below and you can get the processed dataset in your path
```bash
python dataset/interhand.py --data_path         --save_path
```
## Train
run the code below and you can add --gpu at the end to specify the gpu you want to use. And you can get model weights file in the "output/model/exp"
```bash
python experiment/train.py
```
## Evaluation
run the code below.You need to revise the weight file path which gets in the training in the evaluation.py
```bash
python experiment/evaluation.py
```
## Results
The results of InterHand2.6M dataset.
<img src="https://github.com/zjhnightnight/hand/blob/main/1.png" width="600" height="600" /><br/>
The results of RGB2Hand and EgoHand dataset.
<img src="https://github.com/zjhnightnight/hand/blob/main/2.png" width="600" height="350" /><br/>
