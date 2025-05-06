Requirements：
packages:
python3.9
pytorch2.0.0
pytorch3d0.7.4
cuda11.3
tqdm
opencv
numpy
yacs
matplotlib
imageio
cmake
chumpy

mano:
download mano data and put mano_left.pkl and mano.right.pkl in misc/mano

dataset processing
(1)download InterHand2.6M dataset and unzip it.(We use the v1.0_5fps version and H+M subset for training and evaluting)
(2)run the code"python dataset/interhand.py --data_path         --save_path "

training 
run the code "python experiment/train.py ",You can add --gpu at the end to specify the gpu you want to use.And you can get the Model weights file in "output/model/exp"

evaluation
run the code "python experiment/evaluation.py".You need to revise the weight file path which gets in the training in the evaluation.py