# 2d-mimic

In this project, I have used detectron2 densepose model to draw a 2-d version of
the image and drew landmarks on top of it to make it more human.

## Setup
```
git clone https://github.com/vikey725/2d-mimic.git

conda create -n detectron2 python==3.7

pip install -U torch torchvision cython

pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 
'git+https://github.com/cocodatasetcocoapi.git'

git clone https://github.com/facebookresearch/detectron2.git

pip install -e detectron2

pip install av
pip install imutils

cd 2d-mimic
mkdir checkpoints
```

#Checkpoints
Save these files in checkpoints dir
1. Download landmark checkpoints from [here](dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Download model with name R_50_FPN_WC1_s1x from [chart-based model zoo](https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo)

## Run demo
```
# for 2d mimic
python -m scripts.run_demo -bg 1
# for default visualization 
python -m scripts.run_demo -v 1 -dv True -bg 1 
    1. -v (visualization type) can take values from 0 to 3
    
```




