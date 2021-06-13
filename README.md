# Also-Me

The repo contains source code for also-me project. Also-me creates virtual avatar like segmentation over the camera feed, so you can use in zoom, gmeet and elsewhere. 

## Setup
```
# Clone 2 repos.
git clone https://github.com/vikey725/2d-mimic.git
git clone https://github.com/facebookresearch/detectron2.git

# Create conda and install dependencies
conda create -n alsome python==3.7
pip install -r 2d-mimic/requirements.txt
pip install -e detectron2
pip install 'git+https://github.com/facebookresearch/fvcore.git'
sudo apt-get install python-opencv 
sudo apt-get install ffmpeg 

# Download model & checkpoint files
wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC1_s1x/173862049/model_final_289019.pkl -P 2d-mimic/checkpoints/model_final_289019.pkl
wget https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat?raw=true -P 2d-mimic/checkpoints/shape_predictor_68_face_landmarks.dat

# Install v4l2loopback & its utils for 2nd camera creation
git clone https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make
sudo make install
sudo apt-get install v4l2loopback-utils

# Create the secondary camera 
sudo modprobe v4l2loopback video_nr=7 card_label="Also-Me" exclusive_caps=1
```

## How to Use
```
# For using with Local GPU
1. cd 2d-mimic
2. python -m scripts.run_demo -bg 1

# For using with local CPU (would be very slow)
1. Change line 25 in /2d-mimic/configs/model_config.py to -> cfg.MODEL.DEVICE = 'cpu'
2. cd 2d-mimic
3. python -m scripts.run_demo -bg 1

# For using google colab as GPU server
1. Open the code in colab using [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/gok03/2d-mimic/blob/main/Also_Me_collab_server.ipynb)
2. Run all cells in colab and the command to run in local system will be ouput there.
3. cd 2d-mimic
4. use the copied command from colab (ex: python -m scripts.run_demo -bg 1 -rs 1 -rsip tcp://6.tcp.ngrok.io:18106)
```

### misc commands, use if required
```
# (Optional) Check the created camera
v4l2-ctl --list-devices 
You should find something like - 
Also-Me (platform:v4l2loopback-007):
    /dev/video7

# (Optional) To see the new camera output
ffplay /dev/video7

# (Optional) To delete the secondary camera
sudo modprobe -r v4l2loopback
```

## References
https://github.com/cedriclmenard/irislandmarks.pytorch
