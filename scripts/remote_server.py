import time
import cv2
import argparse
import numpy as np
import pyfakewebcam
import zmq
import random
import sys
from google.colab.patches import cv2_imshow
from simplejpeg import decode_jpeg, encode_jpeg
import msgpack
import base64
from pyngrok import ngrok
from configs.color_config import ColorConfig
from configs.model_config import detector, landmark_predictor
from code.predictor import Predictor
from imutils.video import WebcamVideoStream

from scripts.utils import draw_line

def pack_message(msg):
    return msgpack.packb(msg)

def unpack_message(msg):
    return msgpack.unpackb(msg)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def run_webcam(args): 
    port = '5555'
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)
    vis_type = args.get('vis_type', None)
    out_type = args.get('out_type', None)
    background_img = cv2.imread(f"backgrounds/bg1.jpg")
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    dp_predictor = Predictor(visualizer_type=vis_type, output_type=out_type)
    connection_string = ngrok.connect(5555, "tcp").public_url
    print("---------------------------")
    print("python -m scripts.run_demo -bg 1 -rs 1 -rsip "+connection_string)
    print("---------------------------")
    while True:
        frame = socket.recv()
        frame = decode_jpeg(unpack_message(frame), colorspace = "RGB", fastdct=True)
        background_img = cv2.resize(background_img, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)
        out = dp_predictor.dp_predict(frame, background_img)
        # combined_image = np.concatenate((frame, out), axis=1)
        # cv2.imshow('frame', combined_image)

        # 2nd Camera ouput
        out=cv2.resize(out, (640, 480), interpolation=cv2.INTER_AREA)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        socket.send(pack_message(encode_jpeg(out, colorspace = "RGB", fastdct=True)))

if __name__ == '__main__':
    ngrok.set_auth_token("1t4SKLGLQrQWbMnhgqeLMDoZWO5_81uYnSFsT87KsBm4A6zGZ")
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vis-type", type=int, default=0,
                    help="Visualizer type (0, 1, 2, 3")

    ap.add_argument("-o", "--out-type", type=int, default=0,
                    help="Visualizer type (0, 1")
    args = vars(ap.parse_args())
    run_webcam(args)
