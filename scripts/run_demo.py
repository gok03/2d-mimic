import time
import cv2
import argparse
import numpy as np
import pyfakewebcam
import multiprocessing as mp
import zmq
import random
import sys
import cv2
from simplejpeg import decode_jpeg, encode_jpeg
import msgpack

from configs.color_config import ColorConfig
from configs.model_config import detector, landmark_predictor
from code.predictor import Predictor
from imutils.video import WebcamVideoStream

from scripts.utils import draw_line

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def pack_message(msg):
    return msgpack.packb(msg)

def unpack_message(msg):
    return msgpack.unpackb(msg)

def predict(frame, socket, cv2):
    frame = cv2.resize(frame, (640, 480))  # resize the frame
    socket.send(pack_message(encode_jpeg(frame, quality = 100, colorspace = "RGB", fastdct=True)))

    msg = socket.recv()

    source = decode_jpeg(unpack_message(msg), colorspace = "RGB", fastdct=True, fastupsample = True)
    # source = cv2.resize(source, (480, 360))  # resize the frame
    cv2.imshow('image',source)
    cv2.waitKey(1)
    #time.sleep(1)
    camera.schedule_frame(source)


def run_webcam(args):
    if (args.get('remote_server', 0) == 0):
        camera = pyfakewebcam.FakeWebcam('/dev/video9', 640, 480) 
        vis_type = args.get('vis_type', None)
        out_type = args.get('out_type', None)
        background_img = cv2.imread(f"backgrounds/bg{args['background_image']}.jpg")
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        dp_predictor = Predictor(visualizer_type=vis_type, output_type=out_type)
        stream = WebcamVideoStream(src=0).start()
        time.sleep(1.0)
        camera_started_flag = True
        try:
            while True:
                frame = stream.read()
                background_img = cv2.resize(background_img, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)

                out = dp_predictor.dp_predict(frame, background_img)

                combined_image = np.concatenate((frame, out), axis=1)
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

                cv2.imshow('image',out)
                cv2.waitKey(1)
                camera.schedule_frame(out)

                if camera_started_flag:
                    print("Camera started ...")
                   camera_started_flag = False
        except KeyboardInterrupt:
            context.destroy()
                
    else:
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect(args.get('remote_server_ip'))
        stream = WebcamVideoStream(src=0).start()
        camera = pyfakewebcam.FakeWebcam('/dev/video9', 640, 480)
        camera_started_flag = True 
        try:
            while True:
                    frame = stream.read()
                    frame = cv2.resize(frame, (640, 480))  # resize the frame
                    socket.send(pack_message(encode_jpeg(frame, quality = 100, colorspace = "RGB", fastdct=True)))

                    msg = socket.recv()

                    source = decode_jpeg(unpack_message(msg), colorspace = "RGB", fastdct=True, fastupsample = True)

                    cv2.imshow('image',source)
                    cv2.waitKey(1)
                    camera.schedule_frame(source)

                    if camera_started_flag:
                        print("Camera started ...")
                        camera_started_flag = False

        except KeyboardInterrupt:
            context.destroy()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vis-type", type=int, default=0,
                    help="Visualizer type (0, 1, 2, 3")

    ap.add_argument("-o", "--out-type", type=int, default=0,
                    help="Visualizer type (0, 1")

    # You can save more images in the backgrounds folder,
    # follow naming convention bg{img_no}.jpg, eg: bg1.jpg
    ap.add_argument("-bg", "--background-image", type=int, default=1,
                    help="Background image no.")

    ap.add_argument("-rs","--remote_server", type=int, default=0,
                    help="remote type (0, 1")

    ap.add_argument("-rsip","--remote_server_ip", help="tcp ip address port, ex: tcp://0.tcp.ngrok.io:13261")

    args = vars(ap.parse_args())
    run_webcam(args)
