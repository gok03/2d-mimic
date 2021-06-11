import time
import cv2
import argparse
import numpy as np

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


def run_webcam(args):
    vis_type = args.get('vis_type', None)
    out_type = args.get('out_type', None)
    background_img = cv2.imread(f"backgrounds/bg{args['background_image']}.jpg")

    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    dp_predictor = Predictor(visualizer_type=vis_type, output_type=out_type)
    stream = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    cnt = 0
    while True:
        frame = stream.read()
        background_img = cv2.resize(background_img, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)
        out = dp_predictor.dp_predict(frame, background_img)
        # combined_image = np.concatenate((frame, out), axis=1)
        cv2.imshow('frame', out)

        # fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



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
    args = vars(ap.parse_args())
    run_webcam(args)