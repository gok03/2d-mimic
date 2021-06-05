import time
import cv2
import argparse
import numpy as np

from configs.color_config import ColorConfig
from configs.model_config import detector, landmark_predictor
from code.predictor import Predictor
from imutils.video import WebcamVideoStream

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def run_webcam(args):
    vis_type = args.get('vis_type', None)
    background_img = cv2.imread(f"backgrounds/bg{args['background_image']}.jpg")
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    dp_predictor = Predictor(visualizer_type=vis_type, default_visualization=args['default_visualization'])
    stream = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    while True:
        frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bbox, UV, image_vis, labels = dp_predictor.predict(frame)
        # print(np.array(image_vis).shape)
        # input()
        out = cv2.resize(background_img, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)
        if labels is None:
            print(bbox)
            out = out.astype(np.uint8)
            combined_image = np.concatenate((frame, out), axis=1)
            if args["default_visualization"]:
                combined_image = np.concatenate((frame, frame), axis=1)
            cv2.imshow('frame', combined_image)
        else:

            labels = np.array(labels)
            rects = detector(gray)
            shape = []
            for (i, rect) in enumerate(rects):
                shape = landmark_predictor(gray, rect)
                shape = shape_to_np(shape)

            x, y, w, h = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])


            for idx in range(24):
                mask = np.zeros((frame.shape[0], frame.shape[1])) != 0
                mask[y:y + h, x:x + w] += (labels == idx + 1)
                out[:, :, :][mask] = ColorConfig.COLORS[idx]
            out = out.astype(np.uint8)

            try:
                cv2.fillConvexPoly(out, np.array(shape[36:42]), (0, 0, 0))
                cv2.fillConvexPoly(out, np.array(shape[42:48]), (0, 0, 0))
                cv2.fillConvexPoly(out, np.array(shape[48:60]), (0, 8, 100))
                cv2.fillConvexPoly(out, np.array(shape[60:68]), (0, 0, 130))
                cv2.fillPoly(out, [np.array(shape[17:22])], (0, 0, 0))
                cv2.fillPoly(out, [np.array(shape[22:27])], (0, 0, 0))
                cv2.drawContours(out, [np.array(shape[27:31])], 0, (0, 0, 0), 2)
                cv2.drawContours(out, [np.array(shape[31:33])], 0, (0, 0, 0), 2)
                cv2.drawContours(out, [np.array(shape[34:36])], 0, (0, 0, 0), 2)
                for i in range(1, 17):
                    cv2.line(out, shape[i - 1], shape[i], 2)
            except Exception as e:
                print(e)

            combined_image = np.concatenate((frame, out), axis=1)
            if args["default_visualization"]:
                combined_image = np.concatenate((frame, image_vis), axis=1)
            # combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
            # combined_image = crop_img

            cv2.imshow('frame', combined_image)

        # fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vis-type", type=int, default=0,
                    help="Visualizer type (0, 1, 2, 3")
    ap.add_argument("-dv", "--default-visualization", type=bool, default=False,
                    help="Inbuilt visualization")

    # You can save more images in the backgrounds folder,
    # follow naming convention bg{img_no}.jpg, eg: bg1.jpg
    ap.add_argument("-bg", "--background-image", type=int, default=1,
                    help="Background image no.")
    args = vars(ap.parse_args())
    run_webcam(args)