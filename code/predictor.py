import sys

import cv2

from configs.model_config import cfg, visualizers
from configs.shade_config import ShadeConfig
from detectron2.engine.defaults import DefaultPredictor
import numpy as np
import torch
import mediapipe as mp

from configs.color_config import ColorConfig
from configs.landmark_config import LandmarkConfig
from scripts.utils import draw_line

sys.path.append("../../detectron2/projects/DensePose")
from densepose.vis.extractor import create_extractor

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
facemesh = mp_face_mesh.FaceMesh(max_num_faces=1)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

from code.iris_landmarks import IrisLandmarks
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = IrisLandmarks().to(gpu)
# net.load_weights("checkpoints/irislandmarks.pth")



class Predictor:
    def __init__(self, visualizer_type: int = 0, output_type: int = 0):
        """
            Class to predict densepose output

            Args:
                visualizer_type: can take values from 0 to 3
                    (please refer visualizers in model_config)
                output_type: 0 for IUV and 1 for colored mask
        """
        self.vis = visualizers[visualizer_type]
        self.out_type = output_type
        self.dp_predictor = DefaultPredictor(cfg)
        self.extractor = create_extractor(self.vis)
        self.iris_predictor = IrisLandmarks().to(gpu)
        self.iris_predictor.load_weights("checkpoints/irislandmarks.pth")


    def is_black_pixel(self, color_rgb):
        black_lower_range = [80, 50, 50]

        if ((color_rgb[0] <= black_lower_range[0] and color_rgb[1] <= black_lower_range[1] and color_rgb[2] <=
                     black_lower_range[2])):
            return True
        return False

    def lm_predict(self, img, out):
        mp_result = facemesh.process(img)
        if not mp_result.multi_face_landmarks:
            return out
        faces = []
        if mp_result.multi_face_landmarks:
            for face_landmarks in mp_result.multi_face_landmarks:
                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                cv2.fillPoly(out, [np.array([face[idx] for idx in LandmarkConfig.outer_layer1])], (94, 102, 161))

                # for eye patches
                x1, y1 = int((face[159][0] + face[145][0])/2), int((face[159][1] + face[145][1])/2)
                left_eye_patch = img[y1-32:y1+32, x1-32: x1+32]
                left_eye_patch = cv2.resize(left_eye_patch, (64, 64), interpolation=cv2.INTER_AREA)
                eye_gpu, iris_gpu = self.iris_predictor.predict_on_image(left_eye_patch)
                iris = iris_gpu.cpu().numpy()
                xs1, ys1 = iris[:, :, 0][0], iris[:, :, 1][0]
                xs1 = [int(x1 - 32 + xs1[i]) for i in range(5)]
                ys1 = [int(y1 - 32 + ys1[i]) for i in range(5)]

                x2, y2 = int((face[386][0] + face[374][0]) / 2), int((face[386][1] + face[374][1]) / 2)
                right_eye_patch = img[y2 - 32:y2 + 32, x2 - 32: x2 + 32]
                right_eye_patch = cv2.resize(left_eye_patch, (64, 64), interpolation=cv2.INTER_AREA)
                eye_gpu, iris_gpu = self.iris_predictor.predict_on_image(right_eye_patch)
                iris = iris_gpu.cpu().numpy()
                xs2, ys2 = iris[:, :, 0][0], iris[:, :, 1][0]
                xs2 = [int(x2 - 32 + xs2[i]) for i in range(5)]
                ys2 = [int(y2 - 32 + ys2[i]) for i in range(5)]


                    # print(id, x, y)
                for id, lm in enumerate(face):
                    if id in LandmarkConfig.left_eybrow or id in LandmarkConfig.right_eybrow\
                            or id in LandmarkConfig.outer_left_eye or id in LandmarkConfig.outer_right_eye\
                            or id in LandmarkConfig.inner_left_eye or id in LandmarkConfig.inner_right_eye\
                            or id in LandmarkConfig.lip_layer1 or id in LandmarkConfig.lip_layer2\
                            or id in LandmarkConfig.lip_layer3 or id in LandmarkConfig.lip_layer4 \
                            or id in LandmarkConfig.nose:
                            # or id in LandmarkConfig.nose or id in LandmarkConfig.outer_layer1:
                        cv2.circle(out, (lm[0], lm[1]), 1, (0, 0, 0), -1)


                # cv2.fillPoly(out, [np.array([face[idx] for idx in ShadeConfig.left_side_pts])], (94, 102, 161))
                # cv2.fillPoly(out, [np.array([face[idx] for idx in ShadeConfig.right_side_pts])], (94, 102, 161))
                cv2.fillPoly(out, [np.array([face[idx] for idx in LandmarkConfig.right_eybrow])], (0, 0, 0))
                cv2.fillPoly(out, [np.array([face[idx] for idx in LandmarkConfig.left_eybrow])], (0, 0, 0))
                cv2.fillPoly(out, [np.array([face[idx] for idx in LandmarkConfig.outer_right_eye])], (133, 144, 205))
                cv2.fillPoly(out, [np.array([face[idx] for idx in LandmarkConfig.outer_left_eye])], (133, 144, 205))
                cv2.fillPoly(out, [np.array([face[idx] for idx in LandmarkConfig.inner_right_eye])], (255, 255, 255))
                cv2.fillPoly(out, [np.array([face[idx] for idx in LandmarkConfig.inner_left_eye])], (255, 255, 255))

                # for i in range(5):
                cv2.circle(out, (xs1[0], y1), 5, (0, 0, 0), -1)
                cv2.circle(out, (xs2[0], y2), 5, (0, 0, 0), -1)
        return out

    def dp_predict(self, img, background_img):
        """
            function to predict densepose
            Args:
                 frame: input image for prediction
        """
        bg_img = background_img.copy()
        predictions = self.dp_predictor(img)

        instances = predictions['instances']
        result = self.extractor(instances)
        out = bg_img
        if result[1] is None:
            return bg_img
        bboxes = np.array(result[1].cpu())
        for idx, densepose_chart_result in enumerate(result):
            if idx != 0:
                break
            if densepose_chart_result is None:
                return bg_img
            bbox = bboxes[0]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if self.out_type == 0:
                IUV = torch.cat(
                    (densepose_chart_result[0].labels[None].type(torch.float32), densepose_chart_result[0].uv * 255.0)
                ).type(torch.uint8).cpu()
                IUV = np.array(IUV)
                IUV = np.moveaxis(IUV, 0, 2)
                mask = np.sum(IUV, axis=2) != 0
                out[y:y + h, x:x + w][mask] = 0
                out[y:y + h, x:x + w] += IUV
            else:
                labels = densepose_chart_result[0].labels.cpu()
                labels = np.array(labels)
                for idx in range(24):
                    mask = np.zeros((img.shape[0], img.shape[1])) != 0
                    mask[y:y + h, x:x + w] += (labels == idx + 1)
                    out[:, :, :][mask] = ColorConfig.COLORS[idx]

        out = self.lm_predict(img, out)
        return out

if __name__ == '__main__':
    img = cv2.imread("yg.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred = Predictor()
    res = pred.lm_predict(img, img)
    cv2.imwrite("out_all.jpg", res)





