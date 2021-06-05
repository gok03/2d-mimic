import sys
from configs.model_config import cfg, visualizers
from detectron2.engine.defaults import DefaultPredictor

sys.path.append("../../detectron2/projects/DensePose")
from densepose.vis.extractor import create_extractor

class Predictor:
    def __init__(self, visualizer_type: int = 1, default_visualization: bool = False):
        """
            Class to predict densepose output

            Args:
                visualizer_type: can take values from 1 to 4
                    (please refer visualizers in model_config)
                default_visualization: whether you want default
                    visualization or 2-D visualization
        """
        self.default_visualization = default_visualization
        self.vis = visualizers[visualizer_type]
        self.dp_predictor = DefaultPredictor(cfg)
        self.extractor = create_extractor(self.vis)

    def predict(self, frame):
        """
        function to predict densepose
        Args:
             frame: input image for prediction
        """
        img = frame
        bbox, UV, image_vis, labels = None, None, None, None
        predictions = self.dp_predictor(img)
        instances = predictions['instances']

        result = self.extractor(instances)
        for idx, (densepose_chart_result) in enumerate(result):
            if idx != 0:
                break
            if densepose_chart_result is None:
                return bbox, UV, image_vis, labels
            UV = densepose_chart_result[0].uv.cpu()
            labels = densepose_chart_result[0].labels.cpu()
        bbox = result[1]
        if self.default_visualization:
            image_vis = self.vis.visualize(img, result)

        return (bbox, UV, image_vis, labels)



