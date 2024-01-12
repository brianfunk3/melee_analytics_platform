from melee_analytics_platform.pascal_voc_xml_modeling import Model
import numpy as np
import cv2

class StreamPredict():
    def __init__(self,
                 project_name,
                 project_dir,
                 streamer,
                 detection_threshold = .9):
        """
        Lord help me
        """
        self.model = Model(project_name = project_name, project_dir=project_dir)
        self.streamer = streamer
        self.detection_threshold = detection_threshold

    def predict(self):
        return self.model.predict(image = cv2.cvtColor(np.array(self.streamer.get_img()), cv2.COLOR_RGB2BGR), detection_threshold = self.detection_threshold)