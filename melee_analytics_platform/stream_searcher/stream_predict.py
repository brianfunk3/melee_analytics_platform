
from melee_analytics_platform.melee_stuff import MeleeCatcher
import numpy as np
import cv2
import time

class StreamPredict():
    def __init__(self,
                 ):
        """
        Need to make it so this class fully controls what is sent to melee frame holder
        The holder will be a class that recieves predictions and processes them. The watch
        method needs to be here, the holder needs the model and such. The holder will probably
        need a base class
        """

        

        # holder for run times to predict
        self.run_times = []

        # TODO assign our melee specific things
        self.catcher = catcher

    def predict(self):
        img = self.streamer.get_img()
        if img is None:
            # ads
            if not self.ads:
                print('entering ad mode')
                self.ads = True
            return None, None, None, None
        elif img == False:
            print('end of stream')
            self.end_of_stream = True
            return None, None, None, None
        else:
            if self.ads:
                print('ads over')
                self.ads = False
            img = np.array(img)
            boxes, scores, classes = self.model.predict(image = img, detection_threshold = self.detection_threshold)
            return img, boxes, scores, classes