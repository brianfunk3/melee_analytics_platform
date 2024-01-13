# this file gonna be unhinged
from melee_analytics_platform.stream_searcher.stream_predict import StreamPredict
from melee_analytics_platform.utils.utils import text_from_prediction
import numpy as np
from easyocr import Reader
import re
import time

class MeleeWatcher():
    def __init__(self,
                 project_name,
                 project_dir,
                 streamer,
                 consecutive_threshold = 3,
                 detection_threshold = .95):
        
        # assign some variables
        self.project_name = project_name
        self.project_dir = project_dir
        self.streamer = streamer
        self.consecutive_threshold = consecutive_threshold
        self.detection_threshold = detection_threshold

        # make a reader
        self.reader = Reader(lang_list=['en'], gpu=True)

        # create a nice list of classes we'd want to do some OCR on
        self.ocr_classes = ['Clock', 'Damage']
        self.meleeframes = []

        # make our streampredictor
        self.stream_predict = StreamPredict(self.project_name, self.project_dir, self.streamer, detection_threshold = self.detection_threshold)

        # make our cool game start and stop thing
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.game_start = False

        # some things for times
        self.run_times = []

    def get_frame_data(self):
        # get the results
        img, boxes, scores, classes = self.stream_predict.predict()

        # toss them into a melee frame
        if len(boxes) > 0:
            self.consecutive_hits +=1
            self.consecutive_misses = 0
            if self.consecutive_hits > self.consecutive_threshold:
                if not self.game_start:
                    self.game_start = True
                    print('Game Found, watching...')
                ocr_results = [text_from_prediction(img, boxes[i], self.reader, numbers_only=True) if classes[i] in self.ocr_classes else None for i in range(len(boxes))]
                mf = MeleeFrame(img, boxes, scores, classes, ocr_results)
                self.meleeframes.append(mf)
        else:
            self.consecutive_hits=0
            self.consecutive_misses+1
            if (self.consecutive_misses > self.consecutive_threshold) and self.game_start:
                self.game_start = False
                print('Game Ended')
        
    def watch(self):
        # this loop sucks, need to fix
        while len(self.meleeframes) < 100:
            st = time.time()
            self.get_frame_data()
            et = time.time()
            self.run_times.append(et-st)
        avg_seconds = round(sum(self.run_times)/len(self.run_times),3)
        print(f'Done: averaged {avg_seconds} seconds to process a frame')
        

class MeleeFrame():
    """
    lets make a nice lil class to represent a frame shall weeeeee
    """
    def __init__(self, img, boxes, scores, classes, ocr_results):
        # pop the variables in some goobers
        self.img = img
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.ocr_results = ocr_results

    def __repr__(self):
        return f'{self.classes} -> {self.ocr_results}'