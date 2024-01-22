# this file gonna be unhinged
from melee_analytics_platform.utils.utils import text_from_prediction, get_midpoint
from melee_analytics_platform.pascal_voc_xml_modeling import Model
from easyocr import Reader
import numpy as np
import pandas as pd
import time

class MeleeCatcher():
    """
    So this data structure needs to recieve melee frame information and process/organize it
    There will be a core stucture that will have an "append" or "send" or something method
    where is recieves an image, boxes, scores, and classes - all numpy arrays. Once those are
    sent it needs to do some melee specific things like:
    - Do some OCR on clock and damages
    - Do some box math on the stocks to get the number of them
    - Assign damages and stocks to each other

    We'll probably have to mess around with the idea of a "match" within this where we ID
    the number of players, as well as general clock/stock/damage locations at a match level
    Assigning it and doing checks to the assigned should speed some things up?
    """
    def __init__(self,
                 project_dir,
                 streamer,
                 project_name ='melee_tracker',
                 detection_threshold = .9,
                 frame_limit = 100000000000000):
        
        # assign some things
        self.streamer = streamer
        self.detection_threshold = detection_threshold
        
        # make a model and assign some things to the class
        self.model = Model(project_name = project_name, project_dir=project_dir)

        # make a reader for doing some OCR things
        self.reader = Reader(lang_list=['en'], gpu=True)

        # create a nice list of classes we'd want to do some OCR on
        self.ocr_classes = ['Clock', 'Damage']
        self.meleeframes = []
        self.frame_limit = frame_limit
        self.times = []

    def process_frame(self):
        # get our image
        img = self.streamer.get_proper_frame()

        # escape if we did not get an image for some reason
        if img is None:
            return
        else:
            img = np.array(img)

        # bop it into a model
        boxes, scores, classes = self.model.predict(image = img, detection_threshold = self.detection_threshold)

        if len(boxes) == 0:
            return

        # its ocr o'clock
        ocr_results = [text_from_prediction(img, boxes[i], self.reader, numbers_only=True) if classes[i] in self.ocr_classes else None for i in range(len(boxes))]

        # get mid point from clock
        if 'Clock' in classes:
            clocks = classes[classes=='Clock'] # TODO this breaks if no clock
            if len(clocks) > 1:
                clock_len = 130
                clock_mid = 347
            elif len(clocks) == 1:
                clock_len = int(boxes[classes == 'Clock'][0][2] - boxes[classes == 'Clock'][0][0])
                clock_mid = get_midpoint(boxes[classes == 'Clock'][0])[0]
            else:
                clock_len = 130
                clock_mid = 347
        else:
            clock_len=130
            clock_mid=347

        # calculate the "expected" total length of the melee part of the frame
        melee_length = clock_len * (423/150)
        quad_length = melee_length/4

        # finally lets estimate 
        subtracter = clock_mid-(2*quad_length)

        # now find the "P#" of every class
        players = np.clip(np.floor((boxes[:,0] - subtracter)/quad_length) + 1, 1, 4)

        # clocks don't have a class so we need to set to null
        players[classes=='Clock'] = 0

        # find the "size" of each box for stocks
        stocks = np.round((boxes[:,2] - boxes[:,0])/(boxes[:,3]-boxes[:,1]),0).astype(int)

        # now we have to somehow combine this into a nice structure - honestly maybe pandas?
        df = pd.DataFrame({'frame_num': [len(self.meleeframes)+1] * len(boxes),
                           'player': players,
                           'class': classes,
                           'ocr_result': ocr_results,
                           'stocks': stocks,
                           'scores': scores})
        
        self.meleeframes.append(df)

    def shape_melee_df(self):
        # slap em together
        df = pd.concat(self.meleeframes)

        # conditionally make "value" column
        df['new_value']=(np.select([(df['class'].eq('Clock') | df['class'].eq('Damage')),
                                     df['class'].eq('Stock')],
                                    [df['ocr_result'], df['stocks']],
                                    np.nan))
        
        # fill some na values and do type conversion for players
        df['new_value'] = df.replace(r'^\s*$', np.nan, regex=True)['new_value'].fillna(0).astype(int)
        df['player'] = df['player'].astype(int).astype(str)

        # filter to what we want
        df = df[df['class'].isin(['Clock', 'Stock', 'Damage'])][['frame_num', 'player', 'class', 'new_value']]

        # pivot em out
        df = df.pivot_table(values='new_value', index='frame_num', columns=['player','class'], aggfunc='max')

        # flatten the multiindex of columns
        df.columns = ['_'.join(col) for col in df.columns.values]

        # remove accidental player columns
        bad_cols = []
        for col in df.columns:
            null_pct = df[col].isnull().sum() * 100 / len(df)
            if null_pct > 95:
                bad_cols.append(col)

        df=df.drop(columns=bad_cols)
        
        return df

    def watch(self):
        while (self.streamer.end_of_stream_counts < 15) and (len(self.meleeframes) < self.frame_limit):
            st = time.time()
            st_len = len(self.meleeframes)
            self.process_frame()
            et = time.time()
            et_len = len(self.meleeframes)
            if et_len > st_len:
                self.times.append(et-st)
        print(f"Average time to process a frame: {round(sum(self.times)/len(self.times), 3)} seconds")
        return self.shape_melee_df()