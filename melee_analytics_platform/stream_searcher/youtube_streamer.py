from melee_analytics_platform.utils.selenium_helpers import buildHeadless, getBy
from selenium.webdriver.common.by import By
import time
import io
from PIL import Image

class YouTubeStreamer():
    """
    Given a youtube video id, get frames from the video in a nice way. Note this CANNOT be livestreams.
    Use FPS limit to speed things up and mimic a twitch stream where we won't get every frame

    NEED
    - A way to know if a video is over

    """
    def __init__(self,
                 vid_id,
                 headless = True):
        # assign some boolean for stream status
        self.ads = False
        self.end_of_stream_counts = 0

        # build a driver
        self.driver = buildHeadless(headless=headless)
        # navigate to the page
        self.driver.get(f'https://www.youtube.com/watch?v={vid_id}')
        # press big button
        big_button = getBy(self.driver, 'class', 'ytp-large-play-button')
        big_button.click()
        # wait a bit in case things get scary
        time.sleep(3)
        self.check_for_nightmares()
        # get our video element now
        self.element = getBy(self.driver, 'tag', 'video')

    def check_for_nightmares(self, delay = 3):
        # check if theres a dumb ad thing
        nightmare_thing = getBy(self.driver, 'xpath', '//*[@id="dismiss-button"]/yt-button-shape/button', delay=delay)
        if nightmare_thing != False:
            nightmare_thing.click()

    
    def ad_check(self):
        """
        We need a nice checkerooski for if an ad is playing. But we have to be specific about
        if because we need this freak to be fast. We can't check if a specific ad element
        exists, because it'll spend time waiting for it to possibly load in. So instead we have
        check if the ad element is empty since that'll be lightning mcqueen
        """
        return len(getBy(self.driver, 'class', 'ytp-ad-module').find_elements(By.TAG_NAME, 'div')) > 0
    
    def end_check(self):
        return getBy(self.driver,'class', 'ytp-play-button').get_attribute("title") == 'Replay'

    def get_img(self):
            location = self.element.location
            size = self.element.size

            data = self.driver.get_screenshot_as_png()
            im = Image.open(io.BytesIO(data))
            
            x = location['x']
            y = location['y']
            w = size['width']
            h = size['height']
            width = x + w
            height = y + h

            im = im.crop((int(x), int(y), int(width), int(height)))
            return im
        
    def get_proper_frame(self):
        if self.end_check():
            self.end_of_stream_counts+=1      
        elif self.ad_check():
            self.end_of_stream_counts=0
            if not self.ads:
                print('in ads')
            self.ads = True
        else:
            self.end_of_stream_counts=0
            if self.ads:
                print('ads over')
                self.check_for_nightmares(delay=1)
                self.ads = False
            return self.get_img()