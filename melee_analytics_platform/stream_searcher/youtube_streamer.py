from melee_analytics_platform.utils.selenium_helpers import buildHeadless, getBy
import time
import io
from PIL import Image

class YouTubeStreamer():
    """
    Given a youtube video id, get frames from the video in a nice way. Note this CANNOT be livestreams.
    Use FPS limit to speed things up and mimic a twitch stream where we won't get every frame
    """
    def __init__(self, vid_id):
        # build a driver
        self.driver = buildHeadless()
        # navigate to the page
        self.driver.get(f'https://www.youtube.com/watch?v={vid_id}')
        # press big button
        big_button = getBy(self.driver, 'class', 'ytp-large-play-button')
        big_button.click()
        # wait a bit in case things get scary
        time.sleep(3)
        # check if theres a dumb ad thing
        nightmare_thing = getBy(self.driver, 'xpath', '//*[@id="dismiss-button"]/yt-button-shape/button', delay=3)
        if nightmare_thing != False:
            nightmare_thing.click()
        # get out video element now
        self.element = getBy(self.driver, 'tag', 'video')

    
    def big_button_out_there(self):
        # determines if the video should be unpaused. Weird thing that can happen with ads
        pass

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