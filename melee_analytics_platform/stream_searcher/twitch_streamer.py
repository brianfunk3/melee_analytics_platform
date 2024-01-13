from melee_analytics_platform.utils.selenium_helpers import buildHeadless, getBy
import time
from PIL import Image
import io
import numpy as np

class TwitchStreamer():
    """
    Given a twitch streamer, navigate to their channel and start watching the stream
    """
    def __init__(self, username):
        # build a driver
        self.driver = buildHeadless()
        # navigate to the page
        self.driver.get(f'https://www.twitch.tv/{username}')
        # find our element to keep an eye on
        self.element = getBy(self.driver, 'tag', 'video')
        # wait a bit for the mute thing to unhide
        time.sleep(4)

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