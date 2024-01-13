import yt_dlp
import numpy as np
import cv2
from random import randint, sample
import os
import pandas as pd
import datetime

class VideoStream:
    """
    This class is really just a nice little holder for our youtube video information. Was
    mostly take from online, though I lost the source of the code - sorry! Luckily it's
    pretty straightforward to maintain. It takes the video format dictionary as an input
    and sorts things into a nice set of attributes

    Attributes:
    url: str
        the string url of the video
    resolution: str
        string format of the video resolution
    height: int
        height of video in pixels
    width: int
        width of video in pixels
    """
    def __init__(self, video_format):
        self.url = video_format['url']
        self.resolution = video_format['format_note']
        self.height = video_format['height']
        self.width = video_format['width']

    def __str__(self):
        return f'{self.resolution} ({self.height}x{self.width}): {self.url}'

def list_video_streams(url):
    """
    Again mostly sourced from the internet (same source as VideoStream class). Given a
    YouTube url, this function will return the streams, resolutions, and duration of the
    video. Worth noting that I had to add a try catch in the event a video was delisted
    or set to private. Also added the duration as a return value so that we can easily
    limit the random frame pulling down the line.

    Parameter:
    url: str
        YouTube url of the video we'd like to get the stream, resolution, and duration of

    Returns: [streams], [resolutions], duration
    """
    # this library is LOUD by default. Quiet it here
    ydl_opts = {"quiet": True}
    # try to get the video info, return nothing if we hit a wall
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            meta_data = ydl.extract_info(url, download=False)
        except:
            return None, None, None, None
        # get the total number of frames in the video
        duration = meta_data['duration'] 
        fps = meta_data['fps']
        # organize the valid streams
        streams = [VideoStream(format) for format in meta_data['formats'][::-1] if format['vcodec'] != 'none' and 'format_note' in format]
        # and do the same for the resolutions in a kind of janky way
        bad, unique_indices = np.unique(np.array([stream.resolution for stream in streams]), return_index=True)
        # now use the indices to finalize
        streams = [streams[index] for index in np.sort(unique_indices)]
        # and once more for resolutions
        resolutions = np.array([stream.resolution for stream in streams])
        # send em all
        return streams[::-1], resolutions[::-1], duration, fps

def cap_from_youtube(url, resolution=None):
    """
    Returns an open CV capture given a url and a duration

    Parameters:
    url: str
        The YouTube url to get the capture from
    resolution: str
        The desired resolution of the output. Default None to get the best available

    Returns: cv2.VideoCapture, duration: int
    """
    # grab all the streams, resolutions, and the duration
    streams, resolutions, duration, fps = list_video_streams(url)
    # if the video is unavailable, get us out of here
    if streams is None:
        return None, None, None
    # this is usually our exit, get the best option since it's sorted
    if not resolution or resolution == 'best':
        return cv2.VideoCapture(streams[-1].url), duration, fps
    # raise error if some is too picky
    if resolution not in resolutions:
        raise ValueError(f'Resolution {resolution} not available')
    # find where the resolution we want is
    res_index = np.where(resolutions == resolution)[0][0]
    # now return the a capture for that stream, as well as the duration
    return cv2.VideoCapture(streams[res_index].url), duration, fps

def gen_random_nums(max_num, num_nums):
    """
    Generate a list of random numbers in the range of 0-max_num
    """
    return [randint(1, max_num-1) for num in range(num_nums)]

def pull_rando_images(url, directory, num_images = 20):
    """
    Given a YouTube url and a directory, save num_images random images from the video
    in the directory as .jpg files.

    Parameters:
    url: str
        The YouTube url to download the random images from
    directory: str
        The directory to save the image to
    num_images: int
        The number of random images to grab and save

    Returns: None
    """
    # make a path just in case
    if directory.endswith('/') and not os.path.exists(f"{directory}"):
        os.makedirs(f"{directory}")
    # get the capture and the duration
    cap, duration, fps = cap_from_youtube(url)
    # catch for bad times
    if cap is None:
        return
    # get the indices of the random images to pull
    image_indices = gen_random_nums(duration*fps, num_images)    
    # now iterate through those indices and save them off
    for index in image_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        if cap.isOpened():
            ret, frame = cap.read()
            if frame is not None:
                cv2.imwrite(f'{directory}frame_{index}.jpg', frame)

def get_video_ids(min_date = None, character = None, guarantee_character = False):
    """
    Returns a filtered list of YouTube video ids sourced from our 'smash_vods.csv' that
    have a date greater than min_date and have character (if provided). Use guarantee_character
    for force ONLY that character - meaning both players are using character. 

    Parameters:
    min_date: str
        The minimum date of the competition to return, in 'YYYYMMDD' format
    character: str
        A melee character name to filter to video ids to
    guarantee_character: bool
        When set to True, forces both players to be using character
    """
    # honestly I don't know how to handle the ids that aren't the standard 11 length so I'll just filter them woop
    smash_df = pd.read_csv('smash_vods.csv')
    smash_df['id_len'] = smash_df['video_id'].str.len()
    smash_df = smash_df[smash_df['id_len'] == 11]
    # min date filter
    smash_df['date'] = pd.to_datetime(smash_df['date'])
    if min_date is not None:
        min_date = datetime.datetime.strptime(min_date, '%Y%M%d')
        smash_df = smash_df[smash_df['date'] > min_date]
    # we need to convert the character columns to lists
    smash_df['p1_chars'] = smash_df.p1_chars.apply(lambda x: x[1:-1].replace("'","").split(','))
    smash_df['p2_chars'] = smash_df.p2_chars.apply(lambda x: x[1:-1].replace("'","").split(','))
    # optional lets filter to a specific character?
    if character is not None:
        if guarantee_character:
            smash_df = smash_df[smash_df.p1_chars.apply(lambda x: [character] == x) | smash_df.p2_chars.apply(lambda x: [character] == x)]
        else:
            smash_df = smash_df[smash_df.p1_chars.apply(lambda x: character in x) | smash_df.p2_chars.apply(lambda x: character in x)]
    # and away we goooo
    return list(smash_df['video_id'].values)

def pull_smash_images(num_images,
                      min_date = None,
                      character = None,
                      guarantee_character = False,
                      min_images_per_vid = 5):
    """
    Okay folks this is where it all comes together. We use the smash vods csv and pull num_images from each
    video. Includes the min_date, character, and guarantee_character parameters from get_video_ids. 
    min_images_per_vid will limit the number of videos we pull from in order to get min_images_per_vid per
    video and hit the num_image number of images.
    """
    # get our video IDs
    vid_ids = get_video_ids(min_date = min_date,
                            character = character,
                            guarantee_character = guarantee_character)
    
    # check how many images per video we're getting
    if (len(vid_ids)/num_images) < min_images_per_vid:
        # we need to sample x
        sample_size = int(num_images/min_images_per_vid)
        vid_ids = sample(vid_ids, sample_size)
        images_per_vid = min_images_per_vid
    else:
        images_per_vid = int(len(vid_ids)/num_images)
    print(f'Collecting {images_per_vid} images per video')

    # now we iterate and let it rip
    img_counter = 0
    for id in vid_ids:
        # this print is actually wrong in the event that a video is private/delisted, meaning the whole
        # count logic is wrong too. Might fix later for worth noting for now. End of the day we still get
        # a ton of images and thats enough for me for now
        print(f'pulled {img_counter} images')
        pull_rando_images(f'https://www.youtube.com/watch?v={id}', f'images/{id}_', images_per_vid)
        img_counter += images_per_vid