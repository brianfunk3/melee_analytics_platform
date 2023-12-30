from selenium_helpers import getBy, buildHeadless
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from random import randint
import time
import pandas as pd
import numpy as np

def rand_wait(bot = 3, top = 7):
    """
    Randomly returns from a time between bot and top seconds. So that we can scrape
    as politely as possible and not ruin anyone's backend. Woof that kinda sounds
    inappropriate with not context lol

    Parameters:
    bot: int 
        the floor for the random integer to return, default 3

    top: int
        the ceiling for the random integer to return, default 7

    Returns: int
    """
    return randint(bot, top)

def get_row_df(driver):
    """
    Assuming that driver is on the right page on the melee vods website, return a dataframe
    containing all the info we'd want from it, including the video id, date, event, competitors,
    and characters used.

    Parameters:
    driver: selenium.webdriver (on https://vods.co/melee page)
        the driver to pull the table from. Yeah this function should probably take the table
        and not the driver but I reckon I'll be the only one to care and since it works I'm
        not gonna bother

    Returns: pandas DataFrame
    """
    # get the table there
    table = getBy(driver, 'id', 'records-inner').get_attribute('outerHTML')
    # convert to soup
    soup = BeautifulSoup(table, features="lxml")
    # find all the row's we want
    rows = soup.find_all("div", {"class": "match-record"})
    row_dicts = []
    # itereate over each row to individually select what we want
    for row in rows:
        # get the most important part, the video id
        video_id = row['data-video-id']
        # now we need to go a bit deeper for everything else
        spans = row.find_all("span")
        # get the date
        date = spans[0].text.strip()
        # and the name of the event
        event = spans[1].text.strip()
        # check the number of strongs in spans to know if we need to cut it in half
        if len(spans[2].find_all("strong")) > 1:
            player_uno = spans[2].find_all("strong")[-1].text.strip()
            player_uno_char_len = len(spans[2].find_all('img'))
            player_uno_char = [char.get('alt') for char in spans[2].find_all('img')[:-int(player_uno_char_len/2)]]
        else:
            player_uno = spans[2].text.strip()
            player_uno_char = [char.get('alt') for char in spans[2].find_all('img')]
        if len(spans[4].find_all("strong")) > 1:
            player_dos = spans[4].find_all("strong")[-1].text.strip()
            player_dos_char_len = len(spans[4].find_all('img'))
            player_dos_char = [char.get('alt') for char in spans[4].find_all('img')[:-int(player_dos_char_len/2)]]
        else:
            player_dos = spans[4].text.strip()
            player_dos_char = [char.get('alt') for char in spans[4].find_all('img')]
        round = spans[5].text.strip()
        # organize it a bit into a dictionary to hold for a bit
        row_dict = {'video_id': video_id,
                    'date': date,
                    'event': event,
                    'p1': player_uno,
                    'p1_chars': player_uno_char,
                    'p2': player_dos,
                    'p2_chars': player_dos_char,
                    'round': round}
        row_dicts.append(row_dict)
    # now convert the list of dictionaries into a pandas dataframe
    smash_df = pd.DataFrame(row_dicts)
    # some people have some unhinged names and it ruins things
    smash_df = smash_df.replace(r'^\s*$', np.nan, regex=True)
    # fill down date and event since it's not in every row
    smash_df['date'] = smash_df['date'].ffill()
    smash_df['event'] = smash_df['event'].ffill()
    # and out yee go
    return smash_df

def smash_n_grab(max_pages = 100000):
    """
    This function will create a selenium Chrome webdriver and get return a pandas DataFrame
    with historical smash VOD information. The info comes from vods.co/melee, who I'd love
    to support if I could figure out who made/maintains/pays for the website. The website is
    honestly so unbelievably useful and I owe them people who run it a few beers. ANYWAY.

    Parameters:
    max_pages: int
        The maximum number of pages to iterate through. Really just used for testing since
        the max will never actually be that many pages

    Returns: pandas DataFrame 
    """
    # make a driver
    driver = buildHeadless()

    # go to the smash vods site
    driver.get('https://vods.co/melee')
    
    # unhide everything
    getBy(driver, 'id', 'spoiler-toggle').click()

    # take a snooze
    time.sleep(rand_wait())

    # get the max number of pages
    max_count = int(getBy(driver, 'id', 'top-pager').get_attribute('data-total-pages'))
    max_count = min(max_count, max_pages)
    print(f'Max count is {str(max_count)}')
    
    # init our list of rows
    output_list = []

    # get the first one since it's special
    page_one = get_row_df(driver)
    page_one['page'] = 0
    output_list.append(page_one)

    # take a snooze
    time.sleep(rand_wait())

    # go through it all, make dataframes, and add to the list
    for i in range(max_count-1):
        #print out where we are
        print(i)
        
        # navigate to the page
        driver.get(f'https://vods.co/melee?p={i+1}')

        # get our dataframe
        this_page = get_row_df(driver)
        # add a column with the page number
        this_page['page'] = i+1
        
        # add it to the final output list
        output_list.append(this_page)

        # take a snooze
        time.sleep(rand_wait())
    # smash em together and return
    return pd.concat(output_list)