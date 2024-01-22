from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

"""
This is a file I've used for basically any webscraping I've done in the last like
5 years. I love it. It's not exactly best practices but best practices aren't made for
fun projects like this
"""

def buildHeadless(headless = True):
    options = ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("enable-automation")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-browser-side-navigation")
    options.add_argument("--disable-gpu")
    options.add_argument("--mute-audio")
    if headless:
        options.add_argument("--headless")
    driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()), options = options)
    driver.set_page_load_timeout(5)

    return driver

def getBy(driverIn, byType, key, delay = 10):
    if (byType== 'id'):
        try:
            myElem = WebDriverWait(driverIn, delay).until(EC.presence_of_element_located((By.ID, key)))
            return myElem
        except TimeoutException:
            return False
    elif (byType == 'xpath'):
        try:
            myElem = WebDriverWait(driverIn, delay).until(EC.presence_of_element_located((By.XPATH, key)))
            return myElem
        except TimeoutException:
            return False
    elif (byType == 'class'):
        try:
            myElem = WebDriverWait(driverIn, delay).until(EC.presence_of_element_located((By.CLASS_NAME, key)))
            return myElem
        except TimeoutException:
            return False
    elif (byType == 'name'):
        try:
            myElem = WebDriverWait(driverIn, delay).until(EC.presence_of_element_located((By.NAME, key)))
            return myElem
        except TimeoutException:
            return False
    elif (byType == 'css'):
        try:
            myElem = WebDriverWait(driverIn, delay).until(EC.presence_of_element_located((By.CSS_SELECTOR, key)))
            return myElem
        except TimeoutException:
            return False
    elif (byType == 'tag'):
        try:
            myElem = WebDriverWait(driverIn, delay).until(EC.presence_of_element_located((By.TAG_NAME, key)))
            return myElem
        except TimeoutException:
            return False
    elif (byType == 'link_text'):
        try:
            myElem = WebDriverWait(driverIn, delay).until(EC.presence_of_element_located((By.LINK_TEXT, key)))
            return myElem
        except TimeoutException:
            return False
    else:
        print("Please use a valid byType")
        return False