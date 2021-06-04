from requests import Session
from requests_futures.sessions import FuturesSession
from bs4 import BeautifulSoup
from os import path, mkdir
import re
import pandas as pd
import time
import cchardet
import lxml
from numpy import random
from http_request_randomizer.requests.proxy.requestProxy import RequestProxy
import sys


# ----------------------- Scraping -----------------------


class Scraper:
    def __init__(self, data_path: str, file_name: str):
        """
        Initializing the parameters
        :param data_path: path of the data directory on Google drive
        :param file_name: Name of the original csv file to be scraped
        """

        self.filename = file_name
        self.datasets = path.join(datapath, 'datasets')
        self.scraped = path.join(datapath, 'Scraped')

        req_proxy = RequestProxy()
        self.proxy_list = req_proxy.get_proxy_list()
        http_proxy = None
        https_proxy = None
        self.proxyDict = {
            "http": http_proxy,
            "https": https_proxy,
        }
        self.import_data()

    def import_data(self):
        self.scraped_filename = path.join(self.scraped, self.filename[:self.filename.find('.csv')] + '_scraped.csv')
        if not path.exists(self.scraped):
            mkdir(self.scraped)

        self.table = pd.read_csv(path.join(self.datasets, self.filename))
        self.table = self.table[[colname for colname in self.table.columns if 'Unnamed:' not in colname]]

        old_cols = self.table.columns.to_list()

        scrape_cols = ['image', 'has_video', 'story', 'risks', 'creator_bio', 'n_tiers', 'tiers_values',
                       'n_images', ' n_gifs', 'n_websites', 'fb_linked', 'n_collab',
                       'collab_names']
        self.cols = old_cols + scrape_cols
        self.df = pd.DataFrame(columns=self.cols)
        if path.exists(self.scraped_filename):
            self.df = pd.read_csv(self.scraped_filename)
            # cut table to restart after last index of previous run
            self.table = self.table[self.df.index.stop:]

    def scrape(self):

        global i
        start = time.time()
        k = False
        fs = FuturesSession()
        sess = Session()
        r = fs.get("https://www.kickstarter.com").result()
        soup = BeautifulSoup(r.text, 'html.parser')
        xcsrf = soup.find("meta", {"name": "csrf-token"})["content"]
        headers = {
            "x-csrf-token": xcsrf
        }
        query = """
        query Campaign($slug: String!) {
          project(slug: $slug) {
            risks
            story(assetWidth: 680)
          }
        }"""

        last_10 = []
        for i, row in self.table.iterrows():
            url = row['project_url']
            successful = False
            page = time.time()
            slug = re.search('/projects/(.*)\?', url).group(1)
            url = re.search('(.*)\?', url).group(1)
            print(f"------ Page {i}: {slug} ------")

            while not successful:
                try:
                    rs = [fs.get(url, timeout=10), fs.post("https://www.kickstarter.com/graph", proxies=self.proxyDict,
                                                           headers=headers,
                                                           json={
                                                               "operationName": "Campaign",
                                                               "variables": {
                                                                   "slug": slug
                                                               },
                                                               "query": query
                                                           }, timeout=25)]

                    time.sleep(1.5)
                    r = None
                    r = sess.get(url + '/creator_bio')

                    soup = BeautifulSoup(r.content, 'lxml')
                    ## Websites
                    websites = 0
                    if soup.find("ul", {'class': 'links list f5 bold'}):
                        websites = len(soup.find("ul", {'class': 'links list f5 bold'}).find_all('li'))

                    ## Bio
                    bio = soup.find('div', {'class': 'readability'})
                    if bio:
                        bio_text = []
                        for p in bio.find_all('p'):
                            if p.find(text=True) != None:
                                bio_text.append(p.find(text=True))

                        bio_text = ' '.join(bio_text)
                    else:
                        bio_text = None
                    ## Facebook
                    fb = soup.find("div", {'class': "facebook py2 border-bottom f5"})
                    fb_linked = fb.find("a", {'class': 'popup'}) != None

                    # Collaborators
                    n_collab = 0
                    collaborators = None
                    if soup.find("div", {'class': 'pt3 pt7-sm mobile-hide row'}):
                        collaborators = soup.find("div", {'class': 'pt3 pt7-sm mobile-hide row'}).findChildren('a')
                        n_collab = len(collaborators)

                    # names
                    collab_names = []
                    if collaborators:
                        for col in collaborators:
                            collab_names.append(re.search(r'/profile/(.*?)/about', str(col)).group(1))

                    r = None
                    r = rs[0].result()
                    soup = BeautifulSoup(r.content, 'lxml')

                    ## Image
                    image = soup.find('img')['src']
                    ## Video
                    has_video = soup.find('video') is not None
                    ## Pledge tiers
                    tiers = soup.find_all("div", {"class": "pledge__info"})
                    tiers_values = []
                    for tier in tiers:
                        s = str(tier.find("span", {"class": "money"}))
                        if re.search(r'\d+', s):
                            tiers_values.append(int(re.findall(r'\d+', s)[0]))
                        else:
                            tiers_values.append('0')
                    n_tiers = len(tiers_values)

                    r = None
                    r = rs[1].result()
                    result = r.json()

                    story_html = result["data"]["project"]["story"]
                    story = BeautifulSoup(story_html, 'html.parser')
                    n_gifs = len(story.find_all('img', {'class': "fit js-lazy-image"}))
                    n_images = len(story.find_all('img')) - n_gifs

                    # text
                    story_text = ' '.join([p for p in story.find_all(text=True) if i not in ['\n', ' ']])

                    risks = result["data"]["project"]["risks"]

                    self.df.loc[i] = pd.Series(
                        self.table.loc[i].values.tolist() + [image, has_video, story_text, risks, bio_text, n_tiers,
                                                             tiers_values, n_images, n_gifs, websites,
                                                             fb_linked, n_collab, collab_names], index=self.cols)
                    successful = True
                    if time.time() - page < 2:
                        time.sleep(2 - (time.time() - page))
                    print('Time for this page was {}s'.format(round(time.time() - page, 2)))

                    last_10.append(time.time() - page)
                    if len(last_10) == 11:
                        last_10.pop(0)
                        if sum([x >= 5 for x in last_10]) >= 5:
                            print('Proxy too slow')
                            proxy = self.proxy_list.pop(random.choice(len(self.proxy_list))).get_address()
                            if len(self.proxy_list) == 0:
                                req_proxy = RequestProxy()
                                self.proxy_list = req_proxy.get_proxy_list()
                            http_proxy = 'http://' + proxy
                            https_proxy = 'https://' + proxy
                            self.proxyDict = {
                                "http": http_proxy,
                                "https": https_proxy,
                            }

                            last_10 = []

                except KeyboardInterrupt:
                    k = True
                    print('interrupted!')
                    break

                except:
                    if r and r.status_code == 200:
                        time.sleep(1.5)
                        print('Problems parsing, scraping skipped!!')
                        self.df.loc[i] = self.table.loc[i]
                        successful = True
                    elif r and r.status_code == 429:
                        print("Too many requests, rotate ip")
                        time.sleep(5)
                        proxy = self.proxy_list.pop(random.choice(len(self.proxy_list))).get_address()
                        if len(self.proxy_list) == 0:
                            req_proxy = RequestProxy()
                            proxy_list = req_proxy.get_proxy_list()
                        http_proxy = 'http://' + proxy
                        https_proxy = 'https://' + proxy
                        self.proxyDict = {
                            "http": http_proxy,
                            "https": https_proxy,
                        }

                    else:
                        print("Bad Proxy")
                        time.sleep(5)
                        proxy = self.proxy_list.pop(random.choice(len(self.proxy_list))).get_address()
                        if len(self.proxy_list) == 0:
                            req_proxy = RequestProxy()
                            self.proxy_list = req_proxy.get_proxy_list()
                        http_proxy = 'http://' + proxy
                        https_proxy = 'https://' + proxy
                        self.proxyDict = {
                            "http": http_proxy,
                            "https": https_proxy,
                        }

            if k:
                break

            if i % 500 == 0:
                print(f'\n\n ----- Reached Page {i}, saving dataframe to {self.scraped_filename} ----- \n\n')
                self.df.to_csv(self.scraped_filename, index=False)

        print(f'Total time for {i - self.table.index.start} pages is {round(time.time() - start, 2)} seconds')

        self.df.to_csv(self.scraped_filename, index=False)


if __name__ == "__main__":
    name = sys.argv[1]
    datapath = '/content/drive/MyDrive/Project/Data/'
    scraper = Scraper(data_path=datapath, file_name=name)
    scraper.scrape()
