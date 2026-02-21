import json
import re
import requests
from bs4 import BeautifulSoup
import time
def clean_html(text):
    return re.sub(r"<[^>]*>", "", text)
def crawl():
    biology_data = []
    files = ['page1.json', 'page2.json', 'page3.json', 'page4.json', 'page5.json', 'page6.json', 'page7.json']
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            data = json.load(f)
            for item in data['searchResults']:
                biology_data.append(item)
    with open('biology.json', 'w', encoding='utf8') as f:
        json.dump(biology_data, f, ensure_ascii=False)

def clean():
    with open('biology.json', 'r', encoding='utf8') as f:
        data = json.load(f)
        biology_data = []
        id = 0
        for item in data:
            if(id < 50):
                biology_data.append({
                'title': clean_html(item['title']),
                'abstract': 'https://www.sciencedirect.com' + item['link'],
            })
            else:
                biology_data.append({
                'title': clean_html(item['title']),
                'abstract': '',
            })
            print('Crawling', id)
            id += 1
            time.sleep(0)
    with open('biology_data.json', 'w', encoding='utf8') as f:
        json.dump(biology_data, f, ensure_ascii=False)
