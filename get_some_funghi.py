import os
import requests
from lxml import html

search_url = "https://eol.org/search?q={}"

edible_funghi = []
poisonous_funghi = []
inedible_funghi = []

with  os.scandir("/home/yagiz/Sourcebox/git/Transfer-Learning-Suite/dataset/val/edible/") as entries:
    for entry in entries:
        edible_funghi.append(entry.name[2:-6]) if entry.name[2:-6] not in edible_funghi else edible_funghi
with  os.scandir("/home/yagiz/Sourcebox/git/Transfer-Learning-Suite/dataset/val/inedible/") as entries:
    for entry in entries:
        inedible_funghi.append(entry.name[2:-6]) if entry.name[2:-6] not in inedible_funghi else inedible_funghi
with  os.scandir("/home/yagiz/Sourcebox/git/Transfer-Learning-Suite/dataset/val/poisonous/") as entries:
    for entry in entries:
        poisonous_funghi.append(entry.name[2:-6]) if entry.name[2:-6] not in poisonous_funghi else poisonous_funghi


def get_images(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    import ipdb; ipdb.set_trace()

    funghi_images_list = tree.xpath("//a[@class='swipebox']")
    count = 0
    name = ""
    for funghi_image in funghi_images_list:
        if "inedible" in url:
            category = "i"
        elif "poisonous" in url:
            category = "p"
        else:
            category = "e"

        if funghi_image.get("title") == name:
            count += 1
        else:
            count = 0

        funghi_image_url = "http://www.mushroom.world" + funghi_image.get("href")[3:]
        img_data = requests.get(funghi_image_url).content
        name = funghi_image.get("title")
        filename = '{}_{}_{}.jpg'.format(category, name ,count)
        print(filename)

        with open(filename, 'wb') as handler:
            handler.write(img_data)


for funghi_name in edible_funghi:
    search_url = search_url.format(funghi_name).replace(" ", "%20")
    get_images(search_url)
