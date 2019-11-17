import requests
from bs4 import BeautifulSoup
import urllib.request
import cv2
import numpy as np

# name of the pokemon we are getting images for
POKEMON_NAME = 'Squirtle'
MAX_IMAGES = 5

# Pikachu Google Image Search
URL = 'https://www.google.com/search?q=squirtle&rlz=1C5CHFA_enUS814US814&sxsrf=ACYBGNSpBufdU77j-MUSKmN8cS_plt6I6Q:1573446076162&source=lnms&tbm=isch&sa=X&ved=0ahUKEwijj8Wzp-HlAhXfIjQIHZm1D-YQ_AUIEigB&biw=651&bih=648'
r = requests.get(URL)

soup = BeautifulSoup(r.content,'html5lib')

# Gets the HTML code within the image results container
search_results = soup.find('div', {'id':'search'})

#print(search_results.prettify())

# loop through all images in results
count = 0
for result in search_results.find_all('img')[:MAX_IMAGES]:
    try:
        # get image url link
        img_url = result.get('src')
        print(img_url)
        request = urllib.request.Request(img_url)
        response = urllib.request.urlopen(request)
        binary_str = response.read()
        byte_array = bytearray(binary_str)
        numpy_array = np.asarray(byte_array, dtype="uint8")
        image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
        filename = '/dataset/%s/scrapped_%d.jpg' % (POKEMON_NAME, count)
        cv2.imwrite(filename, image)
        print('Saved image %d' % count)
        count += 1
    except Exception as e:
        print(str(e))

print("DONE")
print('Downloaded {} images.'.format(count))


verify_image = cv2.imread('dataset/Squirtle/scrapped_0.jpg', 0)
cv2.imshow('Image', verify_image)