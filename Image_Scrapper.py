import requests
from bs4 import BeautifulSoup
import urllib
import cv2

URL = 'https://www.google.com/search?tbm=isch&sxsrf=ACYBGNSAHOMDD_W4tUseuooQsWVIYdlEbA%3A1572927237445&source=hp&biw=1440&bih=711&ei=BffAXb2UGPSU0PEPhe27QA&q=pikachu+cartoon&oq=pikachu+cartoon&gs_l=img.3..0l10.1500.4906..5596...0.0..0.172.1225.11j4......0....1..gws-wiz-img.......35i39.k4dxc89N3uw&ved=0ahUKEwj9qvjJmtLlAhV0CjQIHYX2DggQ4dUDCAY&uact=5'
r = requests.get(URL)

soup = BeautifulSoup(r.content,'html5lib')
image_results = soup.find('div', {'id':'search'})


# name of the pokemon we are getting images for
pokemon_name = 'Pikachu'

count = 0
# loop through all images in results
for img in image_results.find_all('img'):
	# get image url link
	img_url = img.get('src')
	print(img_url)

	filename = pokemon_name + str(count)
	
	cv2.imwrite('/dataset/' + filename + 'jpg', img_url)
	
	count += 1

print('Downloaded {} images.'.format(count))