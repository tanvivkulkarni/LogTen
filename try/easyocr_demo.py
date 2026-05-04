import requests

files = {'file': open('images\\LB1 page 2_page-0001.jpg', 'rb')}
response = requests.post('https://api.easyocr.org/ocr', files=files)
result = response.json()
print(result['words'])