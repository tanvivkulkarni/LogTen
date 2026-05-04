import base64
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("images\\LB1 page 2_page-0001.jpg", "rb") as image_file:
    content = base64.b64encode(image_file.read())

image = vision.Image(content=content)

response = client.label_detection(image=image)

for label in response.label_annotations:
    print(label.description)