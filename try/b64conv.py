import base64

with open("images\\LB1 page 2_page-0001.jpg", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

with open("output.txt", "w") as f:
    f.write(encoded)