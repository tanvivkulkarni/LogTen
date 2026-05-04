from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

result = ocr.ocr('images/LB1 page 1_page-0001.jpg', cls=True)

with open("output_paddle.txt", "w", encoding="utf-8") as f:
    for line in result:
        for word in line:
            f.write(word[1][0] + "\n")

print("✅ Saved")