import json
import os.path
import numpy as np
import cv2 as cv

# target_files = [
#     r'D:\syntext\target.json',
#     r'D:\syntext\target1.json'
# ]
#
# data = []
# for target_file in target_files:
#     with open(target_file, 'r', encoding='utf-8') as f:
#         data.extend(list(json.loads(f.readline()).values()))
#
# train_data = data[:139050]
# valid_data = data[139050:]
#
# with open(r"D:\syntext\train.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(train_data))
#
# with open(r"D:\syntext\valid.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(valid_data))

with open(r"D:\syntext\train.json", 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())

image = cv.imread(os.path.join(r'D:\syntext\image', data[0]['file_name']))
for bbox in data[0]['target']:
    cv.polylines(image, [np.array(bbox['bbox']).astype(np.int32)], True, (255, 0, 0))
cv.imshow("abc", image)
cv.waitKey(0)
