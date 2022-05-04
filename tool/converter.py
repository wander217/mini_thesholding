import json
import os.path
import random
import cv2 as cv
import numpy as np

label_file = r"C:\Users\Trinh_Thinh\Downloads\syntext2\syntext2\train.json"
image_file = r'C:\Users\Trinh_Thinh\Downloads\syntext2\syntext2\images\emcs_imgs'
image_save_dir = r'D:\syntext\image'
label_save_dir = r'D:\syntext\target1.json'
with open(label_file, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())
image = data['images']
print(image[0])
image_data = {}
for item in image:
    image_data[item['id']] = {
        "file_name": item['file_name'],
        "target": []
    }
annotation = data['annotations']
for item in annotation:
    image_data[item['image_id']]['target'].append({
        'bbox': np.array(item['bezier_pts']).reshape((-1, 2)).tolist(),
        'text': '!!!'
    })
start = 94724
item_data = [value for key, value in image_data.items()]
for i, item in enumerate(item_data):
    image = cv.imread(os.path.join(image_file, item['file_name']))
    h, w, c = image.shape
    scale = min([640 / w, 640 / h])
    new_h = int(scale * h)
    new_w = int(scale * w)
    image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    cv.imwrite(os.path.join(image_save_dir, "img{}.jpg".format(start + i)), image)
    item['file_name'] = "img{}.jpg".format(start + i)
    item['shape'] = [new_w, new_h]
    for polygon in item['target']:
        tmp = np.array(polygon['bbox']) / np.array([w, h]) * np.array([new_w, new_h])
        x_min = tmp[:, 0].min()
        x_max = tmp[:, 0].max()
        y_min = tmp[:, 1].min()
        y_max = tmp[:, 1].max()
        tmp = np.array([
            [x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max]
        ])
        # cv.polylines(image, [tmp.astype(np.int32)], True, (255, 0, 0))
        polygon['bbox'] = tmp.tolist()
    # cv.imshow("image", image)
    # cv.waitKey(0)
    # break

with open(label_save_dir, 'w', encoding='utf-8') as f:
    f.write(json.dumps(image_data))
