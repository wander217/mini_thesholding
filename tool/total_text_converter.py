import json
import os
import random
import cv2 as cv
import numpy as np

# train_image_dir = r'C:\Users\Trinh_Thinh\Downloads\ch4_test_images'
# train_target_dir = r'C:\Users\Trinh_Thinh\Downloads\icdar2015\icdar2015\test_gts'
#
# save_dir = r'D:\icdar15\valid'
# if not os.path.isdir(os.path.join(save_dir, "image/")):
#     os.mkdir(os.path.join(save_dir, "image/"))
#
# data = []
# for i, train_target_path in enumerate(os.listdir(train_target_dir)):
#     print(train_target_path[3:])
#     train_image_file = '.'.join([train_target_path[3:].split(".")[0], "jpg"])
#     image = cv.imread(os.path.join(train_image_dir, train_image_file))
#     h, w, c = image.shape
#     scale = min([640 / w, 640 / h])
#     new_h, new_w = int(scale * h), int(scale * w)
#     image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
#     cv.imwrite(os.path.join(save_dir, "image/image{}.jpg".format(i)), image)
#     with open(os.path.join(train_target_dir, train_target_path), 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         target = []
#         for line in lines:
#             tmp = line.strip("\ufeff").strip("\r\t").strip("\n").strip().split(",")
#             last = 2 if len(tmp) % 2 == 0 else 1
#             bbox = np.array(tmp[:-last]).astype(np.int32).reshape((-1, 2))
#             bbox = bbox / np.array([w, h]) * np.array([new_w, new_h])
#             text = tmp[-last]
#             target.append({
#                 "bbox": bbox.tolist(),
#                 "text": text
#             })
#         data.append({
#             "file_name": "image{}.jpg".format(i),
#             "target": target
#         })
#
# with open(os.path.join(save_dir, "target.json"), 'w', encoding='utf-8') as f:
#     f.write(json.dumps(data))

# with open(r"D:\total_text\train\target.json", 'r', encoding='utf-8') as f:
#     data = json.loads(f.readline())
#
# for i, selected in enumerate(data):
#     # selected = data[random.randint(0, len(data) - 1)]
#     image = cv.imread(os.path.join(r'D:\total_text\train\image', selected['file_name']))
#     for target in selected['target']:
#         cv.polylines(image, [np.array(target['bbox']).astype(np.int32)], True, (0, 0, 255), 2)
#     cv.imshow("abc", image)
#     cv.waitKey(0)

mask_dir = r'C:\Users\Trinh_Thinh\Downloads\groundtruth_pixel\Test'
save_dir = r'D:\total_text\valid'

if not os.path.isdir(os.path.join(save_dir, "mask/")):
    os.mkdir(os.path.join(save_dir, "mask/"))

data = []
for i, mask_path in enumerate(os.listdir(mask_dir)):
    image = cv.imread(os.path.join(mask_dir, mask_path))
    h, w, c = image.shape
    scale = min([640 / w, 640 / h])
    new_h, new_w = int(scale * h), int(scale * w)
    image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    cv.imwrite(os.path.join(save_dir, "mask/image{}.jpg".format(i)), image)
