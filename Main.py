import os
import matplotlib.pyplot as plt
from termcolor import colored
import true_boxes
from YOLOv3.EPS_YOLOv3 import pred_boxes
from model_evaluation import mAP

test_path = r'C:\Users\Tim\IRP\Main\version 2.0\data_2\test_data'

os.chdir(test_path)
test_images_paths = []
all_pred_boxes = []

for file in os.listdir():
    if file.endswith('.jpg'):
        file_path = f"{test_path}\{file}"
        test_images_paths.append(file_path)

for image in test_images_paths:
    image_tag = image[-7:-4]
    image_boxes = pred_boxes.get_pred_boxes(image)
    for box in image_boxes:
        box.insert(0, image_tag)
        all_pred_boxes.append(box)

true_boxes = true_boxes.get_true_boxes(test_path)

# for true, predicted in zip(true_boxes, all_pred_boxes):
#     print(colored(true, 'green'), colored(predicted, 'red'))

iou_range = range(50, 95, 5)

all_mAPs = []

# for iou in iou_range:
#     mAP_single_IOU, precisions, recalls = mAP.mean_average_precision(all_pred_boxes, true_boxes, iou_threshold=(iou/100))
#     all_mAPs.append(mAP_single_IOU)
#
# print(sum(all_mAPs)/len(all_mAPs))

mAP_single_IOU, precisions, recalls, TP, FP = mAP.mean_average_precision(all_pred_boxes, true_boxes, iou_threshold=0.5)

print(mAP_single_IOU)
print(recalls)
print(precisions)

plt.plot(recalls, precisions)
plt.title("Precision-Recall Graph for EPS")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


