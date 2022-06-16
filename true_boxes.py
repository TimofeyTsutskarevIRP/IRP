import os
import cv2 as cv

test_path = r'C:\Users\Tim\IRP\Main\version 2.0\data_2\test_data'

def read_text_file(file_path):

    with open(file_path, 'r') as f:
        print(f.read())

def get_true_boxes(path, model_type = "EPS"):

    os.chdir(path)
    true_boxes_path = []
    images_paths = []

    for file in os.listdir():
        if file.endswith('.txt'):
            file_path = f"{test_path}\{file}"
            true_boxes_path.append(file_path)
        elif file.endswith('.jpg') or file.endswith('jpeg'):
            image_path = f"{test_path}\{file}"
            images_paths.append(image_path)

    del true_boxes_path[-1]

    true_boxes = []

    for text_file in true_boxes_path:
        image_boxes = []
        text_tag = text_file[-7:-4]
        for image in images_paths:
            if image.endswith('jpg'):
                image_tag = image[-7:-4]

                if image_tag == text_tag:

                    img = cv.imread(image)
                    hT, wT, cT = img.shape

                    with open(text_file, 'r') as f:
                        for line in f.read().splitlines():
                            bounding_box = []
                            split_line = line.split()

                            w, h = float(split_line[3])*wT, float(split_line[4])*hT

                            x1 = int(float(split_line[1])*wT - w/2)
                            y1 = int(float(split_line[2])*hT - h/2)
                            x2 = int(float(split_line[1])*wT + w/2)
                            y2 = int(float(split_line[2])*hT + h/2)

                            bounding_box.append(text_tag)
                            if model_type == "COCO":
                                bounding_box.append(0)
                            else:
                                bounding_box.append(int(split_line[0]))
                            bounding_box.append(x1)
                            bounding_box.append(y1)
                            bounding_box.append(x2)
                            bounding_box.append(y2)


                            image_boxes.append(bounding_box)
                            true_boxes.append(bounding_box)



    return true_boxes

boxes = get_true_boxes(test_path)