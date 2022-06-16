import cv2 as cv
import numpy as np

test_path = r'C:\Users\Tim\IRP\Main\version 2.0\data_2\test_data'

confidenceThreshold = 0.5
nms_Threshold = 0.3

classesFile = r'C:\Users\Tim\IRP\Main\version 2.0\data_2\training_data\classes.txt'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = r'C:\Users\Tim\IRP\Main\version 2.0\YOLOv3\EPS_YOLOv3\yolov3.cfg'
modelWeights = r'C:\Users\Tim\IRP\Main\version 2.0\YOLOv3\EPS_YOLOv3\yolov3_training_final.weights'

# Initialize YOLOV3 network
net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIDs = []
    confs = []

    #pred_boxes is for calculating the mean average precision
    bbox_mAP = []
    pred_boxes = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            # np.argmax returns the index of the maximum value in scores.
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidenceThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                if x < 0:
                    x = 0
                elif y < 0:
                    y = 0
                elif x > wT:
                    x = wT
                elif y > hT:
                    y = hT

                bbox.append([x,y,w,h])
                classIDs.append(classID)
                confs.append(float(confidence))

                bbox_mAP.append([classID, confidence,x,y,w,h])

    indicies = cv.dnn.NMSBoxes(bbox,confs,confidenceThreshold,nms_Threshold)

    for i in np.flip(indicies):
        i = i[0]
        box = bbox[i]
        box_mAP = bbox_mAP[i]

        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(img,f'{classNames[classIDs[i]].upper()} {int(confs[i]*100)}%',
                    (x,y+40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # x1 and y1 is TOP LEFT, x2 and y2 is BOTTOM RIGHT
        x2 = (x + w)
        y2 = (y + h)
        box_mAP[4] = x2
        box_mAP[5] = y2
        pred_boxes.append(box_mAP)

    return pred_boxes
#Rescale the size of the frame
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def read_test_images(file_path):

    with open(file_path, 'r') as f:
        print(f.read())

def get_pred_boxes(path):


    img = cv.imread(path)

    blob = cv.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0),True,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)


    pred_boxes = findObjects(outputs, img)

    return pred_boxes