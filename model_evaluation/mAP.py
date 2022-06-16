import torch
from collections import Counter
from model_evaluation.iou import intersection_over_union

TP_list = []
FP_list = []

def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, num_classes=2
):
    # true_boxes is a list of all the annotated boxes in the test dataset
    # pred_boxes is a list of all the boxes in the test dataset predicted by the model
    # true_boxes (list): [[train_idx, class_pred, x1, y1, x2, y2], ...]
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precision = []
    average_precision_1 = []
    average_precision_2 = []
    epsilon = 1e-6
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes (per image) = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}

        #sorts the detections by probability score
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            # ground_truth_img is a list of 'true' bounding boxes in 1 image
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[2:]),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        TP_list.append(int(torch.max(TP_cumsum)))
        FP_cumsum = torch.cumsum(FP, dim=0)
        FP_list.append(int(torch.max(FP_cumsum)))
        recalls = TP_cumsum / (total_true_bboxes)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum))

        #concatenate
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        #calculate the average precision per class
        average_precision.append(torch.trapz(precisions, recalls))

        if c == 0:
            average_precision_1.append(average_precision)
        if c == 1:
            average_precision_2.append(average_precision)

    return sum(average_precision) / len(average_precision), precisions, recalls, TP_list, FP_list
