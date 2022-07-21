import argparse
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.ops.boxes as bops
import yolov5
import pandas as pd 
import datetime
import yaml
import json

import os
from mlflow import log_metric, log_param, log_artifacts

import norfair
from norfair.tracker import Detection, Tracker
from norfair.video import Video
from norfair.drawing import Paths

DISTANCE_THRESHOLD_BBOX: float = 3.33
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def crop_image(image, x, y, width, height):
    """
    image: a cv2 frame
    x, y, width, height: the region to cut out
    """
    return image[y:y + height, x:x + width]


def center(points):
    return [np.mean(np.array(points), axis=0)]


def iou_pytorch(detection, tracked_object):
    # Slower but simplier version of iou

    detection_points = np.concatenate([detection.points[0], detection.points[1]])
    tracked_object_points = np.concatenate(
        [tracked_object.estimate[0], tracked_object.estimate[1]]
    )

    box_a = torch.tensor([detection_points], dtype=torch.float)
    box_b = torch.tensor([tracked_object_points], dtype=torch.float)
    iou = bops.box_iou(box_a, box_b)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)
    return np.float(1 / iou if iou else MAX_DISTANCE)


def iou(detection, tracked_object):
    # Detection points will be box A
    # Tracked objects point will be box B.

    box_a = np.concatenate([detection.points[0], detection.points[1]])
    box_b = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and tracker
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)
    return 1 / iou if iou else (MAX_DISTANCE)


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor,
    track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores)
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores)
            )

    return norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
#parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument("--classes", nargs="+", type=int, help="Filter by class: --classes 0, or --classes 0 2 3")
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track_points", type=str, default="centroid", help="Track points: 'centroid' or 'bbox'")
parser.add_argument("--video", type=str, default="0", help="put the video path - or 0 for camera")
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)

#need to figure out embedded way to make this camera - pass argument 
##input_path = args.files
##video = Video(input_path = input_path)
#may want to incorporate some time thing here so saving data every 5-10
video = Video(camera=0)
video =Video(input_path=args.video)  if args.video != "0" else video 

distance_function = iou if args.track_points == "bbox" else euclidean_distance
distance_threshold = (
DISTANCE_THRESHOLD_BBOX
if args.track_points == "bbox"
else DISTANCE_THRESHOLD_CENTROID
)

tracker = Tracker(
distance_function=distance_function,
distance_threshold=distance_threshold,
)
paths_drawer = Paths(center, attenuation=0.01)

#initiating way to save 
#gonna try dictionary 
#peds = pd.DataFrame(columns=['ID','Locations'])
peds = {}
count = 0
for frame in video:
    #check if this is making it a lot slower 
    frame = crop_image(frame,350,350,650,500)
    yolo_detections = model(
        frame,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thresh,
        image_size=args.img_size,
        classes=args.classes
    )
    detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)
    tracked_objects = tracker.update(detections=detections)
    if args.track_points == "centroid":
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
    elif args.track_points == "bbox":
        norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_boxes(frame, tracked_objects)
    frame = paths_drawer.draw(frame, tracked_objects)
    video.write(frame)
    #modify print obejcts as table to print ongoing count 
    #print_objects_as_table(tracked_objects)
    for obj in tracked_objects:
        if obj.id in peds.keys():
            #peds.loc[peds.ID == obj.id, peds.Locations] += [obj.estimate]
            #print(type(obj.estimate[0]))
            #start with ID - Location dictionary 
            #add in time 
            now = datetime.datetime.now()
            peds[obj.id].append((obj.estimate[0],now.time()))
            #print(type(peds.loc[peds.ID == obj.id]['Locations']))
            #peds.loc[peds.ID == obj.id]['Locations']+=tuple(obj.estimate[0])
            #obj.estimate is array 
            #peds.loc[peds.ID == obj.id]['Locations'] is series 
        else:
            count += 1
            #add new key and value 

            now = datetime.datetime.now()
            peds[obj.id] = [(obj.estimate[0],now.time())]

            #now = datetime.datetime.now()
            #peds.loc[len(peds.index)] = [obj.id, tuple(obj.estimate[0])] 
            #this is not working - increasing count too much somehow? 
            #peds.append((obj.id,obj.estimate))
            #estimate is x and y --> impute time of each - fast way to do this? 
            #then add to dataframe then write data frame 
    print(count)
    #save dataframe
print(peds)
#worth it to try getting shape of dataframe? - num of unique IDs?
#currently counting each person each frame          
# Log an artifact (output file)
if not os.path.exists("outputs"):
    os.makedirs("outputs")
with open("outputs/count.txt", "w") as f:
    f.write(str(peds))
log_artifacts("outputs")

