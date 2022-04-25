import cv2
import numpy as np

import requests
import json

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
import sys,os
import math
from datetime import datetime
from datetime import date

import os

with open('program_config.json', 'r') as f:
  json_data = json.load(f)


# loading model parameters from json
coco_file_path = json_data["COCO_FILE_PATH"]
weight_file_path = json_data["WEIGHT_FILE_DIR"]
conf_file_path = json_data["CONF_FILE_DIR"]
tracker_model_file = json_data["TRACKER_MODEL_FILE"]
danger_folder_path = json_data["DANGER_PERSON_DIST_IMG"]
moderate_folder_path = json_data["MODERATE_PERSON_DIST_IMG"]
video_path = json_data["VIDEO_PATH"]

input_image_size = json_data["INPUT_IMAGE_RESOLUTION"]
detection_threshold = json_data["MODEL_THRESHOLD"]
nms_threshold = json_data["NMS_THRESHOLD"]
max_cosin_dist = json_data["MAX_COSINE_DISTANCE"]
sd_core_max_thresh = json_data["SD_MAX_DIST_THRESHOLD"]
sd_danger_thresh = json_data["SD_DANGER_THRESHOLD"]
sd_moderate_thresh = json_data["SD_MODERATE_THRESHOLD"]
video_width_reso = json_data["VIDEO_WIDTH"]
video_height_reso = json_data["VIDEO_HEIGHT"]

# Reading Video File 
cap = cv2.VideoCapture(video_path)
current_date = date.today()

whT = input_image_size

classesFile = coco_file_path
classNames = []

close_people_track = []
modrate_people_track = []

# Loading all yolov3 pre-trained classes 
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Model Detection Threshold 
confThreshold = detection_threshold

# reduce the value if there are multiple wrong detections
nmsThreshold = nms_threshold

# NMS distance threshold
max_cosine_distance = max_cosin_dist
nn_budget = None


modelConfigrations = conf_file_path
modelWeights = weight_file_path

model_filename = tracker_model_file
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


net = cv2.dnn.readNetFromDarknet(modelConfigrations,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def calculateCentroid(xmin,ymin,xmax,ymax):
    # calculating the center point of a bounding box
    xmid = ((xmax+xmin)/2)
    ymid = ((ymax+ymin)/2)
    centroid = (xmid,ymid)

    return int(xmid),int(ymid)

def get_distance(x1,y1,x2,y2):
    # calculating euclidean distance
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    return distance

def findObjects(outputs,img):
    try:
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold and classNames[classId] == "person":
                    w,h = int(det[2]*wT) , int(det[3]*hT)
                    x,y = (int(det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        # NMS returns the index of the detection boxs to keep
        indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
        
        new_bbox = []
        new_classNames = []
        new_confs = []

        for i in indices:
            i = i[0]
            box = bbox[i]
            x,y,w,h = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            new_bbox.append([x,y,w,h])
            new_classNames.append(classNames[classIds[i]])
            new_confs.append(confs[i])
        
        boxes = np.array(new_bbox) 
        names = np.array(new_classNames)
        scores_conf = np.array(new_confs)
        features = np.array(encoder(img, boxes))

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores_conf, names, features)]

        tracker.predict()
        tracker.update(detections)

        tracked_bboxes = []


        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                    continue
            bbox = track.to_tlbr()
            tracking_id = track.track_id # Get the ID for the particular track
            tracked_bboxes.append([int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),tracking_id])

        for single_person in tracked_bboxes:
            main_cent_x,main_cent_y = calculateCentroid(int(single_person[0]), int(single_person[1]) ,int(single_person[2]), int(single_person[3]))

            for other_person in tracked_bboxes:
                if single_person != other_person:
                    secd_cent_x,secd_cent_y = calculateCentroid(int(other_person[0]), int(other_person[1]) ,int(other_person[2]), int(other_person[3]))
                    euclidean_dist = get_distance(main_cent_x,main_cent_y,secd_cent_x,secd_cent_y)
                    

                    if int(single_person[0]) < int(other_person[2]) and int(single_person[1]) < int(other_person[3]):
                                    # crop_img = img[int(single_person[0]):int(single_person[1]),int(other_person[2]):int(other_person[3])]
                        crop_img = img[int(single_person[1]):int(other_person[3]),int(single_person[0]):int(other_person[2])]
                    else:
                        crop_img = img[int(other_person[1]):int(single_person[3]),int(other_person[0]):int(single_person[2])]
                    

                    img_name = "pp1_"+ str(single_person[4]) + "_pp2" + str(other_person[4]) + ".jpg"
                    
                    # Eliminating all the euclidean distances above 100
                    # If the distance in lower than 30 then it is consider that people are too close to each other_person
                    # If the distance is above 30 and below 60 the it is consider has moderte distance
                    if euclidean_dist < 100:
                        close_ppl_track = (single_person[4],other_person[4])
                        close_ppl_track_rev = (other_person[4],single_person[4])

                        
                        if euclidean_dist <= 30:
                            # print("min euclidean_dist === >",euclidean_dist)
                            
                            
                            if close_ppl_track not in close_people_track:
                                close_people_track.append(close_ppl_track)
                                close_people_track.append(close_ppl_track_rev)
                                filename = os.path.join(danger_folder_path,img_name)
                                # filename = danger_folder_path + img_name 
                                try:
                                    h,w,c = crop_img.shape
                                    if h > w and h > 30 and w > 30:
                                        # Saving the images with person violating the social distance 
                                        cv2.imwrite(filename,crop_img)
                                    # print("Image Saved")
                                except Exception as e:
                                    print("Error in saving img",e)

                            # img = cv2.putText(img,"Dist:"+str(euclidean_dist),(int(main_cent_x),int(main_cent_y) - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
                            img = cv2.line(img, (main_cent_x,main_cent_y), (secd_cent_x,secd_cent_y), (0,0,255), 2)
                        elif euclidean_dist < 60 and euclidean_dist > 30:
                            # print("max euclidean_dist === >",euclidean_dist)
                            if close_ppl_track not in modrate_people_track:
                                modrate_people_track.append(close_ppl_track)
                                modrate_people_track.append(close_ppl_track_rev)
                                filename = os.path.join(moderate_folder_path,img_name)
                                try:
                                    h,w,c = crop_img.shape
                                    if h > w and h > 50 and w > 50:
                                        filename = os.path.join(moderate_folder_path,img_name)
                                        cv2.imwrite(filename,crop_img)
                                    # print("Image Saved")
                                    now = datetime.now()

                                except Exception as e:
                                    print("Error in saving img",e)
                        else:
                            pass

            img = cv2.rectangle(img, (int(single_person[0]), int(single_person[1])), (int(single_person[2]), int(single_person[3])),(0,255,0), 2)
            img = cv2.putText(img,"ID:"+str(single_person[4]),(int(single_person[0]),int(single_person[1]) - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Erron in Find Objects === >",exc_type, fname, exc_tb.tb_lineno)

    return img 





while True:
    ret,frame = cap.read()

    if ret is True:
        frame = cv2.resize(frame,(video_width_reso,video_height_reso))

    blob = cv2.dnn.blobFromImage(frame,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    
    frame = findObjects(outputs,frame)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)

    cv2.imshow("Output",frame)
    cv2.waitKey(1)    