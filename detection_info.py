import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load YOLO object detector and Haar Cascade face detector
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define classes for YOLO
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def get_detection_info(indices, boxes, confidences, class_ids):
    """
    A function that returns information on the detected objects and their
    bounding boxes, class IDs, and confidence values.
    
    Args:
        indices: A list of indices of the detected objects.
        boxes: A list of bounding boxes for the detected objects.
        confidences: A list of confidence values for the detected objects.
        class_ids: A list of class IDs for the detected objects.
        
    Returns:
        detection_info: A list of dictionaries containing information on the
        detected objects.
    """
    detection_info = []
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        detection_info.append({
            "label": label,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "confidence": confidence
        })
    return detection_info

# Set threshold values for object and face detection
conf_threshold = 0.5
nms_threshold = 0.4

# Open input video file
cap = cv2.VideoCapture(r'C:\Users\jyoti\OneDrive\Desktop\sip\Untitled video - Made with Clipchamp.mp4')

# Initialize DataFrame to store detection info
detection_df = pd.DataFrame(columns=["frame_num", "label", "x", "y", "width", "height", "confidence"])

frame_num = 0
while True:
    # Read video frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply object detection using YOLO
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    # Initialize bounding box, class IDs, and confidence values for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # Apply non-maximum suppression to remove redundant bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Get detection information
    detection_info = get_detection_info(indices, boxes, confidences, class_ids)
    
    # Apply face detection using Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        detection_info.append({
            "label": "face",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "confidence": None
        })
    
    # Add detection info to DataFrame
    frame_df = pd.DataFrame(detection_info)
    frame_df.insert(0, "frame_num", frame_num)
    detection_df = pd.concat([detection_df, frame_df], ignore_index=True)
    
    # Display video frame with bounding boxes
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    frame_num += 1

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Plot detection info
fig, ax = plt.subplots(figsize=(10,5))
detection_df.groupby("label").size().plot(kind="bar", ax=ax)
ax.set_xlabel("Object Label")
ax.set_ylabel("Count")
plt.show()

