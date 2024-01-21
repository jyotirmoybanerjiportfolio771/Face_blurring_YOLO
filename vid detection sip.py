import cv2
import numpy as np

# Load YOLO object detector and Haar Cascade face detector
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define classes for YOLO
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set threshold values for object and face detection
conf_threshold = 0.5
nms_threshold = 0.4

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read video frame
    ret, frame = cap.read()
    
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
    
    # Draw bounding boxes around detected objects and apply face detection
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        if label == 'person':
            # Draw bounding box around person
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Apply face detection using Haar Cascade
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            for (x_face, y_face, w_face, h_face) in faces:
                # Blur detected face
                face = frame[y_face:y_face+h_face, x_face:x_face+w_face]
                face = cv2.GaussianBlur(face, (101, 101), 0)
                frame[y_face:y_face+h_face, x_face:x_face+w_face] = face

    # Show video frame
    cv2.imshow("Video", frame)
    
      # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()

   
