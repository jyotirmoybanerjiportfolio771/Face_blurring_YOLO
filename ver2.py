import cv2
import numpy as np

# Load YOLO object detector and Haar Cascade face detector
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Storage Variable for Information
detected_information = []
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
cap = cv2.VideoCapture("link for your source video file")

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
    
    # Draw bounding boxes around detected objects and apply face detection
    if len(indices) > 0:
        for i in indices.flatten():
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
                    if face.shape[0] > 0 and face.shape[1] > 0:
                        face = cv2.GaussianBlur(face, (23, 23), 30)
                    frame[y_face:y_face+h_face, x_face:x_face+w_face] = face
                
                # Get detection info
                detection_info = get_detection_info(indices, boxes, confidences, class_ids)
                # Print detection info
                print(detection_info)
                detected_information.append(detection_info)
                
    # Show video frame
    cv2.imshow("Video",frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
     break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()

#printing the array
print("\n\n",detected_information)


import matplotlib.pyplot as plt

# Set dark mode
plt.style.use('dark_background')

# Plot subplots for confidences and x y width and height
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot confidences
confidences = [d['confidence'] for detection in detected_information for d in detection if d['label'] == 'person']
ax1.plot(confidences)
ax1.set_title('Confidences')

# Plot x y width and height
x_values = [d['x'] for detection in detected_information for d in detection if d['label'] == 'person']
y_values = [d['y'] for detection in detected_information for d in detection if d['label'] == 'person']
width_values = [d['width'] for detection in detected_information for d in detection if d['label'] == 'person']
height_values = [d['height'] for detection in detected_information for d in detection if d['label'] == 'person']

ax2.plot(x_values, label='x')
ax2.plot(y_values, label='y')
ax2.plot(width_values, label='width')
ax2.plot(height_values, label='height')
ax2.legend()
ax2.set_title('X, Y, Width, Height')

# Show plot
plt.show()

