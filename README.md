﻿# Object and Face Detection with YOLO and Haar Cascade

This Python script utilizes the OpenCV library to perform object detection using the YOLO (You Only Look Once) deep learning model and face detection using the Haar Cascade classifier. The detected objects are visualized in a video stream, and the information about the detected people, including their bounding boxes, is stored and later analyzed.

## Requirements
- OpenCV (cv2)
- NumPy
- Matplotlib

Make sure to have the YOLO model weights file ("yolov3.weights"), configuration file ("yolov3.cfg"), and class names file ("coco.names") in the same directory as the script. Additionally, the Haar Cascade face detector XML file ("haarcascade_frontalface_default.xml") should be present.

## Instructions
1. Install the required libraries: `pip install opencv-python numpy matplotlib`
2. Download the YOLO files (weights, configuration, class names) and Haar Cascade XML file.
3. Update the file paths in the script accordingly.
4. Download the corresponding YOLO weights from (in `https://pjreddie.com/darknet/yolo/#google_vignette`)

## How to Run
Execute the script, and it will open a video stream. The script performs YOLO object detection to identify persons and then applies Haar Cascade face detection on those regions. Detected faces are blurred, and information about the detected people is printed and stored.

Press 'q' to quit the video stream.

## Output
The script prints information about detected people, including their labels, coordinates, dimensions, and confidence scores, in real-time. After execution, the script also generates a matplotlib plot displaying the confidence scores and the x, y, width, and height values of the detected people.

## Additional Notes
- Adjust the file paths if the YOLO and Haar Cascade files are stored in a different location.
- Ensure the video file path (in `cv2.VideoCapture()`) is correct or use a different video source.

Feel free to modify the script based on your specific use case or integrate it into a larger project.# Face_blurrring_video
 
