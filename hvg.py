import cv2

# Initialize the HOG descriptor
hog = cv2.HOGDescriptor()

# Initialize the SVM classifier
svm = cv2.ml.SVM_create()

# Set the SVM parameters
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

# Load the video file
cap = cv2.VideoCapture('video.mp4')

# Loop through each frame in the video
while cap.isOpened():
    # Read the frame from the video
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the HOG descriptors for the frame
        descriptors = hog.compute(gray)

        # Use the SVM classifier to predict the presence of faces
        prediction = svm.predict(descriptors)[1]

        # Draw a rectangle around the detected faces
        for i in range(prediction.shape[0]):
            if prediction[i] == 1:
                x, y, w, h = hog.detectMultiScale(gray[i:i+64, j:j+128])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with the detected faces
        cv2.imshow('frame', frame)

        # Wait for the 'q' key to be pressed to quit the program
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()