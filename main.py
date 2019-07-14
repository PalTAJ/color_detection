
import cv2
import knn_classifier
import featureExtractor

import numpy as np


featureExtractor.main(1) # extracting features from dataset and creates our training set.

## uncomment any part and add the suitable paths for images or video.



### Single Image Testing

# test_image = 'greentest.jpg'
# source_image = cv2.imread(test_image)

# featureExtractor.main(2,test_image) # extract feature for our test image
# prediction = knn_classifier.main('features.data', 'test.data')
# cv2.putText(source_image, prediction,(15, 45),cv2.FONT_HERSHEY_PLAIN , 3,200, 	thickness = 3)
# cv2.imshow('color ', source_image)
# cv2.waitKey(0)
#

### Video Testing

source_video = "video.mp4"

cap = cv2.VideoCapture(source_video)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite("frame.jpg", frame)


    featureExtractor.main(2,"frame.jpg") # extract feature for our test image
    prediction = knn_classifier.main('features.data', 'test.data')
    cv2.putText(frame, prediction,(15, 45),cv2.FONT_HERSHEY_PLAIN , 3,200, 	thickness = 3)

    cv2.imshow('frame',frame)

    if cv2.waitKey(120) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


