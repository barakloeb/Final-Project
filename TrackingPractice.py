# bbox = cv2.selectROI("TelloTracking" , img , False)
# tracker.init(img, bbox)
import time
import cv2

cap = cv2.VideoCapture(0)

tracker = cv2.TrackerMOSSE_create()

face_cascade = cv2.CascadeClassifier('C:\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# My Vars:
found = False
tracking = False
lastBbox = (0, 0, 0, 0)
imgCount = 0
success = False

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


while True:
    imgSuccess, img = cap.read()
    imgCount = imgCount + 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)  # DETECT THE FACES
    success, bbox = tracker.update(img)

    if len(faces) == 0:
        imgCount = 0
        if found:
            found = False
            tracking = True
            if success:
                drawBox(img, bbox)
            else:
                print('Failed to track')

        if tracking and not found:
            if success:
                drawBox(img, bbox)
            else:
                print('Failed to track')
                found = False
                tracking = False
                lastBbox = (0, 0, 0, 0)
                imgCount = 0
                success = False
                tracker = cv2.TrackerMOSSE_create()



    else:
        x, y, w, h = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
        #    for (x, y, w, h) in faces:
        lastBbox = (x, y, w, h)
        if imgCount > 5:
            tracker.init(img, lastBbox)



        cv2.putText(img, str(len(faces)), (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        found = True
        tracking = False
        roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
        roi_color = img[y:y + h, x:x + w]
        # setting Face Box properties
        fbCol = (255, 0, 0)  # BGR 0-255
        fbStroke = 2
        # end coords are the end of the bounding box x & y
        end_cord_x = x + w
        end_cord_y = y + h
        end_size = w * 2
        # Draw the face bounding box
        cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), fbCol, fbStroke)

    timer = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    time.sleep(1 / fps)
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("TrackKing", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

#  the old code: ////////////////////////////////////
# while True:
#     timer = cv2.getTickCount()
#     success, img = cap.read()
#
#     success, bbox = tracker.update(img)
#
#     if success:
#         drawBox(img, bbox)
#     else:
#         cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
#     cv2.putText(img, str(int(fps)) , (75,50) , cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
#     cv2.imshow("TrackKing", img)
#
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
