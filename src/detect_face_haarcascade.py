import cv2
import sys
import time
import os
# cascPath = sys.argv[1]

def main():
    print('Loading HaarCascade Classifier:')
    # print(os.path.join(os.getcwd(), '../haarcascade/haarcascade_frontalface_default.xml'))
    classifier_path = os.path.join(os.getcwd(), '../haarcascade/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(classifier_path)

    fps = 0
    frame_num = 0

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        start_time = time.time()
        ret, frame = video_capture.read()
        frame_num = frame_num + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),)

    # Draw a rectangle around the faces

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        end_time = time.time()
        fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
        start_time = end_time
        frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
        cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # frame = cv2.putText(frame, "{:.0f} iterations/sec".format(cps.countsPerSec()), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    # Display the resulting frame

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything is done, release the capture

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()