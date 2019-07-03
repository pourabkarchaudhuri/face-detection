#!/usr/bin/env python

from threading import Thread, Lock
import cv2
# from CountsPerSec import CountsPerSec
# from multiprocessing.pool import ThreadPool
import threading
import time, os
# import requests
count = 0
temp =0
# frame1 = None

class WebcamVideoStream :
    
    def __init__(self, src = 0) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self, face_cascade) :
        global temp, frame1
        temp = temp+1
        # print (count)
        # pool = ThreadPool(processes=1)
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        thread1 = threading.Thread(target=self.detect, args=(frame, face_cascade))
        thread1.start()
        thread1.join()
        return frame
        # if (temp%15 == 0) :
            
        #     # print (frame1)
        #     frame1 = frame
        # if (temp <=14) :
        #     return frame
        # else:
        #     return frame1

    def detect(self,frame, face_cascade) :
        # print ("detect")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        for (x,y,w,h) in faces: 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2) 
            roi_gray = gray[y:y+h, x:x+w] 
            roi_color = frame[y:y+h, x:x+w]

    def stop(self) :
        self.started = False
        self.thread.join()
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()


def main():

    print("Starting ")
    vs = WebcamVideoStream().start()
    fps = 0
    frame_num = 0
    # cps = CountsPerSec().start()
    classifier_path = os.path.join(os.getcwd(), '../haarcascade/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(classifier_path)
    
    global frame1 
    # eventlet.monkey_patch()
    while True :
        # count = count+1
        # if (count%50 == 0) :
        #     requests.post('http://127.0.0.1:5000/post')
            # if response.status_code == 200:
            # print('Success!')
            # elif response.status_code == 404:
            #     print('Not Found.')
        start_time = time.time()

        # print("Start time: ", start_time)
        frame_num = frame_num + 1

        frame = vs.read(face_cascade)
        # print('Frame: {0} Processed'.format(frame_num))
        end_time = time.time()
        # print("End time: ", end_time)
        fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
        start_time = end_time
        frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
        # frame = cv2.putText(frame, "{:.0f} iterations/sec".format(cps.countsPerSec()),
        #     (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("webcam", frame)
        # cps.increment()
        # print(threading.enumerate())
        if cv2.waitKey(1) == ord("q") :
            break
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
   main()