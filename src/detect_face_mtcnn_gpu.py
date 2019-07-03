#!/usr/bin/env python2
import cv2
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
import mtcnn
import time

def main():
    fps = 0
    frame_num = 0
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)
    video_capture.set(4, 480)

    minsize = 25 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)
        while(True):
            start_time = time.time()
            ret, frame = video_capture.read()
            frame_num = frame_num + 1
            if not ret:
                break
            # Display the resulting frame
            img = frame[:,:,0:3]
            boxes, _ = mtcnn.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            print(boxes)
            for i in range(boxes.shape[0]):
                pt1 = (int(boxes[i][0]), int(boxes[i][1]))
                pt2 = (int(boxes[i][2]), int(boxes[i][3]))
                
                cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))
            # frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)

            end_time = time.time()
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time
            frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()