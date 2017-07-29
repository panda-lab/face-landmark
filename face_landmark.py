import sys
import os

import time
import pprint

import caffe
import dlib
import cv2
import numpy as np

def file_list_fn(path):

    file_list = []
    files = os.listdir(path)
    for f in files:
        file_list.append(f)
    return file_list

net_work_path = '/home/code/face-landmark/model/landmark_deploy.prototxt'
weight_path = '/home/code/face-landmark/model/VanFace.caffemodel'
images_dir = '/home/code/face-landmark/images/'
result_dir = '/home/code/face-landmark/results/'

image_list = file_list_fn(images_dir)
caffe.set_mode_cpu()
net = caffe.Net(net_work_path, weight_path, caffe.TEST)
net.name = 'FaceThink_face_landmark_test'

detector = dlib.get_frontal_face_detector()

total_detecting_time = 0.0
total_landmark_time = 0.0
face_total = 0.0
for image in image_list:
    print("Processing file: {}".format(image))
    img = cv2.imread(images_dir + image)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    det_start_time = time.time()
    dets = detector(img, 1)
    det_end_time = time.time()
    det_time = det_end_time - det_start_time
    total_detecting_time += det_time
    print "Detecting time is {}".format(det_time)
    print "Number of faces detected: {}".format(len(dets))
    for i, d in enumerate(dets):
            print "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom())

    for index, det in enumerate(dets):
        face_total += 1
        x1 = det.left()
        y1 = det.top()
        x2 = det.right()
        y2 = det.bottom()
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img.shape[1]: x2 = img.shape[1]
        if y2 > img.shape[0]: y2 = img.shape[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        roi = img[y1:y2 + 1, x1:x2 + 1, ]
        gary_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        w = 60
        h = 60

        print image
        res = cv2.resize(gary_img, (w, h), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
        resize_mat = np.float32(res)

        m = np.zeros((w, h))
        sd = np.zeros((w, h))
        mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
        new_m = mean[0][0]
        new_sd = std_dev[0][0]
        new_img = (resize_mat - new_m) / (0.000001 + new_sd)

        if new_img.shape[0] != net.blobs['data'].data[0].shape or new_img.shape[1] != net.blobs['data'].data[1].shape:
            print "Incorrect " + image + ", resize to correct dimensions."

        net.blobs['data'].data[...] = new_img
        landmark_time_start = time.time()
        out = net.forward()
        landmark_time_end = time.time()
        landmark_time = landmark_time_end - landmark_time_start
        total_landmark_time += landmark_time
        print "landmark time is {}".format(landmark_time)
        points = net.blobs['Dense3'].data[0].flatten()

        point_pair_l = len(points)
        for i in range(point_pair_l / 2):
            x = points[2*i] * (x2 - x1) + x1
            y = points[2*i+1] * (y2 - y1) + y1
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)

    cv2.imwrite(result_dir + image, img)

print total_detecting_time
print total_landmark_time
print face_total
per_face_det_time = total_detecting_time / face_total
per_face_landmark_time = total_landmark_time / face_total

per_image_det_time = total_detecting_time / len(image_list)
per_image_landmark_time = total_landmark_time / len(image_list)

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print "per face detecting time is {}".format(per_face_det_time)
print "per face landmark time is {}".format(per_face_landmark_time)
print "per image detecting time is {}".format(per_image_det_time)
print "per image detecting time is {}".format(per_image_landmark_time)



