# -*- coding: utf-8 -*-

#############################################
##      D415 Depth画像の表示
#############################################
import pyrealsense2 as rs
import numpy as np
import cv2
import dlib

# ストリーム(Depth/Color)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

#   距離[m] = depth * depth_scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 0.5  # meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

# OpenCVのカスケードファイル

CASCADE_PATH = "../haarcascades/"
CASCADE = cv2.CascadeClassifier(
    CASCADE_PATH + "haarcascade_frontalface_default.xml")

LEARNED_MODEL_PATH = "../learned-models/"
PREDICTOR = dlib.shape_predictor(
    LEARNED_MODEL_PATH + "shape_predictor_68_face_landmarks.dat")

# dlib detector
detector = dlib.get_frontal_face_detector()

# 顔の位置を検出　返却値は位置を表すリスト(x,y,w,h)
def face_position(gray_img):
    faces = CASCADE.detectMultiScale(gray_img, minSize=(100, 100))
    return faces

# ランドマーク検出

def facemark(gray_img):
    global detector
    rects = detector(gray_img, 1)

    landmarks = []

    for rect in rects:
        landmarks.append(
        np.array([[p.x, p.y] for p in PREDICTOR(gray_img, rect).parts()]))

    return landmarks


try:
    while True:
        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        landmarks = facemark(gray)

        print(landmarks)



        # Depth画像前処理(1m以内を画像化)
        grey_color = 0
        # 3colors array
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))

        # バイラテラルフィルタ
        delta_distance = 20
        delta_color = 20
        filter_range = 9
        depth_image_3d = np.array(depth_image_3d, dtype=np.float32)
        depth_image_3d = cv2.bilateralFilter(
            depth_image_3d, filter_range, delta_color, delta_distance)

        bg_removed = np.where((depth_image_3d > clipping_distance) | (
            depth_image_3d <= 0), grey_color, color_image)

        depth_max = 100
        depth_min = 90
        bg_grayscale = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
        """
        canny edge　やらない。
        #bg_edge = cv2.Canny(bg_grayscale, depth_min, depth_max)

        neiborhood8 = np.array([[255, 255, 255],
                                [255, 255, 255],
                                [255, 255, 255]],
                                np.uint8)

        #bg_edge_close = cv2.morphologyEx(bg_edge, cv2.MORPH_CLOSE, neiborhood8)
        #bg_edge_close = cv2.dilate(bg_edge, neiborhood8, 1)
        """

        #opencvの認識できるエッジの形に変換
        ret, thresh = cv2.threshold(bg_grayscale, 10, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # マスク作成, マスク
        blank = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))

        for i in range(len(contours)):
            approx = cv2.approxPolyDP(
                contours[i], 0.001*cv2.arcLength(contours[i], True), True)
            cv2.drawContours(blank, [approx], 0, (255, 255, 255), -1)

        blank = np.float32(blank)
        blank_grayscale = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        ret, hand = cv2.threshold(blank_grayscale, 127, 255, 0)

        moment_all = cv2.moments(hand, False)

        if(moment_all["m00"] != 0):
            gx_all, gy_all = int(
                moment_all["m10"]/moment_all["m00"]), int(moment_all["m01"]/moment_all["m00"])

            cv2.circle(bg_removed, (gx_all, gy_all), 15,
                       (255, 255, 255), thickness=-1)

            if(hand[gy_all][gx_all] == 255):
                gx, gy = gx_all, gy_all
            else:
                right_hand = hand[:, gx_all:]
                left_hand = hand[:, :gx_all]

                moment_right = cv2.moments(right_hand, False)
                moment_left = cv2.moments(left_hand, False)

                gx_right, gy_right = int(
                    moment_right["m10"]/moment_right["m00"]), int(moment_right["m01"]/moment_right["m00"])
                gx_left, gy_left = int(
                    moment_left["m10"]/moment_left["m00"]), int(moment_left["m01"]/moment_left["m00"])

                cv2.circle(bg_removed, (gx_all + gx_right, gy_right),
                           15, (255, 0, 0), thickness=-1)
                cv2.circle(bg_removed, (gx_left, gy_left),
                           15, (0, 255, 0), thickness=-1)

        depth_image = depth_image * (hand / np.max(hand))

        # depth map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # レンダリング
        images = np.hstack((bg_removed, depth_colormap))
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xff == 27:
            break

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()
