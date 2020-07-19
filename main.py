# -*- coding: utf-8 -*-

#############################################
##      D415 Depth画像の表示
#############################################
import pyrealsense2 as rs
import numpy as np
import cv2

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
clipping_distance_in_meters = 0.5 # meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

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

        # Depth画像前処理(1m以内を画像化)  
        grey_color = 0
        # 3colors array
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image))

        # バイラテラルフィルタ
        delta_distance = 20 
        delta_color = 20
        filter_range = 9
        depth_image_3d = np.array(depth_image_3d, dtype=np.float32)
        depth_image_3d = cv2.bilateralFilter(depth_image_3d, filter_range, delta_color, delta_distance)

        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        #canny edge　やらない。
        depth_max = 100
        depth_min = 90
        bg_grayscale = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
        #bg_edge = cv2.Canny(bg_grayscale, depth_min, depth_max)

        neiborhood8 = np.array([[255, 255, 255],
                                [255, 255, 255],
                                [255, 255, 255]],
                                np.uint8)

        #bg_edge_close = cv2.morphologyEx(bg_edge, cv2.MORPH_CLOSE, neiborhood8)
        #bg_edge_close = cv2.dilate(bg_edge, neiborhood8, 1)
        
        #opencvの認識できるエッジの形に変換
        ret,thresh = cv2.threshold(bg_grayscale,10,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        
        # マスク作成, マスク
        blank = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))

        for i in range(len(contours)):
            approx = cv2.approxPolyDP(contours[i], 0.001*cv2.arcLength(contours[i], True), True)
            cv2.drawContours(blank, [approx], 0, (255, 255, 255), -1)
        
        blank = np.float32(blank)
        blank_grayscale = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        ret, hand = cv2.threshold(blank_grayscale,127,255,0)

        depth_image = depth_image * (hand / np.max(hand))

        # depth map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # レンダリング
        images = np.hstack((bg_removed, depth_colormap))
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) & 0xff == 27:
            break

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()