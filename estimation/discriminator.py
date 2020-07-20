import cv2
import numpy as np
import pyrealsense2 as rs

class Realsense():
    def __init__(self, clip):
        # ストリーム(Depth/Color)の設定
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # ストリーミング開始
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)

        # 距離[m] = depth * depth_scale 
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.clipping_distance_in_meters = clip # meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Alignオブジェクト生成
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def detect(self):
        # フレーム待ち(Color & Depth)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            pass

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        grey_color = 0
        # 3colors array
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image))

        # バイラテラルフィルタ
        delta_distance = 20 
        delta_color = 20
        filter_range = 9
        depth_image_3d = np.array(depth_image_3d, dtype=np.float32)
        depth_image_3d = cv2.bilateralFilter(depth_image_3d, filter_range, delta_color, delta_distance)

        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        bg_grayscale = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)

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

        moment_all = cv2.moments(hand, False)

        flag_right, flag_left = False, False
        gx_right, gy_right = 0, 0
        gx_left, gy_left = 0, 0


        if(moment_all["m00"] != 0):
            gx_all, gy_all = int(moment_all["m10"]/moment_all["m00"]) , int(moment_all["m01"]/moment_all["m00"])

            if(hand[gy_all][gx_all] == 255):
                # 半分でdivideするとキメうち
                if(gx_all < (hand.shape[1] / 2)):
                    flag_right, flag_left = True, False
                    gx_right, gy_right = gx_all, gy_all
                    gx_left, gy_left = 0, 0
                else:
                    flag_right, flag_left = False, True
                    gx_right, gy_right = 0, 0
                    gx_left, gy_left = gx_all, gy_all
            else:
                right_hand = hand[:,gx_all:]
                left_hand = hand[:,:gx_all]

                flag_right, flag_left = True, True
                moment_right = cv2.moments(right_hand, False)
                moment_left = cv2.moments(left_hand, False)

                gx_right, gy_right = gx_all + int(moment_right["m10"]/moment_right["m00"]) , int(moment_right["m01"]/moment_right["m00"])
                gx_left, gy_left = int(moment_left["m10"]/moment_left["m00"]) , int(moment_left["m01"]/moment_left["m00"])

        response = {"status": 200,
                    "result": {"right": {"flag": flag_right,
                                        "x": gx_right,
                                        "y": gy_right
                                        },
                                "left": {"flag": flag_left,
                                        "x": gx_left,
                                        "y": gy_left
                                        }
                                }
                    }

        print(response)

        return response