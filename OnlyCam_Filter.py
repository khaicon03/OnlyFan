## compare file
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# unfull ver
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class DualRowFollowerROI:
    def __init__(self):
        rospy.loginfo("Dual Row Follower (ROI-based) started.")

        self.bridge = CvBridge()

        # QUAN TRỌNG: xuất lệnh ra /cmd_vel_row
        self.cmd_pub = rospy.Publisher("/cmd_vel_row", Twist, queue_size=10)

        # ========= PARAMS từ launch =========
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")

        # sử dụng để hiện vị trí biên camera
        self.show_debug  = rospy.get_param("~show_debug", True)

        # tốc độ tiến
        self.forward_speed = rospy.get_param("~forward_speed", 0.25)

        # target theo 10cm
        self.left_target_ratio  = rospy.get_param("~left_target_ratio", 0.30)
        self.right_target_ratio = rospy.get_param("~right_target_ratio", 0.70)
        self.center_ratio       = rospy.get_param("~center_ratio", 0.50)

        # vùng trái/phải loại line giả
        self.left_min_x_ratio  = rospy.get_param("~left_min_x_ratio", 0.05)
        self.left_max_x_ratio  = rospy.get_param("~left_max_x_ratio", 0.45)
        self.right_min_x_ratio = rospy.get_param("~right_min_x_ratio", 0.55)
        self.right_max_x_ratio = rospy.get_param("~right_max_x_ratio", 0.95)

        # gain điều khiển
        self.k_ang = rospy.get_param("~k_ang", 2.0)
        self.max_ang = rospy.get_param("~max_ang", 1.0)

        # Canny + Hough params
        self.canny1 = 60
        self.canny2 = 150
        self.hough_threshold = 40

        # subscriber camera
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

    # =====================================================================
    # PHÁT HIỆN LUỐNG BẰNG HOUHGLINES + ROI NGANG
    # =====================================================================
    def detect_rows(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, self.canny1, self.canny2)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=int(roi.shape[0] * 0.5),
            maxLineGap=25
        )

        left_xs = []
        right_xs = []

        h, w = roi.shape[:2]

        # chuyển convert sang px
        left_min  = int(w * self.left_min_x_ratio)
        left_max  = int(w * self.left_max_x_ratio)
        right_min = int(w * self.right_min_x_ratio)
        right_max = int(w * self.right_max_x_ratio)

        # duyệt tất cả line
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]

                # Góc line phải thẳng (dựng đứng)
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    angle_deg = 90
                else:
                    angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
                if angle_deg < 70:  # bỏ line nằm ngang / xiên
                    continue

                # lấy điểm gần đáy ảnh hơn
                x = x1 if y1 > y2 else x2

                # kiểm tra nó thuộc trái hay phải
                if left_min <= x <= left_max:
                    left_xs.append(x)
                elif right_min <= x <= right_max:
                    right_xs.append(x)

        return left_xs, right_xs, edges, lines

    # =====================================================================
    # CALLBACK: xử lý từng frame
    # =====================================================================
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        h, w, _ = frame.shape
        roi = frame[int(h * 0.5):h, :]

        # detect
        left_xs, right_xs, edges, lines = self.detect_rows(roi)

        # ======== DEBUG HIỂN THỊ ========
        if self.show_debug:
            roi_vis = roi.copy()
            roi_h, roi_w, _ = roi_vis.shape

            # vùng trái/phải
            left_min  = int(roi_w * self.left_min_x_ratio)
            left_max  = int(roi_w * self.left_max_x_ratio)
            right_min = int(roi_w * self.right_min_x_ratio)
            right_max = int(roi_w * self.right_max_x_ratio)

            # vẽ vùng trái/phải
            cv2.rectangle(roi_vis, (left_min, 0), (left_max, roi_h-1), (255, 0, 0), 2)
            cv2.rectangle(roi_vis, (right_min, 0), (right_max, roi_h-1), (0, 255, 0), 2)

            # vẽ line Hough (vật thể)
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    cv2.line(roi_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # vẽ line đã chọn làm luống
            for x in left_xs:
                cv2.line(roi_vis, (x, 0), (x, roi_h-1), (255, 255, 0), 1)
            for x in right_xs:
                cv2.line(roi_vis, (x, 0), (x, roi_h-1), (0, 255, 255), 1)

            cv2.imshow("ROI_with_lines", roi_vis)
            cv2.imshow("Edges", edges)
            cv2.waitKey(1)
        # =================================

        # Điều khiển
        cmd = Twist()
        cmd.linear.x = self.forward_speed

        # ==============================
        # TRƯỜNG HỢP 1: CÓ 2 LUỐNG
        # ==============================
        if len(left_xs) > 0 and len(right_xs) > 0:
            x_left  = max(left_xs)
            x_right = min(right_xs)
            center = (x_left + x_right) / 2.0
            desired = self.center_ratio * w
            error = (desired - center) / float(w)

        # ==============================
        # TRƯỜNG HỢP 2: CHỈ LUỐNG TRÁI
        # ==============================
        elif len(left_xs) > 0:
            x_left = max(left_xs)
            desired = self.left_target_ratio * w
            error = (desired - x_left) / float(w)

        # ==============================
        # TRƯỜNG HỢP 3: CHỈ LUỐNG PHẢI
        # ==============================
        elif len(right_xs) > 0:
            x_right = min(right_xs)
            desired = self.right_target_ratio * w
            error = (desired - x_right) / float(w)

        # ==============================
        # TRƯỜNG HỢP 4: KHÔNG THẤY LUỐNG
        # ==============================
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # tính góc quay
        ang = self.k_ang * error
        ang = max(-self.max_ang, min(self.max_ang, ang))
        cmd.angular.z = ang

        self.cmd_pub.publish(cmd)


# ========================================================================
if __name__ == "__main__":
    rospy.init_node("row_follow_dual_roi")
    node = DualRowFollowerROI()
    rospy.spin()
