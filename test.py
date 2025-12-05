#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# YOLOv8 (ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class DualRowFollowerROI(object):
    def __init__(self):
        # === Node name ===
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Starting..." % self.node_name)

        # --- ROS Params ---
        # Topic
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.cmd_topic   = rospy.get_param("~cmd_topic",   "/cmd_vel_row")

        # Vận tốc
        self.forward_speed = rospy.get_param("~forward_speed", 0.25)

        # ROI theo chiều Y (tỉ lệ chiều cao ảnh)
        # ví dụ: chỉ lấy vùng 0.5–1.0 (nửa dưới ảnh)
        self.roi_y_start_ratio = rospy.get_param("~roi_y_start_ratio", 0.5)
        self.roi_y_end_ratio   = rospy.get_param("~roi_y_end_ratio",   1.0)

        # ROI trái/phải theo tỉ lệ X
        self.left_min_x_ratio  = rospy.get_param("~left_min_x_ratio",  0.05)
        self.left_max_x_ratio  = rospy.get_param("~left_max_x_ratio",  0.45)
        self.right_min_x_ratio = rospy.get_param("~right_min_x_ratio", 0.55)
        self.right_max_x_ratio = rospy.get_param("~right_max_x_ratio", 0.95)

        # gain điều khiển
        self.k_ang   = rospy.get_param("~k_ang", 2.0)
        self.max_ang = rospy.get_param("~max_ang", 1.0)

        # Canny + Hough params
        self.canny1          = rospy.get_param("~canny1", 50)
        self.canny2          = rospy.get_param("~canny2", 150)
        self.hough_threshold = rospy.get_param("~hough_threshold", 40)
        self.min_line_length = rospy.get_param("~min_line_length", 30)
        self.max_line_gap    = rospy.get_param("~max_line_gap", 15)

        # NGƯỠNG GÓC: chỉ giữ các line có góc so với trục NẰM (x) >= min_angle_deg
        self.min_angle_deg = max(35.0, rospy.get_param("~min_angle_deg", 35.0))

        # DEBUG hiển thị ảnh
        self.show_debug = rospy.get_param("~show_debug", True)

        # ============ YOLO PARAMS ============
        self.use_yolo = rospy.get_param("~use_yolo", False)
        self.yolo_model_path = rospy.get_param("~yolo_model_path", "")
        self.yolo_conf       = rospy.get_param("~yolo_conf", 0.5)
        self.yolo_iou        = rospy.get_param("~yolo_iou", 0.45)
        # danh sách class id cần quan tâm, vd [0] = person, hoặc [0,1,...]
        self.yolo_target_classes = rospy.get_param("~yolo_target_classes", [])
        # nếu phát hiện class mục tiêu ở giữa → dừng
        self.stop_on_detection = rospy.get_param("~stop_on_detection", True)

        self.yolo_model = None
        self._init_yolo()

        # --- Bridge + ROS I/O ---
        self.bridge  = CvBridge()
        self.image_sub = rospy.Subscriber(self.image_topic, Image,
        self.image_callback, queue_size=1)
        self.cmd_pub   = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        self.img_width  = None
        self.img_height = None

        rospy.loginfo("[%s] Initialized. Subscribing to %s, publishing Twist to %s"
                      % (self.node_name, self.image_topic, self.cmd_topic))

    # ================== YOLO ==================
    def _init_yolo(self):
        """Khởi tạo model YOLO nếu được bật và có ultralytics."""
        if not self.use_yolo:
            rospy.loginfo("[%s] YOLO disabled by parameter ~use_yolo." %
                          self.node_name)
            return

        if YOLO is None:
            rospy.logwarn("[%s] ultralytics YOLO not installed. "
                          "YOLO will be disabled." % self.node_name)
            self.use_yolo = False
            return

        if not self.yolo_model_path:
            rospy.logwarn("[%s] ~yolo_model_path is empty. "
                          "YOLO will be disabled." % self.node_name)
            self.use_yolo = False
            return

        if not os.path.exists(self.yolo_model_path):
            rospy.logwarn("[%s] YOLO model path does not exist: %s. "
                          "YOLO will be disabled." %
                          (self.node_name, self.yolo_model_path))
            self.use_yolo = False
            return

        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            rospy.loginfo("[%s] Loaded YOLO model from: %s"
                          % (self.node_name, self.yolo_model_path))
        except Exception as e:
            rospy.logerr("[%s] Failed to load YOLO model: %s" %
                         (self.node_name, str(e)))
            self.use_yolo = False
            self.yolo_model = None

    def run_yolo(self, bgr_image):
        """Chạy YOLO, trả về True nếu có phát hiện class mục tiêu trong vùng giữa."""
        if not (self.use_yolo and self.yolo_model is not None):
            return False

        try:
            # Ultralytics nhận BGR hoặc RGB đều được, nhưng mặc định là BGR
            results = self.yolo_model.predict(
                source=bgr_image,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                verbose=False
            )
        except Exception as e:
            rospy.logerr_throttle(2.0,
                "[%s] YOLO inference error: %s" % (self.node_name, str(e)))
            return False

        if not results:
            return False

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return False

        h, w, _ = bgr_image.shape
        mid_x_min = w * 0.3
        mid_x_max = w * 0.7

        detected_in_center = False

        for box in res.boxes:
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            conf   = float(box.conf.item()) if box.conf is not None else 0.0
            if self.yolo_target_classes and cls_id not in self.yolo_target_classes:
                continue
            if conf < self.yolo_conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = 0.5 * (x1 + x2)

            if mid_x_min <= cx <= mid_x_max:
                detected_in_center = True
                if self.show_debug:
                    cv2.rectangle(bgr_image,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    label = "cls:%d %.2f" % (cls_id, conf)
                    cv2.putText(bgr_image, label,
                                (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

        return detected_in_center

    # ================== IMAGE CALLBACK ==================
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr_throttle(2.0,
                "[%s] cv_bridge error: %s" % (self.node_name, str(e)))
            return

        if self.img_width is None or self.img_height is None:
            self.img_height, self.img_width = frame.shape[:2]
            rospy.loginfo("[%s] First image size: %dx%d" %
                          (self.node_name, self.img_width, self.img_height))

        cmd, debug_img = self.process_frame(frame)

        # Publish cmd_vel_row
        self.cmd_pub.publish(cmd)

        # Hiển thị debug nếu cần
        if self.show_debug and debug_img is not None:
            cv2.imshow("row_follow_debug", debug_img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                rospy.signal_shutdown("ESC pressed")

    # ================== ROW FOLLOW LOGIC ==================
    def process_frame(self, frame):
        """Xử lý 1 frame: bám luống + YOLO (nếu có)."""
        h, w, _ = frame.shape
        debug_img = frame.copy()

        # ----- Cắt ROI theo chiều Y -----
        y_start = int(self.roi_y_start_ratio * h)
        y_end   = int(self.roi_y_end_ratio   * h)
        y_start = max(0, min(h-1, y_start))
        y_end   = max(0, min(h,   y_end))

        roi = frame[y_start:y_end, :, :]

        # Chuyển xám + blur + Canny
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny1, self.canny2)

        # HoughLinesP: tìm line
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi/180,
                                threshold=self.hough_threshold,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)

        left_points = []
        right_points = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                dy = (y2 - y1)
                dx = (x2 - x1)
                if dx == 0:
                    angle_deg = 90.0
                else:
                    angle_rad = np.arctan2(dy, dx)
                    angle_deg = np.degrees(angle_rad)

                # chỉ lấy line tương đối dọc (so với trục x)
                if abs(angle_deg) < self.min_angle_deg:
                    continue

                # midpoint của line trong ROI
                mx = 0.5 * (x1 + x2)
                my = 0.5 * (y1 + y2)

                # vẽ line lên debug
                cv2.line(debug_img[y_start:y_end, :, :],
                         (x1, y1), (x2, y2), (255, 0, 0), 2)

                # phân loại left/right theo vị trí mx
                x_ratio = mx / float(w)
                if self.left_min_x_ratio <= x_ratio <= self.left_max_x_ratio:
                    left_points.append((mx, my))
                elif self.right_min_x_ratio <= x_ratio <= self.right_max_x_ratio:
                    right_points.append((mx, my))

        # Tính center của luống
        lane_center_x = None
        if left_points and right_points:
            left_mean_x  = np.mean([p[0] for p in left_points])
            right_mean_x = np.mean([p[0] for p in right_points])
            lane_center_x = 0.5 * (left_mean_x + right_mean_x)

            cv2.circle(debug_img[y_start:y_end, :, :],
                       (int(left_mean_x),  int(np.mean([p[1] for p in left_points]))),
                       5, (0, 0, 255), -1)
            cv2.circle(debug_img[y_start:y_end, :, :],
                       (int(right_mean_x), int(np.mean([p[1] for p in right_points]))),
                       5, (0, 255, 255), -1)
        elif left_points:
            lane_center_x = np.mean([p[0] for p in left_points])
            cv2.circle(debug_img[y_start:y_end, :, :],
                       (int(lane_center_x), int(np.mean([p[1] for p in left_points]))),
                       5, (0, 0, 255), -1)
        elif right_points:
            lane_center_x = np.mean([p[0] for p in right_points])
            cv2.circle(debug_img[y_start:y_end, :, :],
                       (int(lane_center_x), int(np.mean([p[1] for p in right_points]))),
                       5, (0, 255, 255), -1)

        img_center_x = w / 2.0
        cmd = Twist()
        cmd.linear.x = self.forward_speed

        if lane_center_x is None:
            # Không thấy luống → dừng lại
            rospy.logwarn_throttle(1.0,
                "[%s] No lane detected, stopping." % self.node_name)
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # error > 0: luống lệch trái, cần quay trái → angular.z > 0
            error = (img_center_x - lane_center_x) / (w / 2.0)
            ang   = self.k_ang * error
            ang   = max(-self.max_ang, min(self.max_ang, ang))
            cmd.angular.z = ang

            # Vẽ center và target trên debug
            cv2.line(debug_img,
                     (int(img_center_x), 0),
                     (int(img_center_x), h),
                     (0, 255, 0), 1)
            cv2.circle(debug_img,
                       (int(lane_center_x), int(0.5*(y_start + y_end))),
                       6, (0, 0, 255), -1)

        # ============ YOLO check ============
        if self.stop_on_detection:
            has_target = self.run_yolo(debug_img if self.show_debug else frame)
            if has_target:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                rospy.loginfo_throttle(1.0,
                    "[%s] YOLO detected target in front -> STOP." % self.node_name)

        return cmd, debug_img


if __name__ == "__main__":
    rospy.init_node("row_follow_dual_roi")
    node = DualRowFollowerROI()
    rospy.spin()
    cv2.destroyAllWindows()
