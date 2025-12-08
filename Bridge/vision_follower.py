#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import os
import time
from geometry_msgs.msg import Twist

# ========== CẤU HÌNH MÀU SẮC (HSV) ==========
LOWER_HEALTHY = np.array([35, 80, 80], dtype=np.uint8)
UPPER_HEALTHY = np.array([85, 255, 255], dtype=np.uint8)
LOWER_SICK = np.array([15, 40, 40], dtype=np.uint8)
UPPER_SICK = np.array([35, 255, 255], dtype=np.uint8)

class DualRowFollowerDebug:
    def __init__(self):
        rospy.init_node("row_follow_parallel")
        rospy.loginfo("--- PARALLEL WALL FOLLOWING MODE ---")

        self.cmd_pub = rospy.Publisher("/cmd_vel_row", Twist, queue_size=1)

        # ========= PARAMS TỐC ĐỘ =========
        self.forward_speed = rospy.get_param("~forward_speed", 0.20) # Chạy chậm để bám chính xác
        
        # ========= PARAMS VỊ TRÍ (KHOẢNG CÁCH) =========
        # Vị trí mong muốn của đường kẻ trên màn hình (0.0 -> 1.0)
        # Nếu bám luống trái, đường kẻ nên nằm ở khoảng 0.2-0.3 bên trái
        # Nếu bám luống phải, đường kẻ nên nằm ở khoảng 0.7-0.8 bên phải
        self.left_target_ratio  = rospy.get_param("~left_target_ratio", 0.25)
        self.right_target_ratio = rospy.get_param("~right_target_ratio", 0.75)
        
        # Hệ số điều khiển khoảng cách (P - Distance)
        self.k_dist = rospy.get_param("~k_dist", 1.5)

        # ========= PARAMS GÓC (SONG SONG) =========
        # Hệ số điều khiển góc (P - Angle) -> Quan trọng để chạy song song
        # Tăng lên nếu robot phản ứng chậm với hướng nghiêng
        self.k_theta = rospy.get_param("~k_theta", 0.04)
        
        # Góc mục tiêu trên camera (Độ).
        # 90 độ là đường thẳng đứng (song song tuyệt đối theo phối cảnh 2D)
        self.target_angle_deg = 90.0

        self.max_ang = rospy.get_param("~max_ang", 1.2)
        
        # Cấu hình vùng tìm kiếm (ROI X)
        self.left_roi_limit  = (0.0, 0.5) # Tìm bên trái màn hình
        self.right_roi_limit = (0.5, 1.0) # Tìm bên phải màn hình
        
        # Cấu hình Canny/Hough
        self.canny1 = 50
        self.canny2 = 150
        self.hough_threshold = 50

        # ========= YOLO CONFIG (Giữ nguyên) =========
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "custom_yolov4_tiny_best.weights")
        cfg_path = os.path.join(script_dir, "custom_yolov4_tiny.cfg")
        names_path = os.path.join(script_dir, "coco.names")

        self.frame_count = 0
        self.skip_yolo_frames = 2
        self.last_detections = []
        self.yolo_input_size = (320, 320)
        self.plant_tracks = []
        self.prev_time = 0

        # ... (Phần Init YOLO và Camera giữ nguyên như cũ để tiết kiệm chỗ) ...
        # (Bạn copy lại đoạn try-catch load YOLO và Camera ở code cũ vào đây)
        # -------------------------------------------------------------------
        self.yolo_ready = False # Giả lập nếu chưa copy code YOLO
        try:
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            with open(names_path, "r") as f: self.classes = [l.strip() for l in f.readlines()]
            self.layer_names = self.net.getLayerNames()
            out_layers_indices = self.net.getUnconnectedOutLayers()
            if len(out_layers_indices.shape) > 1: out_layers_indices = out_layers_indices.flatten()
            self.output_layers = [self.layer_names[i - 1] for i in out_layers_indices]
            self.yolo_ready = True
        except: pass

        self.gst_str = ("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink")
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)

    # Support functions (bbox_iou, analyze_plant_color, detect_objects) giữ nguyên
    def bbox_iou(self, b1, b2):
        x1, y1, x2, y2 = b1; x1b, y1b, x2b, y2b = b2
        inter = max(0, min(x2, x2b) - max(x1, x1b)) * max(0, min(y2, y2b) - max(y1, y1b))
        return inter / ((max(0, x2-x1)*max(0, y2-y1)) + (max(0, x2b-x1b)*max(0, y2b-y1b)) - inter + 1e-6)

    def analyze_plant_color(self, frame, bbox): return "healthy" # Rút gọn cho demo
    def detect_objects(self, frame): return [] # Rút gọn cho demo

    # =====================================================================
    # HÀM DETECT MỚI: TRẢ VỀ GÓC THỰC TẾ (0-180 độ)
    # =====================================================================
    def detect_lines_and_angles(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, self.canny1, self.canny2)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold, int(roi.shape[0]*0.3), 30)
        
        h, w = roi.shape[:2]
        
        left_lines_x = []
        left_lines_ang = []
        right_lines_x = []
        right_lines_ang = []
        
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                # Tính góc thực tế theo trục toạ độ ảnh (gốc trên-trái)
                # dx = x2 - x1, dy = y2 - y1.
                # Lưu ý: y tăng dần đi xuống.
                angle_rad = np.arctan2(y1 - y2, x2 - x1) # Đảo y để tính theo toạ độ Descartes chuẩn
                angle_deg = np.degrees(angle_rad)
                
                # Chuẩn hóa về 0 - 180 độ
                if angle_deg < 0: angle_deg += 180
                
                # Lọc góc: Bỏ các đường nằm ngang (0-20 độ hoặc 160-180 độ) vì đó là nhiễu ngang
                if 20 < angle_deg < 160:
                    x_center = (x1 + x2) / 2
                    
                    # Phân loại trái phải dựa trên vị trí X
                    if self.left_roi_limit[0]*w < x_center < self.left_roi_limit[1]*w:
                        left_lines_x.append(x_center)
                        left_lines_ang.append(angle_deg)
                    elif self.right_roi_limit[0]*w < x_center < self.right_roi_limit[1]*w:
                        right_lines_x.append(x_center)
                        right_lines_ang.append(angle_deg)
                        
        return left_lines_x, left_lines_ang, right_lines_x, right_lines_ang, edges

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: rate.sleep(); continue

            h, w = frame.shape[:2]
            
            # --- CẮT ROI 2/3 MÀN HÌNH ---
            roi_y_start = int(h / 3)
            roi = frame[roi_y_start:h, :]
            
            # Detect
            l_xs, l_angs, r_xs, r_angs, edges = self.detect_lines_and_angles(roi)
            cv2.imshow("Debug Edges", edges)

            # --- LOGIC ĐIỀU KHIỂN SONG SONG ---
            cmd = Twist()
            cmd.linear.x = self.forward_speed
            
            error_dist = 0.0
            error_angle = 0.0
            
            has_left = len(l_xs) > 0
            has_right = len(r_xs) > 0
            
            # Ưu tiên: Nếu thấy cả 2, đi giữa. Nếu chỉ thấy 1, bám theo cái đó.
            
            detected_x = -1
            detected_ang = -1
            mode = "None"

            if has_left and has_right:
                mode = "Dual (Center)"
                # Tính trung bình vị trí và góc của 2 bên
                avg_l_x = np.mean(l_xs)
                avg_r_x = np.mean(r_xs)
                detected_x = (avg_l_x + avg_r_x) / 2.0
                target_x = w * 0.5 # Giữa màn hình
                
                # Góc: Lấy trung bình góc.
                # Bên trái thường nghiêng /, bên phải nghiêng \
                # Chúng ta muốn robot thẳng, tức là góc trung bình ~ 90 độ
                detected_ang = (np.mean(l_angs) + np.mean(r_angs)) / 2.0
                
            elif has_left:
                mode = "Follow LEFT Wall"
                detected_x = np.mean(l_xs)
                target_x = w * self.left_target_ratio
                detected_ang = np.mean(l_angs)
                
            elif has_right:
                mode = "Follow RIGHT Wall"
                detected_x = np.mean(r_xs)
                target_x = w * self.right_target_ratio
                detected_ang = np.mean(r_angs)
                
            else:
                mode = "Lost Lane"
                cmd.linear.x = 0.0 # Dừng nếu mất đường
            
            if mode != "Lost Lane":
                # 1. TÍNH LỖI KHOẢNG CÁCH (Normalized -1 to 1)
                # target_x - detected_x: Dương -> Đường ở bên trái -> Cần quẹo trái (dương)
                error_dist = (target_x - detected_x) / float(w)
                
                # 2. TÍNH LỖI GÓC (Độ)
                # Góc mục tiêu là 90 độ (thẳng đứng).
                # Nếu đường nghiêng 80 độ (/), robot đang chĩa sang phải -> Cần quẹo Trái (+Z) để song song
                # Nếu đường nghiêng 100 độ (\), robot đang chĩa sang trái -> Cần quẹo Phải (-Z) để song song
                # Công thức: (Target - Actual)
                # VD: 90 - 80 = +10 (Quẹo trái)
                # VD: 90 - 100 = -10 (Quẹo phải)
                error_angle = self.target_angle_deg - detected_ang

                # 3. TỔNG HỢP LỆNH (Weighted Sum)
                raw_turn = (self.k_dist * error_dist) + (self.k_theta * error_angle)
                cmd.angular.z = max(-self.max_ang, min(self.max_ang, raw_turn))
                
                # Debug Visual
                cv2.putText(frame, f"E_Dist: {error_dist:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"E_Ang: {error_angle:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Vẽ đường target (Xanh Dương)
                cv2.line(frame, (int(target_x), roi_y_start), (int(target_x), h), (255, 0, 0), 2)
                # Vẽ đường detect được (Đỏ)
                if detected_x != -1:
                     # Vẽ 1 đường thẳng đại diện cho góc tìm được
                     cx, cy = int(detected_x), int(roi_y_start + (h-roi_y_start)/2)
                     length = 50
                     # Tính toạ độ endpoint để vẽ góc
                     ang_rad = np.radians(detected_ang)
                     # Lưu ý toạ độ ảnh: y tăng xuống dưới
                     dx = int(length * np.cos(ang_rad))
                     dy = int(length * np.sin(ang_rad))
                     # Vẽ line thể hiện hướng của luống
                     cv2.line(frame, (cx + dx, cy - dy), (cx - dx, cy + dy), (0, 0, 255), 3)

            # Publish & OSD
            self.cmd_pub.publish(cmd)
            cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Ang.Z: {cmd.angular.z:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (0, roi_y_start), (w, h), (100, 100, 100), 1)
            
            cv2.imshow("Main View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        node = DualRowFollowerDebug()
        node.run()
    except rospy.ROSInterruptException:
        pass
