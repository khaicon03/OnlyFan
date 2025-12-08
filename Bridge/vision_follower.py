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
        rospy.init_node("row_follow_full_debug")
        rospy.loginfo("--- FULL DEBUG MODE STARTED ---")

        self.cmd_pub = rospy.Publisher("/cmd_vel_row", Twist, queue_size=1)

        # ========= PARAMS =========
        self.forward_speed = rospy.get_param("~forward_speed", 0.25)
        
        # Params Bám Luống
        self.left_target_ratio  = rospy.get_param("~left_target_ratio", 0.30)
        self.right_target_ratio = rospy.get_param("~right_target_ratio", 0.70)
        self.center_ratio       = rospy.get_param("~center_ratio", 0.50)
        
        self.left_min_x_ratio  = rospy.get_param("~left_min_x_ratio", 0.05)
        self.left_max_x_ratio  = rospy.get_param("~left_max_x_ratio", 0.45)
        self.right_min_x_ratio = rospy.get_param("~right_min_x_ratio", 0.55)
        self.right_max_x_ratio = rospy.get_param("~right_max_x_ratio", 0.95)
        
        self.k_ang = rospy.get_param("~k_ang", 2.0)
        self.max_ang = rospy.get_param("~max_ang", 1.0)
        self.min_angle_deg = max(35.0, rospy.get_param("~min_angle_deg", 35.0))
        
        # BỔ SUNG: Tham số điều khiển góc nghiêng
        self.max_angle_deg = rospy.get_param("~max_angle_deg", 70.0)
        self.balance_angle = (self.min_angle_deg + self.max_angle_deg) / 2.0 # Góc mục tiêu (e.g., 52.5 deg)
        self.k_ang_correct = rospy.get_param("~k_ang_correct", 0.03) # Hệ số khuếch đại cho lỗi góc
        
        self.canny1 = rospy.get_param("~canny1", 50)
        self.canny2 = rospy.get_param("~canny2", 150)
        self.hough_threshold = rospy.get_param("~hough_threshold", 60)

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

        rospy.loginfo(f"Load YOLO: {weights_path}")
        try:
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            try:
                with open(names_path, "r") as f: self.classes = [l.strip() for l in f.readlines()]
            except: self.classes = ["unknown"]

            self.layer_names = self.net.getLayerNames()
            out_layers_indices = self.net.getUnconnectedOutLayers()
            if len(out_layers_indices.shape) > 1: out_layers_indices = out_layers_indices.flatten()
            self.output_layers = [self.layer_names[i - 1] for i in out_layers_indices]
            self.yolo_ready = True
        except Exception as e:
            rospy.logerr(f"YOLO Error: {e}")
            self.yolo_ready = False

        # ========= CAMERA IMX219 (Giữ nguyên) =========
        self.gst_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=480, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)

    # =====================================================================
    # SUPPORT (Giữ nguyên)
    # =====================================================================
    def bbox_iou(self, b1, b2):
        x1, y1, x2, y2 = b1
        x1b, y1b, x2b, y2b = b2
        inter = max(0, min(x2, x2b) - max(x1, x1b)) * max(0, min(y2, y2b) - max(y1, y1b))
        area1 = max(0, x2-x1) * max(0, y2-y1)
        area2 = max(0, x2b-x1b) * max(0, y2b-y1b)
        return inter / (area1 + area2 - inter + 1e-6)

    def analyze_plant_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1=max(0, x1); y1=max(0, y1); x2=min(frame.shape[1], x2); y2=min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_h = cv2.inRange(hsv, LOWER_HEALTHY, UPPER_HEALTHY)
        mask_s = cv2.inRange(hsv, LOWER_SICK, UPPER_SICK)
        tot = float(mask_h.size)
        if tot == 0: return "unknown"
        hr = cv2.countNonZero(mask_h)/tot
        sr = cv2.countNonZero(mask_s)/tot
        if hr < 0.01 and sr < 0.01: return "no_leaf"
        return "healthy" if hr >= sr else "sick"

    def detect_objects(self, frame):
        if not self.yolo_ready: return []
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, self.yolo_input_size, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                    w, h = int(detection[2]*width), int(detection[3]*height)
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
        results = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                label = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                results.append({"box": boxes[i], "label": label, "conf": confidences[i], "class_id": class_ids[i]})
        return results

    # =====================================================================
    # HÀM DETECT_ROWS
    # =====================================================================
    def detect_rows(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, self.canny1, self.canny2)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold, int(roi.shape[0]*0.5), 25)
        
        left_xs, right_xs = [], []
        left_angles, right_angles = [], []
        
        h, w = roi.shape[:2]
        
        left_min, left_max = int(w*self.left_min_x_ratio), int(w*self.left_max_x_ratio)
        right_min, right_max = int(w*self.right_min_x_ratio), int(w*self.right_max_x_ratio)
        
        min_angle_target = self.min_angle_deg
        max_angle_target = self.max_angle_deg
        
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                dx, dy = x2 - x1, y2 - y1
                
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad) # Góc nằm trong khoảng (-180, 180)

                # Chuyển góc về phạm vi [0, 90] cho độ dốc (Slope Angle)
                slope_angle_deg = abs(angle_deg)
                if slope_angle_deg > 90:
                    slope_angle_deg = 180 - slope_angle_deg

                # KIỂM TRA GÓC
                if not (min_angle_target <= slope_angle_deg <= max_angle_target):
                    continue

                # Lấy tọa độ x để kiểm tra phân vùng (Gần phía camera hơn)
                x_check = x1 if y1 > y2 else x2 # Lấy x có y lớn hơn (gần cuối ROI)
                
                # --- PHÂN VÙNG BÊN TRÁI (LEFT LANE) ---
                if left_min <= x_check <= left_max:
                    if (90 < angle_deg < 180) or (-90 < angle_deg < 0):
                        left_xs.append(x_check)
                        left_angles.append(slope_angle_deg)
                        
                # --- PHÂN VÙNG BÊN PHẢI (RIGHT LANE) ---
                if right_min <= x_check <= right_max:
                    if (0 < angle_deg < 90) or (angle_deg < -90):
                        right_xs.append(x_check)
                        right_angles.append(slope_angle_deg)
                        
        return left_xs, left_angles, right_xs, right_angles, edges

    # =====================================================================
    # MAIN RUN
    # =====================================================================
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rate.sleep()
                continue

            curr_time = time.time()
            fps = 1.0 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = curr_time
            h, w = frame.shape[:2]
            self.frame_count += 1
            
            # 1. YOLO (Giữ nguyên)
            if self.frame_count % (self.skip_yolo_frames + 1) == 0:
                self.last_detections = self.detect_objects(frame)
            
            # 2. LOGIC (Giữ nguyên)
            obstacle_detected = False
            safe_zone_min, safe_zone_max = w / 3, 2 * w / 3
            
            for obj in self.last_detections:
                x, y, bw, bh = obj['box']
                label = obj['label']
                cx = x + bw / 2
                
                if label != "plant":
                    if safe_zone_min < cx < safe_zone_max and bh > h * 0.25:
                        obstacle_detected = True
                        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 0, 255), 3)
                        cv2.putText(frame, "!!! STOP !!!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
                if label == "plant":
                    bbox = (x, y, x+bw, y+bh)
                    same_track = None
                    for tr in self.plant_tracks:
                        if self.bbox_iou(bbox, tr["bbox"]) > 0.5: same_track = tr; break
                    if same_track:
                        same_track["last_seen"] = curr_time
                        health = same_track["health"]
                    elif self.frame_count % (self.skip_yolo_frames + 1) == 0:
                        health = self.analyze_plant_color(frame, bbox)
                        self.plant_tracks.append({"bbox": bbox, "health": health, "last_seen": curr_time})
                    else: health = "..."
                    col = (0, 255, 0) if health == "healthy" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), col, 2)
                    cv2.putText(frame, health, (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            self.plant_tracks = [tr for tr in self.plant_tracks if curr_time - tr["last_seen"] < 2.0]

            # 3. ĐIỀU KHIỂN & DEBUG DRAWING
            cmd = Twist()
            target_pixel = w * 0.5 # Default target (Center)
            detected_center = -1
            avg_lane_angle = -1.0 # Giá trị mặc định

            if obstacle_detected:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                state_text = "OBSTACLE STOP"
            else:
                # =============================================================
                # CẬP NHẬT ROI: LẤY 2/3 MÀN HÌNH (Bỏ 1/3 đầu)
                # =============================================================
                roi_start_h = int(h / 3) # Bắt đầu từ 1/3
                roi = frame[roi_start_h:h, :]
                
                left_xs, left_angles, right_xs, right_angles, edges = self.detect_rows(roi)

                cv2.imshow("Debug: Canny Edges", edges)
                
                # --- PHẦN 1: TÍNH TOÁN VỊ TRÍ & GÓC TRUNG BÌNH ---
                if len(left_xs) > 0 and len(right_xs) > 0:
                    detected_center = (max(left_xs) + min(right_xs)) / 2.0
                    target_pixel = self.center_ratio * w
                    cmd.linear.x = self.forward_speed
                    state_text = "Dual Lane"
                    # Tính góc trung bình từ cả hai luống
                    all_angles = left_angles + right_angles
                    if len(all_angles) > 0:
                        avg_lane_angle = np.mean(all_angles)
                        
                elif len(left_xs) > 0:
                    detected_center = max(left_xs)
                    target_pixel = self.left_target_ratio * w
                    cmd.linear.x = self.forward_speed
                    state_text = "Left Lane Only"
                    if len(left_angles) > 0:
                        avg_lane_angle = np.mean(left_angles)
                        
                elif len(right_xs) > 0:
                    detected_center = min(right_xs)
                    target_pixel = self.right_target_ratio * w
                    cmd.linear.x = self.forward_speed
                    state_text = "Right Lane Only"
                    if len(right_angles) > 0:
                        avg_lane_angle = np.mean(right_angles)
                        
                else:
                    cmd.linear.x = 0.0
                    state_text = "No Lane Found"
                    detected_center = -1
                    avg_lane_angle = -1.0

                # --- PHẦN 2: ĐIỀU KHIỂN GÓC (Yaw) ---
                ang = 0.0
                
                # 2a. Điều khiển vị trí ngang (giữ luống ở target_pixel)
                if detected_center != -1:
                    error = (target_pixel - detected_center) / float(w)
                    ang += self.k_ang * error
                
                # 2b. Điều khiển góc nghiêng (giữ góc luống ở balance_angle)
                if avg_lane_angle != -1.0:
                    angle_error = avg_lane_angle - self.balance_angle
                    ang += self.k_ang_correct * angle_error

                # Giới hạn Angular Z cuối cùng
                cmd.angular.z = max(-self.max_ang, min(self.max_ang, ang))

                # --- VẼ VISUALIZATION ---
                # CẬP NHẬT: Vẽ từ điểm bắt đầu ROI mới
                y_roi = roi_start_h
                
                # 1. Vẽ các đường tìm được
                for val in left_xs: cv2.line(frame, (val, y_roi), (val, h), (255, 255, 0), 2)
                for val in right_xs: cv2.line(frame, (val, y_roi), (val, h), (0, 255, 255), 2)
                
                # 2. Vẽ điểm đích (Màu xanh dương)
                cv2.line(frame, (int(target_pixel), y_roi), (int(target_pixel), h), (255, 0, 0), 3)
                
                # 3. Vẽ tâm đường tìm được (Chấm đỏ)
                if detected_center != -1:
                    cv2.circle(frame, (int(detected_center), int(h*0.75)), 8, (0, 0, 255), -1)
                    cv2.line(frame, (int(detected_center), int(h*0.75)), (int(target_pixel), int(h*0.75)), (0, 255, 255), 2)
                
                # BỔ SUNG: Vẽ góc trung bình (Debug mới)
                if avg_lane_angle != -1.0:
                    cv2.putText(frame, f"Avg Angle: {avg_lane_angle:.1f} deg", (w - 220, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


            self.cmd_pub.publish(cmd)

            # --- OSD (ON SCREEN DISPLAY) ---
            # Hiển thị thông số lên màn hình
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {state_text}", (10, 60), font, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Lin.X: {cmd.linear.x:.2f}", (10, 120), font, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Ang.Z: {cmd.angular.z:.2f}", (10, 140), font, 0.6, (255, 255, 255), 1)
            
            # CẬP NHẬT: Vẽ khung ROI (Vùng quét) 2/3
            cv2.rectangle(frame, (0, int(h / 3)), (w, h), (100, 100, 100), 1)

            cv2.imshow("Main Vision: YOLO + Lane + Logic", frame)
            
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
