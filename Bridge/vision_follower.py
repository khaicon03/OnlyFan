#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import json
import os
import time

from geometry_msgs.msg import Twist


class VisionFollower(object):
    def __init__(self):
        rospy.loginfo("VisionFollower (Python2, timer-based) started.")

        # ==== PARAMS ====
        # File JSON mà bên Python3 (vision_trt.py) ghi ra
        self.state_file = rospy.get_param("~vision_state_file",
                                          "/tmp/bridge.json")

        # Timeout: nếu dữ liệu cũ hơn ngần này giây -> coi như mất vision
        self.state_timeout = rospy.get_param("~vision_state_timeout", 0.5)

        # Tốc độ tiến mặc định
        self.forward_speed = rospy.get_param("~forward_speed", 0.25)

        # Gain điều khiển góc lái: ang = k_ang * steer_error
        self.k_ang = rospy.get_param("~k_ang", 2.0)
        self.max_ang = rospy.get_param("~max_ang", 1.0)

        # Chu kỳ điều khiển (giây)
        self.control_period = rospy.get_param("~control_period", 0.1)

        # Chủ đề xuất lệnh
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel_row")
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)

        # Timer thay cho camera callback
        self.timer = rospy.Timer(rospy.Duration(self.control_period),
                                 self.control_loop)

    # ------------------------------------------------------------------
    def read_state(self):
        """
        Đọc file JSON do Python3 vision_trt.py ghi ra.
        Trả về dict hoặc None nếu đọc thất bại.
        """
        if not os.path.exists(self.state_file):
            return None

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            rospy.logwarn("Failed to read vision state: %s", str(e))
            return None

    # ------------------------------------------------------------------
    def stop_robot(self, reason=""):
        """
        Gửi lệnh dừng robot cho an toàn.
        """
        if reason:
            rospy.logwarn("STOP robot: %s", reason)
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    # ------------------------------------------------------------------
    def control_loop(self, event):
        """
        Hàm được gọi đều đặn bởi rospy.Timer.
        Ở đây ta chỉ đọc state & xuất /cmd_vel_row.
        """
        state = self.read_state()
        now = time.time()

        # 1) Không có dữ liệu -> dừng
        if state is None:
            self.stop_robot("no vision_state_file")
            return

        stamp = state.get("stamp", None)
        if stamp is None:
            self.stop_robot("vision state has no stamp")
            return

        # 2) Nếu dữ liệu quá cũ -> dừng
        try:
            age = now - float(stamp)
        except Exception:
            self.stop_robot("stamp not float")
            return

        if age > self.state_timeout:
            self.stop_robot("vision state too old (age=%.3f s)" % age)
            return

        # 3) Nếu có vật cản -> dừng
        obstacle = bool(state.get("obstacle", False))
        if obstacle:
            self.stop_robot("obstacle_on_path")
            return

        # 4) Nếu không thấy luống -> dừng
        has_rows = bool(state.get("has_rows", False))
        if not has_rows:
            self.stop_robot("no rows detected")
            return

        # 5) Lúc này: không obstacle + có rows -> bám luống
        steer_error = float(state.get("steer_error", 0.0))

        cmd = Twist()
        cmd.linear.x = self.forward_speed

        ang = self.k_ang * steer_error
        if ang > self.max_ang:
            ang = self.max_ang
        elif ang < -self.max_ang:
            ang = -self.max_ang
        cmd.angular.z = ang

        self.cmd_pub.publish(cmd)


# =====================================================================
if __name__ == "__main__":
    rospy.init_node("vision_follower")
    node = VisionFollower()
    rospy.spin()
