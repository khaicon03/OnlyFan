#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse
from topic_tools.srv import MuxSelect

class ModeManager(object):
    MODE_ROW = "ROW"
    MODE_NAV = "NAV"

    def __init__(self):
        # ---- Tham số từ ROS param ----
        self.row_topic = rospy.get_param("~row_topic", "/cmd_vel_row")
        self.nav_topic = rospy.get_param("~nav_topic", "/cmd_vel_nav")
        self.mux_select_srv_name = rospy.get_param("~mux_select_service",
                                                   "/cmd_vel_mux/select")

        # dùng odom để trigger hết luống?
        self.use_distance_trigger = rospy.get_param("~use_distance_trigger", True)
        self.row_length = rospy.get_param("~row_length", 10.0)  # mét

        # topic báo hết luống (nếu bạn dùng thêm cảm biến khác)
        self.use_end_of_row_topic = rospy.get_param("~use_end_of_row_topic", False)
        self.end_of_row_topic = rospy.get_param("~end_of_row_topic", "/end_of_row")

        # ---- Biến trạng thái ----
        self.current_mode = None
        self.odom_inited = False
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.distance_in_row = 0.0

        # ---- Subscriber ----
        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        if self.use_end_of_row_topic:
            rospy.Subscriber(self.end_of_row_topic, Bool, self.end_of_row_cb)

        # ---- Service choose mux ----
        rospy.wait_for_service(self.mux_select_srv_name)
        self.mux_select = rospy.ServiceProxy(self.mux_select_srv_name, MuxSelect)

        # ---- Service cho phép đổi mode bằng tay ----
        rospy.Service("~set_row_mode", Trigger, self.handle_set_row_mode)
        rospy.Service("~set_nav_mode", Trigger, self.handle_set_nav_mode)

        rospy.loginfo("ModeManager started.")
        rospy.loginfo(" row_topic = %s", self.row_topic)
        rospy.loginfo(" nav_topic = %s", self.nav_topic)

        # mặc định vào luống đầu tiên → mode ROW
        self.switch_mode(self.MODE_ROW)

    # ==========================================================
    # Callbacks
    # ==========================================================
    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if not self.odom_inited:
            self.last_odom_x = x
            self.last_odom_y = y
            self.odom_inited = True
            return

        # chỉ tích luỹ khoảng cách nếu đang ở MODE_ROW
        if self.current_mode == self.MODE_ROW and self.use_distance_trigger:
            dx = x - self.last_odom_x
            dy = y - self.last_odom_y
            ds = math.sqrt(dx*dx + dy*dy)
            self.distance_in_row += ds

            # cập nhật last
            self.last_odom_x = x
            self.last_odom_y = y

            # kiểm tra hết luống
            if self.distance_in_row >= self.row_length:
                rospy.loginfo("Row length reached (%.2f m). Switch to NAV mode.",
                              self.distance_in_row)
                self.switch_mode(self.MODE_NAV)

        else:
            # vẫn cập nhật last odom cho chuẩn
            self.last_odom_x = x
            self.last_odom_y = y

    def end_of_row_cb(self, msg):
        # nếu dùng topic /end_of_row (data=True) để báo cuối luống
        if msg.data and self.current_mode == self.MODE_ROW:
            rospy.loginfo("Received end_of_row=True. Switch to NAV mode.")
            self.switch_mode(self.MODE_NAV)

    # ==========================================================
    # Service handlers
    # ==========================================================
    def handle_set_row_mode(self, req):
        self.switch_mode(self.MODE_ROW)
        return TriggerResponse(success=True,
                               message="Switched to ROW mode (camera control).")

    def handle_set_nav_mode(self, req):
        self.switch_mode(self.MODE_NAV)
        return TriggerResponse(success=True,
                               message="Switched to NAV mode (move_base control).")

    # ==========================================================
    # Logic chuyển mode
    # ==========================================================
    def switch_mode(self, mode):
        if mode == self.current_mode:
            rospy.loginfo("Already in mode %s", mode)
            return

        if mode == self.MODE_ROW:
            topic = self.row_topic
            # reset quãng đường khi vào luống
            self.distance_in_row = 0.0
            rospy.loginfo("Switching to ROW mode. Select %s on mux.", topic)

        elif mode == self.MODE_NAV:
            topic = self.nav_topic
            rospy.loginfo("Switching to NAV mode. Select %s on mux.", topic)

        else:
            rospy.logwarn("Unknown mode: %s", mode)
            return

        try:
            prev = self.mux_select(topic)
            rospy.loginfo("Mux select: %s (previous: %s)",
                          topic, prev.prev_topic)
            self.current_mode = mode
        except Exception as e:
            rospy.logerr("Failed to call mux select service: %s", e)


if __name__ == "__main__":
    rospy.init_node("mode_manager")
    mgr = ModeManager()
    rospy.spin()
