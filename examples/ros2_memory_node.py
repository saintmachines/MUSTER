"""A ROS 2 node that gives a robot MUSTER memory (sketch; needs rclpy, not tested in this repository's CI).

  observations   std_msgs/String on /memory/observe  -> stored with the receive time
  questions      std_msgs/String on /memory/ask      -> the recalled context on /memory/context

Run:  python examples/ros2_memory_node.py --store robot.muster
"""
import argparse
import datetime

import muster


def main():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="robot.muster")
    a = ap.parse_args()
    store = muster.Store.open(a.store)

    class MemoryNode(Node):
        def __init__(self):
            super().__init__("muster_memory")
            self.create_subscription(String, "/memory/observe", self.observe, 50)
            self.create_subscription(String, "/memory/ask", self.ask, 10)
            self.pub = self.create_publisher(String, "/memory/context", 10)

        def observe(self, msg):
            store.add({"speaker": "robot", "text": msg.data,
                       "time": datetime.datetime.now().isoformat(timespec="seconds")})

        def ask(self, msg):
            out = String()
            out.data = store.context(msg.data).text
            self.pub.publish(out)

    rclpy.init()
    rclpy.spin(MemoryNode())


if __name__ == "__main__":
    main()
