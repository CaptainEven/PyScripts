# encoding=utf-8
"""
Switching between 2 tasks.
Using wait & notify.
"""

import os
import time
import threading
from threading import Condition


class XiaoMing(threading.Thread):
    def __init__(self, condition):
        super().__init__(name='小明')
        self.condition = condition
        self.inner_count = 0  # 内部计数

    def run(self):
        global count, total_count

        with self.condition:
            while count < total_count:
                # ---------- Do something...
                count += 1
                self.inner_count += 1
                print('{:s} | global count: {:d}, inner count: {:d}'
                      .format(self.name, count, self.inner_count))
                # ----------

                self.condition.notify()
                self.condition.wait()


class XiaoHong(threading.Thread):
    def __init__(self, condition):
        super().__init__(name='小红')
        self.condition = condition
        self.inner_count = 0  # 内部计数

    def run(self):
        global count, total_count

        with self.condition:

            while count < total_count:
                self.condition.wait()

                # ---------- Do something...
                count += 1
                self.inner_count += 1
                print('{:s} | global count: {:d}, inner_count: {:d}'
                      .format(self.name, count, self.inner_count))
                # ----------

                self.condition.notify()


def TestTaskSwitch():
    condition = threading.Condition()

    global count, total_count
    count, total_count = 0, 100

    XM = XiaoMing(condition)
    XH = XiaoHong(condition)

    XH.start()
    XM.start()

    XH.join()
    XM.join()


if __name__ == '__main__':
    TestTaskSwitch()
    print('Done.')
