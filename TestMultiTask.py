# encoding=utf-8

import os
import time
import threading
from threading import Condition


class XiaoMing(threading.Thread):
    def __init__(self, condition):
        super().__init__(name='小明')
        self.condition = condition
    
    def run(self):
        global count

        with self.condition:
            while count < 10:
                # ---------- Do something...
                print('{:s}, {:d}'.format(self.name, count))
                count += 1
                # ----------

                self.condition.notify()
                self.condition.wait()


class XiaoHong(threading.Thread):
    def __init__(self, condition):
        super().__init__(name='小红')
        self.condition = condition

    def run(self):
        global count

        with self.condition:

            while count < 10:
                self.condition.wait()

                # ---------- Do something...
                print('{:s}, {:d}'.format(self.name, count))
                count += 1
                # ----------

                self.condition.notify()

    

def TestTaskSwitch():
    condition = threading.Condition()

    global count
    count = 0
    
    XM = XiaoMing(condition)
    XH = XiaoHong(condition)

    XH.start()
    XM.start()
    
    XH.join()
    XM.join()


if __name__ == '__main__':
    TestTaskSwitch()
    print('Done.')