# encoding=utf-8

import pcl
import pcl.pcl_visualization


pcd_f_path = './trunc.pcd'  # 注意更换自己的pcd点云文件
cloud = pcl.load_XYZRGB(pcd_f_path)  # save_XYZRGBA
visual = pcl.pcl_visualization.CloudViewing()

print(cloud[1])
visual.ShowColorCloud(cloud, b'cloud')
v = True
while v:
    v = not (visual.WasStopped())
