import numpy as np
import mayavi.mlab

bin_f_path = "./000011.bin"
point_cloud = np.fromfile(bin_f_path,
                         dtype=np.float32, count=-1).reshape([-1, 4])

txt_f_path = bin_f_path.replace('.bin', '.txt')
np.savetxt(txt_f_path, point_cloud)
print('{:s} saved.'.format(txt_f_path))

print(point_cloud.shape)
x = point_cloud[:, 0]  # x position of point
y = point_cloud[:, 1]  # y position of point
z = point_cloud[:, 2]  # z position of point
r = point_cloud[:, 3]  # reflectance value of point
# b = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

# f = open('000000.txt', 'w')
# for i in range(pointcloud.shape[0]):
# 	f.write(str(x[i]))
# 	f.write(' ')
# 	f.write(str(y[i]))
# 	f.write(' ')
# 	f.write(str(z[i]))
# 	f.write(' ')
# 	f.write(str(z[i]))
# 	f.write(' ')
# 	f.write('\n')

color = r

fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(960, 640))
mayavi.mlab.points3d(x, y, z,
                     z,          # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )

x = np.linspace(5, 5, 50)
y = np.linspace(0, 0, 50)
z = np.linspace(0, 5, 50)
mayavi.mlab.plot3d(x, y, z)
mayavi.mlab.show()
