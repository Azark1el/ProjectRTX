import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("SAMdata.txt", skiprows=1)

time = data[:, 0]
pos_x, pos_y, pos_z = data[:, 1], data[:, 2], data[:, 3]
vel_x, vel_y, vel_z = data[:, 4], data[:, 5], data[:, 6]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(pos_x, pos_y, pos_z, label="Projectile Path", color='b')

ax.scatter(pos_x[0], pos_y[0], pos_z[0], color='g', marker='o', s=100, label="Start")

ax.scatter(pos_x[-1], pos_y[-1], pos_z[-1], color='r', marker='x', s=100, label="End")

ax.set_xlabel("Position X (m)")
ax.set_ylabel("Position Y (m)")
ax.set_zlabel("Position Z (m)")
ax.set_title("3D Projectile Motion")
ax.legend()

plt.show()
