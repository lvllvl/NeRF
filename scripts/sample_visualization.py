import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a 3D point
point = np.array([1, 0, 0])

# Define a quaternion: 90-degree rotation around Y-axis
theta = np.pi / 2  # 90 degrees in radians
qw = np.cos(theta / 2)
qx = 0
qy = np.sin(theta / 2)
qz = 0
q = [qw, qx, qy, qz]

# Quaternion to rotation matrix function
def quaternion_to_rotation_matrix(q):
    """
    :return: A rotation matrix R
    """
    qw, qx, qy, qz = q
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])
    return R

# Convert quaternion to rotation matrix
R = quaternion_to_rotation_matrix(q)

# Apply rotation
rotated_point = R @ point # also we can do rotated_point = np.matmul(R, point)

# Plot original and rotated vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot origin
ax.quiver(0, 0, 0, *point, color='blue', label='Original Vector (X axis)')
ax.quiver(0, 0, 0, *rotated_point, color='red', label='Rotated Vector')

# Coordinate axes for reference
ax.quiver(0, 0, 0, 1, 0, 0, color='gray', alpha=0.3)
ax.quiver(0, 0, 0, 0, 1, 0, color='gray', alpha=0.3)
ax.quiver(0, 0, 0, 0, 0, 1, color='gray', alpha=0.3)

ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Quaternion Rotation: 90° Around Y-axis")
ax.legend()

plt.tight_layout()
plt.show()
