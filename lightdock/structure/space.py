import numpy as np
from lightdock.mathutil.cython.quaternion import Quaternion


class SpacePoints(object):
    """A collection of spatial points"""

    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)

    def clone(self):
        return SpacePoints(self.coordinates.copy())

    def translate(self, vector):
        """Translates coordinates based on vector"""
        self.coordinates += vector

    def rotate(self, q):
        """Rotates coordinates using a quaternion q"""
        for i in range(self.coordinates.shape[0]):
            self.coordinates[i, ...] = q.rotate(self.coordinates[i, ...])

    def rotate_over(self, axis_indices, to_rotate_indices, angle):
        """Rotates specified coordinates over an axis. Angle is expressed in radians"""
        # Axis to rotate over
        axis_coordinates = self.coordinates[axis_indices[1]] - self.coordinates[axis_indices[0]]
        axis_coordinates /= np.linalg.norm(axis_coordinates)

        axis_quat = Quaternion(
            np.cos(angle / 2.0),
            axis_coordinates[0] * np.sin(angle / 2.0),
            axis_coordinates[1] * np.sin(angle / 2.0),
            axis_coordinates[2] * np.sin(angle / 2.0),
        )

        origin = self.coordinates[axis_indices[0]]
        for i in range(to_rotate_indices.shape[0]):
            # Translate to origin
            self.coordinates[to_rotate_indices[i]] -= origin
            # Rotate
            self.coordinates[to_rotate_indices[i], ...] = axis_quat.rotate(self.coordinates[to_rotate_indices[i], ...])
            # Translate back from origin
            self.coordinates[to_rotate_indices[i]] += origin

    def __getitem__(self, item):
        return self.coordinates[item]

    def __setitem__(self, index, item):
        self.coordinates[index] = item

    def __iter__(self):
        for coordinates in self.coordinates:
            yield coordinates

    def __len__(self):
        return self.coordinates.shape[0]

    def __eq__(self, other):
        return np.allclose(self.coordinates, other.coordinates)

    def __ne__(self, other):
        return not (self == other)

    def __sub__(self, other):
        return self.coordinates - other.coordinates

    def __str__(self):
        return str(self.coordinates)

    def shape(self):
        return self.coordinates.shape
