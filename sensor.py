# -*- coding: utf-8 -*=

import math
import numpy as np
import director.vtkAll as vtk
from director.debugVis import DebugData


class RaySensor(object):

    """Ray sensor."""

    def __init__(self, num_rays=24, radius=10, min_angle=0, max_angle=360):
        """Constructs a RaySensor.

        Args:
            num_rays: Number of rays.
            radius: Max distance of the rays.
            min_angle: Minimum angle of the rays in degrees.
            max_angle: Maximum angle of the rays in degrees.
        """
        self._num_rays = num_rays
        self._radius = radius
        self._min_angle = math.radians(min_angle)
        self._max_angle = math.radians(max_angle)

        self._locator = None
        self._state = [0., 0., 0.]  # x, y, theta

        self._results = np.zeros(num_rays)
        self._distances = np.zeros(num_rays)
        self._intersections = [[0, 0, 0] for i in range(num_rays)]

        self._update_rays(self._state[2])

    def set_locator(self, locator):
        """Sets the vtk cell locator.

        Args:
            locator: Cell locator.
        """
        self._locator = locator

    def update(self, x, y, theta):
        """Updates the sensor's readings.

        Args:
            x: X coordinate.
            y: Y coordinate.
            theta: Yaw.
        """
        self._update_rays(theta)
        origin = np.array([x, y, 0])
        self._state = [x, y, theta]

        if self._locator is None:
            return

        for i in range(self._num_rays):
            res, dist, inter = self._cast_ray(origin, origin + self._rays[i])
            self._results[i] = res
            self._distances[i] = dist
            self._intersections[i] = inter

    def _update_rays(self, theta):
        """Updates the rays' readings.

        Args:
            theta: Yaw.
        """
        r = self._radius
        angle_step = (self._max_angle - self._min_angle) / self._num_rays
        self._rays = [
            np.array([
                r * math.cos(theta + self._min_angle + i * angle_step),
                r * math.sin(theta + self._min_angle + i * angle_step),
                0
            ])
            for i in range(self._num_rays)
        ]

    def _cast_ray(self, start, end):
        """Casts a ray and determines intersections and distances.

        Args:
            start: Origin of the ray.
            end: End point of the ray.

        Returns:
            Tuple of (result, distance, intersection).
        """
        tolerance = 0.0                 # intersection tolerance
        pt = [0.0, 0.0, 0.0]            # coordinate of intersection
        distance = vtk.mutable(0.0)     # distance of intersection
        pcoords = [0.0, 0.0, 0.0]       # location within intersected cell
        subID = vtk.mutable(0)          # subID of intersected cell

        result = self._locator.IntersectWithLine(start, end, tolerance,
                                                 distance, pt, pcoords, subID)

        return result, distance, pt

    def to_polydata(self):
        """Converts the sensor to polydata."""
        d = DebugData()
        origin = np.array([self._state[0], self._state[1], 0])
        for result, intersection, ray in zip(self._results,
                                             self._intersections,
                                             self._rays):
            if result:
                color = [1., 0.45882353, 0.51372549]
                endpoint = intersection
            else:
                color = [0., 0.6, 0.58823529]
                endpoint = origin + ray

            d.addLine(origin, endpoint, color=color, radius=0.05)

        return d.getPolyData()
