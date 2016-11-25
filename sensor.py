# -*- coding: utf-8 -*=

import math
import numpy as np
import director.vtkAll as vtk


class Sensor(object):

    def __init__(self, world, num_rays=20, radius=8,
                 min_angle=-90, max_angle=90):

        self.num_rays = num_rays
        self.radius = radius
        self.min_angle = math.radians(min_angle)
        self.max_angle = math.radians(max_angle)

        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(world.to_polydata())
        self.locator.BuildLocator()

        self.results = np.zeros(self.num_rays)
        self.distances = np.zeros(self.num_rays)
        self.intersections = [[0, 0, 0] for i in range(self.num_rays)]

        self._update_rays(0)

    def set_locator(self, locator):
        self.locator = locator

    def update(self, x, y, theta):
        self._update_rays(theta)
        origin = np.array([x, y, 0])

        for i in range(self.num_rays):
            res, dist, inter = self._cast_ray(origin, origin + self.rays[i])
            self.results[i] = res
            self.distances[i] = dist
            self.intersections[i] = inter

    def _update_rays(self, theta):
        self.rays = [
            np.array([
                math.cos(
                    theta + self.min_angle + i *
                    (self.max_angle - self.min_angle) / (self.num_rays - 1)
                ) * self.radius,
                math.sin(
                    theta + self.min_angle + i *
                    (self.max_angle - self.min_angle) / (self.num_rays - 1)
                ) * self.radius,
                0
            ])
            for i in range(self.num_rays)
        ]

    def _cast_ray(self, start, end):

        tolerance = 0.0                 # intersection tolerance
        pt = [0.0, 0.0, 0.0]            # coordinate of intersection
        distance = vtk.mutable(0.0)     # distance of intersection
        pcoords = [0.0, 0.0, 0.0]       # location within intersected cell
        subID = vtk.mutable(0)          # subID of intersected cell

        result = self.locator.IntersectWithLine(start, end, tolerance,
                                                distance, pt, pcoords, subID)

        return result, distance, pt
