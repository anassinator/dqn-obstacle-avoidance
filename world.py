# -*- coding: utf-8 -*=

import numpy as np
from moving_object import Obstacle
from director.debugVis import DebugData


class World(object):

    """Base world."""

    def __init__(self, width, height):
        """Construct an empty world.

        Args:
            width: Width of the field.
            height: Height of the field.
        """
        self._data = DebugData()

        self._width = width
        self._height = height
        self._add_boundaries()

    def _add_boundaries(self):
        """Adds boundaries to the world."""
        self._x_max, self._x_min = self._width / 2, -self._width / 2
        self._y_max, self._y_min = self._height / 2, -self._height / 2

        corners = [
            (self._x_max, self._y_max, 0),  # Top-right corner.
            (self._x_max, self._y_min, 0),  # Bottom-right corner.
            (self._x_min, self._y_min, 0),  # Bottom-left corner.
            (self._x_min, self._y_max, 0)   # Top-left corner.
        ]

        # Loopback to begining.
        corners.append(corners[0])

        for start, end in zip(corners, corners[1:]):
            self._data.addLine(start, end, radius=0.2)

    def generate_obstacles(self, density=0.05, moving_obstacle_ratio=0.20,
                           seed=None):
        """Generates randomly scattered obstacles to the world.

        Args:
            density: Obstacle to world area ratio, default: 0.1.
            moving_obstacle_ratio: Ratio of moving to stationary obstacles,
                default: 0.2.
            seed: Random seed, default: None.

        Yields:
            Obstacle.
        """
        if seed is not None:
            np.random.seed(seed)

        field_area = self._width * self._height
        obstacle_area = int(field_area * density)

        bounds = self._x_min, self._x_max, self._y_min, self._y_max
        while obstacle_area > 0:
            radius = np.random.uniform(1.0, 3.0)
            center_x_range = (self._x_min + radius, self._x_max - radius)
            center_y_range = (self._y_min + radius, self._y_max - radius)
            center_x = np.random.uniform(*center_x_range)
            center_y = np.random.uniform(*center_y_range)
            theta = np.random.uniform(0., 360.)
            obstacle_area -= np.pi * radius ** 2

            # Only some obstacles should be moving.
            if np.random.random_sample() >= moving_obstacle_ratio:
                velocity = 0.0
            else:
                velocity = np.random.uniform(-30.0, 30.0)

            obstacle = Obstacle(velocity, radius, bounds)
            obstacle.x = center_x
            obstacle.y = center_y
            obstacle.theta = np.radians(theta)
            yield obstacle

    def to_polydata(self):
        """Converts world to visualizable poly data."""
        return self._data.getPolyData()
