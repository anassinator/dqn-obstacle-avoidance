# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
from director import vtkAll as vtk
from director import ioUtils, filterUtils


class Robot(object):

    """Robot."""

    def __init__(self, velocity=12.0, scale=0.04, model="celica.obj"):
        """Constructs a Robot.

        Args:
            velocity: Velocity of the robot in the forward direction.
            scale: Scale of the model.
            model: Object model to use.
        """
        self._state = [0., 0., 0.]  # x, y, theta
        self._velocity = velocity

        t = vtk.vtkTransform()
        t.RotateZ(90)
        t.Scale(scale, scale, scale)
        polydata = ioUtils.readPolyData(model)
        self._polydata = filterUtils.transformPolyData(polydata, t)

    @property
    def x(self):
        """X coordinate."""
        return self._state[0]

    @x.setter
    def x(self, value):
        """X coordinate."""
        self._state[0] = float(value)

    @property
    def y(self):
        """Y coordinate."""
        return self._state[1]

    @y.setter
    def y(self, value):
        """Y coordinate."""
        self._state[1] = float(value)

    @property
    def theta(self):
        """Yaw in radians."""
        return self._state[2]

    @theta.setter
    def theta(self, value):
        """Yaw in radians."""
        self._state[2] = float(value)

    def _dynamics(self, state, t, controller=None):
        """Dynamics of the robot.

        Args:
            state: Initial condition.
            t: Time.
            controller: Callable that takes state and t as inputs and returns
                yaw.

        Returns:
            Derivative of state at t.
        """
        dqdt = np.zeros_like(state)
        dqdt[0] = self._velocity * np.cos(state[2])
        dqdt[1] = self._velocity * np.sin(state[2])
        dqdt[2] = controller(state, t) if controller else np.sin(t)
        return dqdt

    def _simulate(self, dt, controller, start_time=0.0, steps=1):
        """Simulates the robot moving.

        Args:
            dt: Time between steps.
            controller: Yaw controller to use.
            start_time: Start time to use, default: 0.
            steps: Number of steps, default: 1.

        Returns:
            State for each step taken.
        """
        t = np.arange(start_time, start_time + (steps + 1) * dt, dt)
        states = integrate.odeint(self._dynamics, self._state, t)
        return states

    def move(self, dt=0.05, controller=None):
        """Moves the robot by a given time step.

        Args:
            dt: Length of time step.
            controller: Yaw controller to use.
        """
        states = self._simulate(dt, controller)
        self._state = states[-1, :]

    def to_polydata(self):
        """Converts robot to visualizable poly data."""
        return self._polydata
