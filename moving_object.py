# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
from sensor import RaySensor
from director import vtkAll as vtk
from director.debugVis import DebugData
from director import ioUtils, filterUtils


class MovingObject(object):

    """Moving object."""

    def __init__(self, velocity, polydata):
        """Constructs a MovingObject.

        Args:
            velocity: Velocity.
            polydata: Polydata.
        """
        self._state = [0., 0., 0.]
        self._velocity = float(velocity)
        self._polydata = polydata
        self._sensors = []

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

    @property
    def velocity(self):
        """Velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        """Velocity."""
        self._velocity = float(value)

    @property
    def sensors(self):
        """List of attached sensors."""
        return self._sensors

    def attach_sensor(self, sensor):
        """Attaches a sensor.

        Args:
            sensor: Sensor.
        """
        self._sensors.append(sensor)

    def _dynamics(self, state, t, controller=None):
        """Dynamics of the object.

        Args:
            state: Initial condition.
            t: Time.
            controller: Callable that takes state and t as inputs and returns
                yaw.

        Returns:
            Derivative of state at t.
        """
        if controller:
            yaw = controller(state, t)
        else:
            yaw = self._velocity * np.sin(t)

        dqdt = np.zeros_like(state)
        dqdt[0] = self._velocity * np.cos(state[2])
        dqdt[1] = self._velocity * np.sin(state[2])
        dqdt[2] = yaw
        return dqdt

    def _simulate(self, dt, controller, start_time=0.0, steps=1):
        """Simulates the object moving.

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

    def move(self, dt=1.0/30.0, controller=None):
        """Moves the object by a given time step.

        Args:
            dt: Length of time step.
            controller: Yaw controller to use.
        """
        states = self._simulate(dt, controller)
        self._state = states[-1, :]
        list(map(lambda s: s.update(*self._state), self._sensors))

    def to_polydata(self):
        """Converts object to visualizable poly data."""
        return self._polydata


class Robot(MovingObject):

    """Robot."""

    def __init__(self, velocity=15.0, scale=0.10, model="A10.obj"):
        """Constructs a Robot.

        Args:
            velocity: Velocity of the robot in the forward direction.
            scale: Scale of the model.
            model: Object model to use.
        """
        t = vtk.vtkTransform()
        t.Scale(scale, scale, scale)
        polydata = ioUtils.readPolyData(model)
        polydata = filterUtils.transformPolyData(polydata, t)
        super(Robot, self).__init__(velocity, polydata)


class Obstacle(MovingObject):

    """Obstacle."""

    def __init__(self, velocity, radius, height=1.0):
        """Constructs a Robot.

        Args:
            velocity: Velocity of the robot in the forward direction.
            radius: Radius of the obstacle.
        """
        data = DebugData()
        self.radius = radius
        self.height = height
        self.center = [0, 0, height / 2 - 0.5]
        self.axis = [0, 0, 1]  # Upright cylinder.
        data.addCylinder(self.center, self.axis, height, radius)
        polydata = data.getPolyData()
        super(Obstacle, self).__init__(velocity, polydata)
