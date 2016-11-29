# -*- coding: utf-8 -*-

import numpy as np
from net import Controller
from scipy import integrate
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
        self._state = np.array([0., 0., 0.])
        self._velocity = float(velocity)
        self._raw_polydata = polydata
        self._polydata = polydata
        self._sensors = []

    @property
    def x(self):
        """X coordinate."""
        return self._state[0]

    @x.setter
    def x(self, value):
        """X coordinate."""
        next_state = self._state.copy()
        next_state[0] = float(value)
        self._update_state(next_state)

    @property
    def y(self):
        """Y coordinate."""
        return self._state[1]

    @y.setter
    def y(self, value):
        """Y coordinate."""
        next_state = self._state.copy()
        next_state[1] = float(value)
        self._update_state(next_state)

    @property
    def theta(self):
        """Yaw in radians."""
        return self._state[2]

    @theta.setter
    def theta(self, value):
        """Yaw in radians."""
        next_state = self._state.copy()
        next_state[2] = float(value)
        self._update_state(next_state)

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

        Returns:
            Derivative of state at t.
        """
        dqdt = np.zeros_like(state)
        dqdt[0] = self._velocity * np.cos(state[2])
        dqdt[1] = self._velocity * np.sin(state[2])
        dqdt[2] = self._control(state, t)
        return dqdt

    def _control(self, state, t):
        """Returns the yaw given state.

        Args:
            state: State.
            t: Time.

        Returns:
            Yaw.
        """
        return self._velocity * np.sin(2 * np.pi * t)

    def _simulate(self, dt, start_time=0.0, steps=1):
        """Simulates the object moving.

        Args:
            dt: Time between steps.
            start_time: Start time to use, default: 0.
            steps: Number of steps, default: 1.

        Returns:
            State for each step taken.
        """
        t = np.arange(start_time, start_time + (steps + 1) * dt, dt)
        states = integrate.odeint(self._dynamics, self._state, t)
        return states

    def move(self, dt=1.0/30.0):
        """Moves the object by a given time step.

        Args:
            dt: Length of time step.
        """
        states = self._simulate(dt)
        self._update_state(states[-1, :])
        list(map(lambda s: s.update(*self._state), self._sensors))

    def _update_state(self, next_state):
        """Updates the moving object's state.

        Args:
            next_state: New state.
        """
        t = vtk.vtkTransform()
        t.Translate([next_state[0], next_state[1], 0.])
        t.RotateZ(np.degrees(next_state[2]))
        self._polydata = filterUtils.transformPolyData(self._raw_polydata, t)
        self._state = next_state

    def to_positioned_polydata(self):
        """Converts object to visualizable poly data.

        Note: Transformations have been already applied to this.
        """
        return self._polydata

    def to_polydata(self):
        """Converts object to visualizable poly data.

        Note: This is centered at (0, 0, 0) and is not rotated.
        """
        return self._raw_polydata


class Robot(MovingObject):

    """Robot."""

    def __init__(self, target, velocity=25.0, scale=0.15, model="A10.obj"):
        """Constructs a Robot.

        Args:
            velocity: Velocity of the robot in the forward direction.
            scale: Scale of the model.
            model: Object model to use.
        """
        self._target = target
        t = vtk.vtkTransform()
        t.Scale(scale, scale, scale)
        polydata = ioUtils.readPolyData(model)
        polydata = filterUtils.transformPolyData(polydata, t)
        super(Robot, self).__init__(velocity, polydata)
        self._nn = Controller()

    def move(self, dt=1.0/30.0):
        """Moves the object by a given time step.

        Args:
            dt: Length of time step.
        """
        super(Robot, self).move(dt)
        collided = self._sensors[0].has_collided()
        state = self._get_state()
        reward = [-10 if collided else -abs(self._angle_to_destination())]
        self._nn.train(state, reward)

    def _angle_to_destination(self):
        x, y = self._target[0] - self.x, self._target[1] - self.y
        return np.arctan2(y, x) - self.theta

    def _get_rotated_distances(self, rot):
        distances = self._sensors[0].distances
        rotated_distances = np.hstack([distances[-rot:], distances[:-rot]])
        return rotated_distances

    def _get_state(self, action=0, rotation=0):
        return np.hstack([[abs(action - self._angle_to_destination())],
                           self._get_rotated_distances(rotation)])

    def _control(self, state, t):
        """Returns the yaw given state.

        Args:
            state: State.
            t: Time.

        Returns:
            Yaw.
        """
        actions = [-np.pi / 4, -np.pi / 8, 0., np.pi / 8, np.pi / 4]
        rotations = [2, 1, 0, -1, -2]
        states = [
            self._get_state(actions[i], rotations[i])
            for i in range(len(actions))
        ]

        utilities = [
            (a, self._nn.evaluate(states[i]))
            for i, a in enumerate(actions)
        ]
        optimal_a, optimal_utility = max(utilities, key=lambda x: x[1][0])
        # if np.random.random_sample() > 0.8:
            # self._action_taken = actions[0]
        print(optimal_a, optimal_utility[0])
        return optimal_a


class Obstacle(MovingObject):

    """Obstacle."""

    def __init__(self, velocity, radius, bounds, height=1.0):
        """Constructs a Robot.

        Args:
            velocity: Velocity of the robot in the forward direction.
            radius: Radius of the obstacle.
        """
        data = DebugData()
        self._bounds = bounds
        self._radius = radius
        self._height = height
        center = [0, 0, height / 2 - 0.5]
        axis = [0, 0, 1]  # Upright cylinder.
        data.addCylinder(center, axis, height, radius)
        polydata = data.getPolyData()
        super(Obstacle, self).__init__(velocity, polydata)

    def _control(self, state, t):
        """Returns the yaw given state.

        Args:
            state: State.
            t: Time.

        Returns:
            Yaw.
        """
        x_min, x_max, y_min, y_max = self._bounds
        x, y, theta = state
        if x - self._radius <= x_min:
            return np.pi
        elif x + self._radius >= x_max:
            return np.pi
        elif y - self._radius <= y_min:
            return np.pi
        elif y + self._radius >= y_max:
            return np.pi
        return 0.
