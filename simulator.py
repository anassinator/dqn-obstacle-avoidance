# -*- coding: utf-8 -*-

import numpy as np
from world import World
from PythonQt import QtGui
from sensor import RaySensor
from director import applogic
from moving_object import Robot
from director import vtkAll as vtk
from director import objectmodel as om
from director.debugVis import DebugData
from director import visualization as vis
from director.consoleapp import ConsoleApp
from director.timercallback import TimerCallback


class Simulator(object):

    """Simulator."""

    def __init__(self, world):
        """Constructs the simulator.

        Args:
            world: World.
        """
        self._robots = []
        self._obstacles = []
        self._world = world
        self._app = ConsoleApp()
        self._view = self._app.createView(useGrid=False)

        self._initialize()

    def _initialize(self):
        """Initializes the world."""
        # Add world to view.
        om.removeFromObjectModel(om.findObjectByName("world"))
        vis.showPolyData(self._world.to_polydata(), "world")

    def _add_polydata(self, polydata, frame_name, color):
        """Adds polydata to the simulation.

        Args:
            polydata: Polydata.
            frame_name: Frame name.
            color: Color of object.

        Returns:
            Frame.
        """
        om.removeFromObjectModel(om.findObjectByName(frame_name))
        frame = vis.showPolyData(polydata, frame_name, color=color)

        vis.addChildFrame(frame)
        return frame

    def add_target(self, target):
        data = DebugData()
        center = [target[0], target[1], 1]
        axis = [0, 0, 1]  # Upright cylinder.
        data.addCylinder(center, axis, 2, 3)
        self._add_polydata(data.getPolyData(), "target", [0, 0.8, 0])

    def add_robot(self, robot):
        """Adds a robot to the simulation.

        Args:
            robot: Robot.
        """
        color = [0.4, 0.85098039, 0.9372549]
        frame_name = "robot{}".format(len(self._robots))
        frame = self._add_polydata(robot.to_polydata(), frame_name, color)
        self._robots.append((robot, frame))
        self._update_moving_object(robot, frame)

    def add_obstacle(self, obstacle):
        """Adds an obstacle to the simulation.

        Args:
            obstacle: Obstacle.
        """
        color = [1.0, 1.0, 1.0]
        frame_name = "obstacle{}".format(len(self._obstacles))
        frame = self._add_polydata(obstacle.to_polydata(), frame_name, color)
        self._obstacles.append((obstacle, frame))
        self._update_moving_object(obstacle, frame)

    def _update_moving_object(self, moving_object, frame):
        """Updates moving object's state.

        Args:
            moving_object: Moving object.
            frame: Corresponding frame.
        """
        t = vtk.vtkTransform()
        t.Translate(moving_object.x, moving_object.y, 0.0)
        t.RotateZ(np.degrees(moving_object.theta))
        frame.getChildFrame().copyFrame(t)

    def _update_sensor(self, sensor, frame_name):
        """Updates sensor's rays.

        Args:
            sensor: Sensor.
            frame_name: Frame name.
        """
        vis.updatePolyData(sensor.to_polydata(), frame_name,
                           colorByName="RGB255")

    def update_locator(self):
        """Updates cell locator."""
        d = DebugData()

        d.addPolyData(self._world.to_polydata())
        for obstacle, frame in self._obstacles:
            d.addPolyData(obstacle.to_positioned_polydata())

        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(d.getPolyData())
        self.locator.BuildLocator()

    def run(self):
        """Launches and displays the simulator."""
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout(widget)
        layout.addWidget(self._view)
        widget.showMaximized()

        # Set camera.
        applogic.resetCamera(viewDirection=[0.2, 0, -1])

        # Set timer.
        self._tick_count = 0
        self._timer = TimerCallback(targetFps=30)
        self._timer.callback = self.tick
        self._timer.start()

        self._app.start()

    def tick(self):
        """Update simulation clock."""
        self._tick_count += 1
        if self._tick_count >= 500:
            print("timeout")
            for robot, frame in self._robots:
                self.complete(robot, frame)

        need_update = False
        for obstacle, frame in self._obstacles:
            if obstacle.velocity != 0.:
                obstacle.move()
                self._update_moving_object(obstacle, frame)
                need_update = True

        if need_update:
            self.update_locator()

        for i, (robot, frame) in enumerate(self._robots):
            self._update_moving_object(robot, frame)
            for sensor in robot.sensors:
                sensor.set_locator(self.locator)
            robot.move()
            for sensor in robot.sensors:
                frame_name = "rays{}".format(i)
                self._update_sensor(sensor, frame_name)
                if sensor.has_collided():
                    print("collided", min(d for d in sensor._distances if d > 0))
                    self.reset(robot, frame)

            if robot.at_target():
                self.reset(robot, frame)

    def reset(self, robot, frame_name):
        self._tick_count = 0
        robot.x, robot.y, robot.theta = -30, -47, 0
        self._update_moving_object(robot, frame_name)
        robot._nn._nn.save()


if __name__ == "__main__":
    world = World(200, 200)
    sim = Simulator(world)
    for obstacle in world.generate_obstacles(0.01, moving_obstacle_ratio=0):
        sim.add_obstacle(obstacle)
    sim.update_locator()

    target = (40, 37)

    sim.add_target(target)
    robot = Robot(target=target)
    robot.x, robot.y = -30, -47
    robot.attach_sensor(RaySensor())
    sim.add_robot(robot)

    # robot._nn._nn.load()
    sim.run()
    robot._nn._nn.save()
