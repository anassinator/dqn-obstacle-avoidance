# -*- coding: utf-8 -*-

import numpy as np
from world import World
from PythonQt import QtGui
from director import applogic
from moving_object import Robot
from director import vtkAll as vtk
from director import objectmodel as om
from director import visualization as vis
from director.consoleapp import ConsoleApp
from director.timercallback import TimerCallback
from director.debugVis import DebugData


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

    def _update_sensor(self, robot, sensor):
        """Updates sensor's rays."""
        d = DebugData()

        origin = np.array([robot.x, robot.y, 0])

        for result, intersection, ray in zip(sensor.results,
                                             sensor.intersections,
                                             sensor.rays):
            if result:
                d.addLine(origin, intersection, color=[1, 0, 0], radius=0.05)
            else:
                d.addLine(origin, origin + ray, color=[0, 0, 1], radius=0.05)

        vis.updatePolyData(d.getPolyData(), 'rays', colorByName='RGB255')

    def _update_locator(self):
        d = DebugData()

        d.addPolyData(self._world.to_polydata())

        for obstacle, frame in self._obstacles:
            d.addCylinder([obstacle.x, obstacle.y, obstacle.height / 2],
                          obstacle.axis, obstacle.height, obstacle.radius)

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
        self._timer = TimerCallback(targetFps=30)
        self._timer.callback = self.tick
        self._timer.start()

        self._app.start()

    def tick(self):
        """Update simulation clock."""
        for obstacle, frame in self._obstacles:
            obstacle.move()
            self._update_moving_object(obstacle, frame)

        self._update_locator()

        for robot, frame in self._robots:
            robot.move()
            robot.sensor.set_locator(self.locator)
            robot.update_sensor()
            self._update_moving_object(robot, frame)
            self._update_sensor(robot, robot.sensor)


if __name__ == "__main__":
    world = World(120, 100)
    sim = Simulator(world)
    sim.add_robot(Robot(world))
    for obstacle in world.generate_obstacles(moving_obstacle_ratio=0.0):
        sim.add_obstacle(obstacle)
    sim.run()
