# -*- coding: utf-8 -*-

import numpy as np
from robot import Robot
from world import World
from PythonQt import QtGui
from director import applogic
from director import vtkAll as vtk
from director import objectmodel as om
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
        self._world = world
        self._app = ConsoleApp()
        self._view = self._app.createView(useGrid=False)

        self._initialize()

    def _initialize(self):
        """Initializes the world."""
        # Add world to view.
        om.removeFromObjectModel(om.findObjectByName("world"))
        vis.showPolyData(self._world.to_polydata(), "world")

    def add_robot(self, robot):
        """Adds a robot to the simulation.

        Args:
            robot: Robot.
        """
        robot_color = [0.4, 0.85098039, 0.9372549]
        frame_name = "robot{}".format(len(self._robots))
        om.removeFromObjectModel(om.findObjectByName(frame_name))
        frame = vis.showPolyData(robot.to_polydata(), frame_name,
                                 color=robot_color)

        self._robots.append((robot, frame))
        vis.addChildFrame(frame)
        self._update_robot(robot, frame)

    def _update_robot(self, robot, frame):
        """Updates robot's state.

        Args:
            robot: Robot.
            frame: Corresponding frame.
        """
        t = vtk.vtkTransform()
        t.Translate(robot.x, robot.y, 0.0)
        t.RotateZ(np.degrees(robot.theta))
        frame.getChildFrame().copyFrame(t)

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
        for robot, frame in self._robots:
            robot.move()
            self._update_robot(robot, frame)


if __name__ == "__main__":
    world = World(120, 100).add_obstacles()
    sim = Simulator(world)
    sim.add_robot(Robot())
    sim.run()
