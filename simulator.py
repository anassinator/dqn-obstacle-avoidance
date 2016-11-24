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

    def __init__(self, robot, world):
        """Constructs the simulator.

        Args:
            robot: Robot.
            world: World.
        """
        self._robot = robot
        self._world = world
        self._app = ConsoleApp()
        self._view = self._app.createView(useGrid=False)
        self._robot_frame = None

        self._initialize()

    def _initialize(self):
        """Initializes the world."""
        # Add world to view.
        om.removeFromObjectModel(om.findObjectByName("world"))
        vis.showPolyData(self._world.to_polydata(), "world")

        # Add robot to view.
        robot_color = [0.4, 0.85098039, 0.9372549]
        om.removeFromObjectModel(om.findObjectByName("robot"))
        self._robot_frame = vis.showPolyData(self._robot.to_polydata(),
                                             "robot", color=robot_color)
        vis.addChildFrame(self._robot_frame)
        self._update_robot()

    def _update_robot(self):
        """Updates robot's state."""
        t = vtk.vtkTransform()
        t.Translate(self._robot.x, self._robot.y, 0.0)
        t.RotateZ(np.degrees(self._robot.theta))
        self._robot_frame.getChildFrame().copyFrame(t)

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
        self._robot.move()
        self._update_robot()

if __name__ == "__main__":
    robot = Robot()
    world = World(120, 100).add_obstacles()
    sim = Simulator(robot, world)
    sim.run()
