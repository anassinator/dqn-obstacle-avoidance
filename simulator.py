# -*- coding: utf-8 -*-

from world import World
from PythonQt import QtGui
from director import applogic
from director import objectmodel as om
from director import visualization as vis
from director.consoleapp import ConsoleApp


class Simulator(object):

    """Simulator."""

    def __init__(self, world):
        """Constructs the simulator.

        Args:
            world: World.
        """
        self._world = world
        self._app = ConsoleApp()
        self._view = self._app.createView(useGrid=False)

        self._initialize()

    def _initialize(self):
        """Initializes the world."""
        # Add world to view.
        om.removeFromObjectModel(om.findObjectByName("world"))
        vis.showPolyData(self._world.to_polydata(), "world")

    def display(self):
        """Launches and displays the simulator."""
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout(widget)
        layout.addWidget(self._view)
        widget.showMaximized()

        # Set camera.
        applogic.resetCamera(viewDirection=[0.2, 0, -1])

        self._app.start()


if __name__ == "__main__":
    world = World(120, 100).add_obstacles()
    sim = Simulator(world)
    sim.display()
