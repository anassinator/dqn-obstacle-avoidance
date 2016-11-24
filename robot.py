# -*- coding: utf-8 -*-

from director import vtkAll as vtk
from director import ioUtils, filterUtils


class Robot(object):

    """Robot."""

    def __init__(self, scale=0.04, model="celica.obj"):
        """Constructs a Robot.

        Args:
            scale: Scale of the model.
            model: Object model to use.
        """
        t = vtk.vtkTransform()
        t.RotateZ(90)
        t.Scale(scale, scale, scale)
        polydata = ioUtils.readPolyData(model)
        self._polydata = filterUtils.transformPolyData(polydata, t)

    def to_polydata(self):
        """Converts robot to visualizable poly data."""
        return self._polydata
