from __future__ import division
import numpy as np
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

__author__ = 'stefan'

class EvalData:
    """
    convenience wrapper for function evaluation
    contains the input data that was used for evaluation and the results
    """

    def __init__(self, input_data, output_data):
        # check type and dimensions
        assert isinstance(input_data, list)
        assert isinstance(output_data, np.ndarray)
        assert len(input_data) == len(output_data.shape)

        for dim in range(len(output_data.shape)):
            assert len(input_data[dim]) == output_data.shape[dim]

        self.input_data = input_data
        self.output_data = output_data


class DataPlot(QtCore.QObject):
    """
    base class for all plotting related classes
    """
    def __init__(self, data):
        QtCore.QObject.__init__(self)

        # just to be sure
        assert isinstance(data, list) or isinstance(data, EvalData)
        if isinstance(data, EvalData):
            data = [data]
        else:
            assert isinstance(data[0], EvalData)

        self._data = data

class AnimatedPlot(DataPlot):
    """
    wrapper that shows an updating one dimensional plot. of n-curves dicretized in t time steps and z spatial steps
    It is assumed that time propagates along axis1 and and location along axis2 of values
    values are therefore expected to be a array of shape (n, t, z)
    """
    # TODO generalize to n-d spatial domain
    def __init__(self, data, title="", dt=1e-2):
        DataPlot.__init__(self, data)

        self._dt = dt
        self._pw = pg.plot(title=title)
        time_data = [data_set.input_data[0] for data_set in self._data]
        spatial_data = [data_set.input_data[1] for data_set in self._data]
        state_data = [data_set.output_data for data_set in self._data]

        spat_min = np.min([np.min(data) for data in spatial_data])
        spat_max = np.max([np.max(data) for data in spatial_data])
        self._pw.setXRange(spat_min, spat_max)
        state_min = np.min([np.min(data) for data in state_data])
        state_max = np.max([np.max(data) for data in state_data])
        self._pw.setYRange(state_min, state_max)
        self._time_text = pg.TextItem('t= 0')
        self._pw.addItem(self._time_text)
        self._time_text.setPos(.9*spat_max, .9*state_min)

        self._curr_frame = 0
        self._timer = pg.QtCore.QTimer()
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(1e3*self._dt)

    def _update_plot(self):
        """
        update plot window
        """
        colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
        for idx, data_set in enumerate(self._data):
            if idx == 0:
                clear = True
            else:
                clear = False

            self._pw.plot(x=data_set.input_data[1], y=data_set.output_data[self._curr_frame],
                          clear=clear, pen=colors[idx])
            self._time_text.setText('t= {0:.2f}'.format(data_set.input_data[0][self._curr_frame]))
            self._pw.addItem(self._time_text)

        if self._curr_frame == self._data[0].output_data.shape[0] - 1:
            # end of time reached -> start again
            self._curr_frame = 0
        else:
            self._curr_frame += 1


class SurfacePlot(DataPlot):
    """
    plot as 3d surface
    """
    def __init__(self, data):
        DataPlot.__init__(self, data)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.show()

        self._grid = gl.GLGridItem()
        self._grid.scale(2, 2, 1)
        self._grid.setDepthValue(10)
        self.gl_widget.addItem(self._grid)

        for n in range(len(self._data)):
            plot_item = gl.GLSurfacePlotItem(x=self._data[n].input_data[0],
                                             y=self._data[n].input_data[1],
                                             z=self._data[n].output_data,
                                             shader='normalColor')
            self.gl_widget.addItem(plot_item)
