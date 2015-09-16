from __future__ import division
import numpy as np
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

__author__ = 'Stefan Ecklebe'
colors = ["r", "g", "b", "c", "m", "y", "k", "w"]


class EvalData:
    """
    convenience wrapper for function evaluation
    contains the input data that was used for evaluation and the results
    """

    def __init__(self, input_data, output_data, name=""):
        # check type and dimensions
        assert isinstance(input_data, list)
        assert isinstance(output_data, np.ndarray)
        assert len(input_data) == len(output_data.shape)

        for dim in range(len(output_data.shape)):
            assert len(input_data[dim]) == output_data.shape[dim]

        self.input_data = input_data
        self.output_data = output_data
        self.name = name

        self.name = name


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
    wrapper that shows an updating one dimensional plot. of n-curves discretized in t time steps and z spatial steps
    It is assumed that time propagates along axis1 and and location along axis2 of values
    values are therefore expected to be a array of shape (n, t, z)
    playback set can be set via "dt" which is the real world step size. default is playback in realtime
    """
    # TODO generalize to n-d spatial domain

    def __init__(self, data, title="", dt=None):
        DataPlot.__init__(self, data)

        self._pw = pg.plot(title=title)
        self._pw.addLegend()
        self._pw.showGrid(x=True, y=True, alpha=0.5)

        time_data = [data_set.input_data[0] for data_set in self._data]
        max_times = [max(data) for data in time_data]
        self._longest_idx = max_times.index(max(max_times))
        if dt is None:
            self._dt = time_data[0][1] - time_data[0][0]
        else:
            self._dt = dt

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

        self._plot_data_items = []
        for idx, data_set in enumerate(self._data):
            self._plot_data_items.append(pg.PlotDataItem(pen=self.colors[idx], name=data_set.name))
            self._pw.addItem(self._plot_data_items[-1])

        self._curr_frame = 0
        self._timer = pg.QtCore.QTimer()
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(1e3*self._dt)

    def _update_plot(self):
        """
        update plot window
        """
        for idx, data_set in enumerate(self._data):
            frame = min(self._curr_frame, data_set.output_data.shape[0]-1)
            self._plot_data_items[idx].setData(x=data_set.input_data[1], y=data_set.output_data[frame])

        self._time_text.setText('t= {0:.2f}'.format(self._data[self._longest_idx].input_data[0][frame]))
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


class SlicePlot(DataPlot):
    """
    plot selected slice of given DataSets
    """
    # TODO think about a nice slice strategy see pyqtgraph for inspiration
    def __init__(self, data, data_slice, title=None):
        DataPlot.__init__(self, data)

        self.data_slice = data_slice
        self.title = title

        self.plot_window = pg.plot(title=title)
        self.plot_window.showGrid(x=True, y=True, alpha=.5)
        self.plot_window.addLegend()

        input_idx = 0 if self.data_slice.shape[0] > self.data_slice.shape[1] else 0
        for data_set in data:
            self.plot_window.plot(data_set.input_data[input_idx], data_set.output_data[self.data_slice],
                                  name=data.name)
