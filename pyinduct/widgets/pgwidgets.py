# coding=utf-8
import numpy as np
import copy as cp
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import matplotlib as mpl
import matplotlib.cm as cm
from ..core import EvalData
from ..utils import get_resource
from pyinduct.tests import show_plots
import matplotlib.pyplot as plt


def show(show_pg=True, show_mpl=True, force=False):
    """
    Shortcut to show all pyqtgraph and matplotlib plots / animations.

    Args:
        show_pg (bool): Show matplotlib plots? Default: True
        show_mpl (bool): Show pyqtgraph plots? Default: True
        force (bool): Show plots even during unittest discover, setup
            and so on? Default: False
    """
    if show_plots or force:
        if show_pg:
            pg.QtGui.QApplication.instance().exec_()

        if show_mpl:
            plt.show()


class DoubleSlider(pg.QtGui.QSlider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value


class AdSlider(pg.QtGui.QWidget):

    def __init__(self, parent=None):
        # TODO shortcuts for play/pause
        # TODO current time / total time
        # TODO edit widths
        super(AdSlider, self).__init__(parent)
        self.hBoxLayout = pg.QtGui.QHBoxLayout()
        self.playButton = pg.QtGui.QPushButton()
        self.playButton.setIcon(QIcon(get_resource("play.png")))
        self.playButton.setFixedHeight(50)
        self.playButton.setFixedWidth(50)
        self.pauseButton = pg.QtGui.QPushButton()
        self.pauseButton.setIcon(QIcon(get_resource("stop.png")))
        self.pauseButton.setFixedHeight(50)
        self.pauseButton.setFixedWidth(50)
        self.slider = DoubleSlider(Qt.Horizontal)
        self.textLabelCurrent = pg.QtGui.QLabel('100')
        self.textLabelCurrent.setFixedHeight(20)
        self.textLabel = pg.QtGui.QLabel(":")
        self.textLabel.setFixedHeight(20)
        self.textLabelTotal = pg.QtGui.QLabel("100")
        self.textLabelTotal.setFixedHeight(20)

        self.hBoxLayout.addWidget(self.playButton)
        self.hBoxLayout.addWidget(self.pauseButton)
        self.hBoxLayout.addWidget(self.slider)
        self.hBoxLayout.addWidget(self.textLabelCurrent)
        self.hBoxLayout.addWidget(self.textLabel)
        self.hBoxLayout.addWidget(self.textLabelTotal)
        self.hBoxLayout.setSizeConstraint(pg.QtGui.QLayout.SetMinimumSize)
        self.setLayout(self.hBoxLayout)


class DataPlot:
    """
    Base class for all plotting related classes.
    """

    def __init__(self, data):

        # just to be sure
        assert isinstance(data, list) or isinstance(data, EvalData)
        if isinstance(data, EvalData):
            data = [data]
        else:
            assert isinstance(data[0], EvalData)

        self._data = data
        # TODO Test input vectors to be Domain objects and use
        # their .step attribute here
        self._dt = data[0].input_data[0][1] - data[0].input_data[0][0]


class PgDataPlot(object):
    """
    Base class for all pyqtgraph plotting related classes.
    """
    windowWidth = 1024
    windowHeight = 768
    colorbarWidth = 60
    sliderHeight = 50

    def __init__(self, **kwargs):
        data = kwargs.get('data', None)
        DataPlot.__init__(self, data)
        pg.mkQApp()
        plotType = kwargs.get('plotType', '2D')
        self.colorMap = kwargs.get('colorMap', "viridis")

        self.plotWidget = None
        self.colorBar = None
        self.slider = None
        self.w = None

        if plotType == '2D':
            self.generate2DWindow()
        elif plotType == '3D':
            self.generate3DWindow()
        elif plotType == '2D-Animation':
            self.generate2DAnimaionWindow()
        elif plotType == '3D-Animation':
            self.generate3DAnimaionWindow()
        else:
            raise ValueError

    def generate2DWindow(self):
        layout = pg.QtGui.QGridLayout()
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.sizeHint = lambda: pg.QtCore.QSize(self.windowWidth, self.windowHeight)
        layout.addWidget(self.plotWidget, 0, 0)
        self.w = pg.QtGui.QWidget()
        self.w.resize(self.windowWidth, self.windowHeight)
        self.w.setLayout(layout)
        self.w.show()

    def generate3DWindow(self):
        layout = pg.QtGui.QGridLayout()
        self.plotWidget = PgAdvancedViewWidget()
        self.plotWidget.setCameraPosition(distance=3, azimuth=-135)
        self.colorBar = PgColorBarWidget(self.colorMap)
        self.plotWidget.setSizePolicy(self.colorBar.sizePolicy())
        layout.addWidget(self.plotWidget, 0, 0)
        layout.addWidget(self.colorBar, 0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnMinimumWidth(1, self.colorbarWidth)
        layout.setColumnStretch(0, 1)
        self.colorBar.sizeHint = lambda: pg.QtCore.QSize(self.colorbarWidth, self.windowHeight)
        layout.setHorizontalSpacing(0)
        self.w = pg.QtGui.QWidget()
        self.w.resize(self.windowWidth, self.windowHeight)
        self.w.setLayout(layout)
        self.w.show()

    def generate2DAnimaionWindow(self):
        layout = pg.QtGui.QGridLayout()
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.sizeHint = lambda: pg.QtCore.QSize(self.windowWidth, self.windowHeight)
        self.slider = AdSlider()
        layout.addWidget(self.plotWidget, 0, 0)
        layout.addWidget(self.slider, 1, 0)
        layout.setHorizontalSpacing(0)
        self.w = pg.QtGui.QWidget()
        self.w.resize(self.windowWidth, self.windowHeight)
        self.w.setLayout(layout)
        self.w.show()

    def generate3DAnimaionWindow(self):
        layout = pg.QtGui.QGridLayout()
        self.plotWidget = PgAdvancedViewWidget()
        self.plotWidget.setCameraPosition(distance=3, azimuth=-135)
        self.colorBar = PgColorBarWidget(self.colorMap)
        self.plotWidget.setSizePolicy(self.colorBar.sizePolicy())
        self.slider = AdSlider()
        layout.addWidget(self.plotWidget, 0, 0)
        layout.addWidget(self.colorBar, 0, 1)
        layout.addWidget(self.slider, 1, 0, 1, 2)
        layout.setColumnStretch(1, 0)
        layout.setColumnMinimumWidth(1, self.colorbarWidth)
        layout.setColumnStretch(0, 1)
        self.colorBar.sizeHint = lambda: pg.QtCore.QSize(self.colorbarWidth, self.windowHeight)
        layout.setHorizontalSpacing(0)
        self.w = pg.QtGui.QWidget()
        self.w.resize(self.windowWidth, self.windowHeight)
        self.w.setLayout(layout)
        self.w.show()


class PgAnimation(PgDataPlot):
    def __init__(self, **kwargs):
        PgDataPlot.__init__(self, **kwargs)
        refresh_time = kwargs.get('refresh_time', 40)
        replay_gain = kwargs.get('replay_gain', 1)
        axis = kwargs.get('animationAxis', 0)

        self.time_data = [np.atleast_1d(data_set.input_data[axis]) for data_set in self._data]
        min_times = [min(data) for data in self.time_data]
        max_times = [max(data) for data in self.time_data]

        self._start_time = min(min_times)
        self._end_time = max(max_times)
        self._longest_idx = max_times.index(self._end_time)

        # slider config
        self.autoPlay = True
        self.slider.playButton.setEnabled(False)
        self.slider.pauseButton.setEnabled(True)

        self.slider.textLabelTotal.setText(str(self._end_time))
        self.slider.textLabelCurrent.setFixedWidth(self.slider.textLabelTotal.width() + 5)
        self.slider.textLabelCurrent.setText(str(self._start_time))
        self.slider.slider.setMinimum(self._start_time)
        self.slider.slider.setMaximum(self._end_time)
        self.slider.slider.setValue(self._start_time)

        # buttons
        self.slider.playButton.clicked.connect(self.playAnimation)
        self.slider.pauseButton.clicked.connect(self.stopAnimation)

        assert refresh_time > 0
        self._tr = refresh_time
        assert replay_gain > 0
        self._t_step = self._tr / 1000 * replay_gain

        self._t = self._start_time

        self._timer = pg.QtCore.QTimer()
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(self._tr)

    # TODO Kopplung zum Slider herstellen, autostart aktivieren
    # TODO sliderReleased für user bearbeitung, und bild update
    # TODO make abstract implementation in special classes
    def _update_plot(self):
        pass

    def playAnimation(self):
        self.autoPlay = True
        self.slider.playButton.setEnabled(False)
        self.slider.pauseButton.setEnabled(True)

    def stopAnimation(self):
        self.autoPlay = False
        self.slider.playButton.setEnabled(True)
        self.slider.pauseButton.setEnabled(False)


class PgSurfacePlot(object):
    def __init__(self, **kwargs):
        animationAxis = kwargs.get('animationAxis', None)
        if animationAxis is not None:
            _PgSurfacePlotAnimation(**dict(kwargs, plotType='3D-Animation'))
        else:
            _PgSurfacePlot(**dict(kwargs, plotType='3D'))


class _PgSurfacePlot(PgDataPlot):
    def __init__(self, **kwargs):
        PgDataPlot.__init__(self, **kwargs)

        self.xlabel = kwargs.get('xlabel', 't')
        self.ylabel = kwargs.get('ylabel', 'z')
        self.zlabel = kwargs.get('zlabel', 'x(z,t)')
        scales = kwargs.get('scales', None)

        self.grid_size = 20

        # calculate minima and maxima
        extrema_list = []
        for data_set in self._data:
            _extrema_list = []
            for entry in data_set.input_data:
                _min_max = [min(entry), max(entry)]
                _extrema_list.append(_min_max)

            extrema_list.append(_extrema_list)

        extrema_arr = np.array(extrema_list)

        extrema = [np.min(extrema_arr[..., 0], axis=0),
                   np.max(extrema_arr[..., 1], axis=0)]

        self.extrema = np.hstack((
            extrema,
            ([min([data_set.min for data_set in self._data])],
             [max([data_set.max for data_set in self._data])])))

        self.deltas = np.diff(self.extrema, axis=0).squeeze()

        if scales is None:
            # scale all axes uniformly if no scales are given
            _scales = []
            for value in self.deltas:
                if np.isclose(value, 0):
                    _scales.append(1)
                else:
                    _scales.append(1 / value)
            self.scales = np.array(_scales)
        else:
            self.scales = scales

        # setup color map
        norm = mpl.colors.Normalize(vmin=self.extrema[0, -1],
                                    vmax=self.extrema[1, -1])
        self.mapping = cm.ScalarMappable(norm, self.colorMap)

        # add plots
        self.plot_items = []
        for idx, data_set in enumerate(self._data):
            plot_item = gl.GLSurfacePlotItem(x=self.scales[0] * np.atleast_1d(self._data[idx].input_data[0]),
                                             y=self.scales[1] * np.flipud(np.atleast_1d(
                                                 self._data[idx].input_data[1])),
                                             z=self.scales[2] * self._data[idx].output_data,
                                             colors=self.mapping.to_rgba(self._data[idx].output_data),
                                             computeNormals=False)

            self.plotWidget.addItem(plot_item)
            self.plot_items.append(plot_item)

        # setup grids
        self.sc_deltas = self.deltas * self.scales
        self._xygrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._xygrid.setSpacing(self.sc_deltas[0] / 10, self.sc_deltas[1] / 10, 0)
        self._xygrid.setSize(1.2 * self.sc_deltas[0], 1.2 * self.sc_deltas[1], 1)
        self._xygrid.translate(
            .5 * (self.extrema[1][0] + self.extrema[0][0]) * self.scales[0],
            .5 * (self.extrema[1][1] + self.extrema[0][1]) * self.scales[1],
            self.extrema[0][2] * self.scales[2] - 0.1 * self.sc_deltas[0]
        )
        self.plotWidget.addItem(self._xygrid)

        self._xzgrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._xzgrid.setSpacing(self.sc_deltas[0] / 10, self.sc_deltas[2] / 10, 0)
        self._xzgrid.setSize(1.2 * self.sc_deltas[0], 1.2 * self.sc_deltas[2], 1)
        self._xzgrid.rotate(90, 1, 0, 0)
        self._xzgrid.translate(
            .5 * (self.extrema[1][0] + self.extrema[0][0]) * self.scales[0],
            self.extrema[0][1] * self.scales[1] + 1.1 * self.sc_deltas[0],
            .5 * (self.extrema[1][2] + self.extrema[0][2]) * self.scales[2]
        )
        self.plotWidget.addItem(self._xzgrid)

        self._yzgrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._yzgrid.setSpacing(self.sc_deltas[1] / 10, self.sc_deltas[2] / 10, 0)
        self._yzgrid.setSize(1.2 * self.sc_deltas[1], 1.2 * self.sc_deltas[2], 1)
        self._yzgrid.rotate(90, 1, 0, 0)
        self._yzgrid.rotate(90, 0, 0, 1)
        self._yzgrid.translate(
            self.extrema[0][0] * self.scales[0] + 1.1 * self.sc_deltas[0],
            .5 * (self.extrema[1][1] + self.extrema[0][1]) * self.scales[1],
            .5 * (self.extrema[1][2] + self.extrema[0][2]) * self.scales[2]
        )
        self.plotWidget.addItem(self._yzgrid)

        # labels
        self.plotWidget.setXLabel(self.xlabel, pos=[
            self.extrema[0][0] * self.scales[0] + self.sc_deltas[0] - self.scales[0] * self.extrema[1][0],
            self.extrema[0][1] * self.scales[1] + 0.35 * self.sc_deltas[1] - self.scales[1] * self.extrema[1][1],
            self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] * self.extrema[1][2]])
        self.plotWidget.setYLabel(self.ylabel, pos=[
            self.extrema[0][0] * self.scales[0] + 0.35 * self.sc_deltas[0] - self.scales[0] * self.extrema[1][0],
            self.extrema[0][1] * self.scales[1] + self.sc_deltas[1] - self.scales[1] * self.extrema[1][1],
            self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] * self.extrema[1][2]])
        self.plotWidget.setZLabel(self.zlabel, pos=[
            self.extrema[0][0] * self.scales[0] + 1.6 * self.sc_deltas[0] - self.scales[0] * self.extrema[1][0],
            self.extrema[0][1] * self.scales[1] + 1.6 * self.sc_deltas[1] - self.scales[1] * self.extrema[1][1],
            self.extrema[0][2] * self.scales[2] + 1.6 * self.sc_deltas[2] - self.scales[2] * self.extrema[1][2]])

        # colorbar
        self.colorBar.setCBRange(self.extrema[0, -1], self.extrema[1, -1])
        # tics
        xTics = np.linspace(self.extrema[0, 1], self.extrema[1, 1], 6)
        yTics = np.linspace(self.extrema[0, 0], self.extrema[1, 0], 6)
        zTics = np.linspace(self.extrema[0, 2], self.extrema[1, 2], 6)
        posXTics = []
        posYTics = []
        posZTics = []
        for i, x in enumerate(np.linspace(
                        self.extrema[0][0] * self.scales[0] + 1.6 * self.sc_deltas[0] - self.scales[0] *
                self.extrema[1][0],
                        self.extrema[0][0] * self.scales[0] + 0.4 * self.sc_deltas[0] - self.scales[0] *
                            self.extrema[1][0],
                        13)):
            if i % 2 == 1:
                posXTics.append([self.extrema[0][1] * self.scales[1] + 0.35 * self.sc_deltas[1] - self.scales[1] *
                                 self.extrema[1][1],
                                 x,
                                 self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] *
                                 self.extrema[1][2]
                                 ])
        for i, y in enumerate(np.linspace(
                        self.extrema[0][1] * self.scales[1] + 0.4 * self.sc_deltas[1] - self.scales[1] *
                self.extrema[1][1],
                        self.extrema[0][1] * self.scales[1] + 1.6 * self.sc_deltas[1] - self.scales[1] *
                            self.extrema[1][1],
                        13)):
            if i % 2 == 1:
                posYTics.append([y,
                                 self.extrema[0][0] * self.scales[0] + 0.35 * self.sc_deltas[0] - self.scales[0] *
                                 self.extrema[1][0],
                                 self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] *
                                 self.extrema[1][2]])
        for i, z in enumerate(
            np.linspace(self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] *
                self.extrema[1][2],
                        self.extrema[0][2] * self.scales[2] + 1.6 * self.sc_deltas[2] - self.scales[2] *
                            self.extrema[1][2],
                        13)):
            if i % 2 == 1:
                posZTics.append([self.extrema[0][0] * self.scales[0] + 1.6 * self.sc_deltas[0] - self.scales[0] *
                                 self.extrema[1][0],
                                 self.extrema[0][1] * self.scales[1] + 1.6 * self.sc_deltas[1] - self.scales[1] *
                                 self.extrema[1][1],
                                 z])

        self.plotWidget.setXTics(xTics, posXTics)
        self.plotWidget.setYTics(yTics, posYTics)
        self.plotWidget.setZTics(zTics, posZTics)

        # set origin (zoom point) to the middle of the figure
        # (a better way would be to realize it directly via a method of
        # self.plotWidget, instead to shift all items)
        [item.translate(-self.scales[0] * self.extrema[1][0] + self.sc_deltas[0] / 2,
                        -self.scales[1] * self.extrema[1][1] + self.sc_deltas[1] / 2,
                        -self.scales[2] * self.extrema[1][2] + self.sc_deltas[2] / 2)
         for item in self.plotWidget.items]


class _PgSurfacePlotAnimation(PgAnimation):
    def __init__(self, **kwargs):
        PgAnimation.__init__(self, **kwargs)

        self.xlabel = kwargs.get('xlabel', 'x')
        self.ylabel = kwargs.get('ylabel', 'z')
        self.zlabel = kwargs.get('zlabel', 'y(x,z,t)')
        scales = kwargs.get('scales', None)
        animationAxis = kwargs.get('animationAxis', None)

        self.grid_size = 20

        # calculate minima and maxima
        extrema_list = []
        for data_set in self._data:
            _extrema_list = []
            for entry in data_set.input_data:
                _min_max = [min(entry), max(entry)]
                _extrema_list.append(_min_max)

            extrema_list.append(_extrema_list)

        extrema_arr = np.array(extrema_list)

        extrema = [np.min(extrema_arr[..., 0], axis=0),
                   np.max(extrema_arr[..., 1], axis=0)]

        self.extrema = np.hstack((
            extrema,
            ([min([data_set.min for data_set in self._data])],
             [max([data_set.max for data_set in self._data])])))

        self.deltas = np.diff(self.extrema, axis=0).squeeze()

        if scales is None:
            # scale all axes uniformly if no scales are given
            _scales = []
            for value in self.deltas:
                if np.isclose(value, 0):
                    _scales.append(1)
                else:
                    _scales.append(1 / value)
            self.scales = np.array(_scales)
        else:
            self.scales = scales

        # setup color map
        norm = mpl.colors.Normalize(vmin=self.extrema[0, -1],
                                    vmax=self.extrema[1, -1])
        self.mapping = cm.ScalarMappable(norm, self.colorMap)

        # add plots
        self.plot_items = []
        for idx, data_set in enumerate(self._data):
            if len(data_set.input_data) == 3:
                if animationAxis is None:
                    raise ValueError("animation_axis has to be provided.")

                # crop scale arrays
                if len(self.scales) != len(data_set.input_data):
                    # only remove time scaling if user provided one
                    self.scales = np.delete(self.scales, animationAxis)

                self.deltas = np.delete(self.deltas, animationAxis)
                self.extrema = np.delete(self.extrema, animationAxis, axis=1)

                # move animation axis to the end
                self._data[idx].input_data.append(self._data[idx].input_data.pop(animationAxis))
                self._data[idx].output_data = np.moveaxis(self._data[idx].output_data,
                                                          animationAxis,
                                                          -1)

                plot_item = gl.GLSurfacePlotItem(x=self.scales[0] * np.atleast_1d(self._data[idx].input_data[0]),
                                                 y=self.scales[1] * np.flipud(np.atleast_1d(
                                                     self._data[idx].input_data[1])),
                                                 z=self.scales[2] * self._data[idx].output_data[..., 0],
                                                 colors=self.mapping.to_rgba(self._data[idx].output_data[..., 0]),
                                                 computeNormals=False)

            self.plotWidget.addItem(plot_item)
            self.plot_items.append(plot_item)

        # setup grids
        self.sc_deltas = self.deltas * self.scales
        self._xygrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._xygrid.setSpacing(self.sc_deltas[0] / 10, self.sc_deltas[1] / 10, 0)
        self._xygrid.setSize(1.2 * self.sc_deltas[0], 1.2 * self.sc_deltas[1], 1)
        self._xygrid.translate(
            .5 * (self.extrema[1][0] + self.extrema[0][0]) * self.scales[0],
            .5 * (self.extrema[1][1] + self.extrema[0][1]) * self.scales[1],
            self.extrema[0][2] * self.scales[2] - 0.1 * self.sc_deltas[0]
        )
        self.plotWidget.addItem(self._xygrid)

        self._xzgrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._xzgrid.setSpacing(self.sc_deltas[0] / 10, self.sc_deltas[2] / 10, 0)
        self._xzgrid.setSize(1.2 * self.sc_deltas[0], 1.2 * self.sc_deltas[2], 1)
        self._xzgrid.rotate(90, 1, 0, 0)
        self._xzgrid.translate(
            .5 * (self.extrema[1][0] + self.extrema[0][0]) * self.scales[0],
            self.extrema[0][1] * self.scales[1] + 1.1 * self.sc_deltas[0],
            .5 * (self.extrema[1][2] + self.extrema[0][2]) * self.scales[2]
        )
        self.plotWidget.addItem(self._xzgrid)

        self._yzgrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._yzgrid.setSpacing(self.sc_deltas[1] / 10, self.sc_deltas[2] / 10, 0)
        self._yzgrid.setSize(1.2 * self.sc_deltas[1], 1.2 * self.sc_deltas[2], 1)
        self._yzgrid.rotate(90, 1, 0, 0)
        self._yzgrid.rotate(90, 0, 0, 1)
        self._yzgrid.translate(
            self.extrema[0][0] * self.scales[0] + 1.1 * self.sc_deltas[0],
            .5 * (self.extrema[1][1] + self.extrema[0][1]) * self.scales[1],
            .5 * (self.extrema[1][2] + self.extrema[0][2]) * self.scales[2]
        )
        self.plotWidget.addItem(self._yzgrid)

        # labels
        self.plotWidget.setXLabel(self.xlabel, pos=[
            self.extrema[0][0] * self.scales[0] + self.sc_deltas[0] - self.scales[0] * self.extrema[1][0],
            self.extrema[0][1] * self.scales[1] + 0.35 * self.sc_deltas[1] - self.scales[1] * self.extrema[1][1],
            self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] * self.extrema[1][2]])
        self.plotWidget.setYLabel(self.ylabel, pos=[
            self.extrema[0][0] * self.scales[0] + 0.35 * self.sc_deltas[0] - self.scales[0] * self.extrema[1][0],
            self.extrema[0][1] * self.scales[1] + self.sc_deltas[1] - self.scales[1] * self.extrema[1][1],
            self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] * self.extrema[1][2]])
        self.plotWidget.setZLabel(self.zlabel, pos=[
            self.extrema[0][0] * self.scales[0] + 1.6 * self.sc_deltas[0] - self.scales[0] * self.extrema[1][0],
            self.extrema[0][1] * self.scales[1] + 1.6 * self.sc_deltas[1] - self.scales[1] * self.extrema[1][1],
            self.extrema[0][2] * self.scales[2] + 1.6 * self.sc_deltas[2] - self.scales[2] * self.extrema[1][2]])

        # colorbar
        self.colorBar.setCBRange(self.extrema[0, -1], self.extrema[1, -1])
        # tics
        xTics = np.linspace(self.extrema[0, 1], self.extrema[1, 1], 6)
        yTics = np.linspace(self.extrema[0, 0], self.extrema[1, 0], 6)
        zTics = np.linspace(self.extrema[0, 2], self.extrema[1, 2], 6)
        posXTics = []
        posYTics = []
        posZTics = []
        for i, x in enumerate(np.linspace(
                        self.extrema[0][0] * self.scales[0] + 1.6 * self.sc_deltas[0] - self.scales[0] *
                self.extrema[1][0],
                        self.extrema[0][0] * self.scales[0] + 0.4 * self.sc_deltas[0] - self.scales[0] *
                self.extrema[1][0],
            13)):
            if i % 2 == 1:
                posXTics.append([self.extrema[0][1] * self.scales[1] + 0.35 * self.sc_deltas[1] - self.scales[1] *
                                 self.extrema[1][1],
                                 x,
                                 self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] *
                                 self.extrema[1][2]
                                 ])
        for i, y in enumerate(np.linspace(
                        self.extrema[0][1] * self.scales[1] + 0.4 * self.sc_deltas[1] - self.scales[1] *
                self.extrema[1][1],
                        self.extrema[0][1] * self.scales[1] + 1.6 * self.sc_deltas[1] - self.scales[1] *
                self.extrema[1][1],
            13)):
            if i % 2 == 1:
                posYTics.append([y,
                                 self.extrema[0][0] * self.scales[0] + 0.35 * self.sc_deltas[0] - self.scales[0] *
                                 self.extrema[1][0],
                                 self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] *
                                 self.extrema[1][2]])
        for i, z in enumerate(
            np.linspace(self.extrema[0][2] * self.scales[2] + 0.4 * self.sc_deltas[2] - self.scales[2] *
                self.extrema[1][2],
                        self.extrema[0][2] * self.scales[2] + 1.6 * self.sc_deltas[2] - self.scales[2] *
                            self.extrema[1][2],
                        13)):
            if i % 2 == 1:
                posZTics.append([self.extrema[0][0] * self.scales[0] + 1.6 * self.sc_deltas[0] - self.scales[0] *
                                 self.extrema[1][0],
                                 self.extrema[0][1] * self.scales[1] + 1.6 * self.sc_deltas[1] - self.scales[1] *
                                 self.extrema[1][1],
                                 z])

        self.plotWidget.setXTics(xTics, posXTics)
        self.plotWidget.setYTics(yTics, posYTics)
        self.plotWidget.setZTics(zTics, posZTics)

        # set origin (zoom point) to the middle of the figure
        # (a better way would be to realize it directly via a method of
        # self.plotWidget, instead to shift all items)
        [item.translate(-self.scales[0] * self.extrema[1][0] + self.sc_deltas[0] / 2,
                        -self.scales[1] * self.extrema[1][1] + self.sc_deltas[1] / 2,
                        -self.scales[2] * self.extrema[1][2] + self.sc_deltas[2] / 2)
         for item in self.plotWidget.items]

    def _update_plot(self):
        """
        Update the rendering
        """
        if self.autoPlay:
            for idx, item in enumerate(self.plot_items):
                # find nearest time index (0th order interpolation)
                t_idx = (np.abs(self.time_data[idx] - self._t)).argmin()

                # update data
                self.slider.textLabelCurrent.setText(str(self._t))
                self.slider.slider.setValue(self._t)
                z_data = self.scales[2] * self._data[idx].output_data[..., t_idx]
                mapped_colors = self.mapping.to_rgba(self._data[idx].output_data[..., t_idx])
                item.setData(z=z_data, colors=mapped_colors)

            self._t += self._t_step

            if self._t > self._end_time:
                self._t = self._start_time


class Pg2DPlot(object):
    def __init__(self, **kwargs):
        animationAxis = kwargs.get('animationAxis', None)
        if animationAxis is not None:
            _Pg2DPlotAnimation(**dict(kwargs, plotType='2D-Animation'))
        else:
            _Pg2DPlot(**dict(kwargs, plotType='2D'))


class _Pg2DPlot(PgDataPlot):
    def __init__(self, **kwargs):
        PgDataPlot.__init__(self, **kwargs)

        self.xData = [np.atleast_1d(data_set.input_data[0]) for data_set in self._data]
        self.yData = [data_set.output_data for data_set in self._data]

        self.plotWidget.showGrid(x=True, y=True, alpha=0.5)
        self.plotWidget.addLegend()

        xData_min = np.nanmin([np.nanmin(data) for data in self.xData])
        xData_max = np.nanmax([np.nanmax(data) for data in self.xData])
        self.plotWidget.setXRange(xData_min, xData_max)

        yData_min = np.nanmin([np.nanmin(data) for data in self.yData])
        yData_max = np.nanmax([np.nanmax(data) for data in self.yData])
        self.plotWidget.setYRange(yData_min, yData_max)

        self._plot_data_items = []
        colorMap = cm.get_cmap(self.colorMap)
        for idx, data_set in enumerate(self._data):
            self._plot_data_items.append(pg.PlotDataItem(pen=pg.mkPen(colorMap(idx / len(self._data), bytes=True),
                                                                      width=2), name=data_set.name))
            self.plotWidget.addItem(self._plot_data_items[-1])
            self._plot_data_items[-1].setData(x=self.xData[idx], y=self.yData[idx])


class _Pg2DPlotAnimation(PgAnimation):
    def __init__(self, **kwargs):
        PgAnimation.__init__(self, **kwargs)

        self.spatial_data = [np.atleast_1d(data_set.input_data[1]) for data_set in self._data]
        self.state_data = [data_set.output_data for data_set in self._data]

        spat_min = np.min([np.min(data) for data in self.spatial_data])
        spat_max = np.max([np.max(data) for data in self.spatial_data])
        self.plotWidget.setXRange(spat_min, spat_max)

        state_min = np.min([np.min(data) for data in self.state_data])
        state_max = np.max([np.max(data) for data in self.state_data])
        self.plotWidget.setYRange(state_min, state_max)

        self._plot_data_items = []
        self._plot_indexes = []
        colorMap = cm.get_cmap(self.colorMap)
        for idx, data_set in enumerate(self._data):
            self._plot_data_items.append(pg.PlotDataItem(pen=pg.mkPen(colorMap(idx / len(self._data), bytes=True),
                                                                      width=2), name=data_set.name))
            self.plotWidget.addItem(self._plot_data_items[-1])

    def _update_plot(self):
        """
        Update the rendering
        """
        if self.autoPlay:
            for idx, item in enumerate(self._data):
                # find nearest time index (0th order interpolation)
                t_idx = (np.abs(self.time_data[idx] - self._t)).argmin()

                # update data
                self.slider.textLabelCurrent.setText(str(self._t))
                self.slider.slider.setValue(self._t)
                self._plot_data_items[idx].setData(x=self.spatial_data[idx], y=self.state_data[idx][t_idx])

            self._t += self._t_step

            if self._t > self._end_time:
                self._t = self._start_time


class PgGradientWidget(pg.GraphicsWidget):
    """
    OpenGL Widget that depends on GraphicsWidget and realizes a color bar that depends on a QGraphicsRectItem and
    QLinearGradient

    Args:
        cmap (matplotlib.cm.Colormap): color map, if None viridis is used
    """

    def __init__(self, colorMap=None):
        pg.GraphicsWidget.__init__(self)
        self.length = 100
        self.maxDim = 20
        self.steps = 11
        self.rectSize = 15
        self._min = 0
        self._max = 1

        if colorMap is None:
            self.colorMap = cm.get_cmap('viridis')
        else:
            self.colorMap = colorMap

        self.gradRect = pg.QtGui.QGraphicsRectItem(pg.QtCore.QRectF(0, 0, self.length, self.rectSize))
        self.gradRect.setParentItem(self)

        self.setMaxDim(self.rectSize)
        self.resetTransform()
        transform = pg.QtGui.QTransform()
        transform.rotate(270)
        transform.translate(-self.height(), 0)
        self.setTransform(transform)
        self.translate(0, self.rectSize)

        self.updateGradient()

    def resizeEvent(self, ev):
        wlen = max(40, self.height())
        self.setLength(wlen)
        self.setMaxDim(self.rectSize)
        self.resetTransform()
        transform = pg.QtGui.QTransform()
        transform.rotate(270)
        transform.translate(-self.height(), 0)
        self.setTransform(transform)
        self.translate(0, self.rectSize)

    def setMaxDim(self, mx=None):
        """
        Sets the maximal width of the color bar widget

        Args:
             mx (float or None): new width of the color bar widget
        """
        if mx is None:
            mx = self.maxDim
        else:
            self.maxDim = mx

        self.setFixedWidth(mx)
        self.setMaximumHeight(2 ** 32)

    def setLength(self, newLen):
        """
        Gets the new length if the window is resize, updates the size and color of QGraphicsRectItem

        Args:
             newLen: new length of the window
        """
        self.length = float(newLen)
        self.gradRect.setRect(1, -self.rectSize, newLen, self.rectSize)
        self.updateGradient()

    def updateGradient(self):
        """
        Updates QGraphicsRectItem color with the gradient
        """
        self.gradRect.setBrush(pg.QtGui.QBrush(self.getGradient()))

    def getGradient(self):
        """
        Calculates for minimal and maximal value the linear gradient and assigns colors to the gradient.

        Return:
            pg.QtGui.QLinearGradient: linear gradient for current min and max values
        """
        norm = mpl.colors.Normalize(vmin=self._min, vmax=self._max)
        m = cm.ScalarMappable(norm=norm, cmap=self.colorMap)

        g = pg.QtGui.QLinearGradient(pg.QtCore.QPointF(0, 0), pg.QtCore.QPointF(self.length, 0))

        t = np.linspace(0, 1, self.steps)
        steps = np.linspace(self._min, self._max, self.steps)
        stops = []
        for idx in range(len(t)):
            _r, _g, _b, _a = m.to_rgba(steps[idx], bytes=True)
            qcol = pg.QtGui.QColor(_r, _g, _b, _a)
            stops.append(tuple([t[idx], qcol]))

        g.setStops(stops)

        return g

    def setRange(self, _min, _max):
        """
        Method updates the min and max value of the colar bar and calculates the new gradient trend

        Args:
            _min (float): set the minimal value of the color bar
            _max (float): set the maximal value of the color bar
        """
        self._min = _min
        self._max = _max
        self.updateGradient()


class PgAdvancedViewWidget(gl.GLViewWidget):
    """
    OpenGL Widget that depends on GLViewWidget and completes it with text labels for the x, y and z axis
    """

    def __init__(self):
        super(PgAdvancedViewWidget, self).__init__()
        self.xlabel = 'x'
        self.posXLabel = [1, 0, 0]

        self.ylabel = 'y'
        self.posYLabel = [0, 1, 0]

        self.zlabel = 'z'
        self.posZLabel = [0, 0, 1]
        self.oldPosZLabel = cp.copy(self.posZLabel)

        self.posXTics = []
        self.posYTics = []
        self.posZTics = []
        self.xTics = np.linspace(0, 1, 6)
        self.yTics = np.linspace(0, 1, 6)
        self.zTics = np.linspace(0, 1, 6)
        for i, pos in enumerate(np.linspace(0, 1, 13)):
            if i % 2 == 1:
                self.posXTics.append([pos, 0, 0])
                self.posYTics.append([0, pos, 0])
                self.posZTics.append([0, 0, pos])

        self.showAllTics = False
        self.shortcut = QShortcut(QKeySequence("a"), self)
        self.shortcut.activated.connect(self.showTics)

    def showTics(self):
        self.showAllTics = not self.showAllTics
        self.update()

    def setXTics(self, xTics, pos):
        """
        Sets x tics on positions

        Args:
            xTics (list): x tics text to render
            pos (list): position as list with [[x, y, z], [x, y, z], ...] coordinate
        """
        if len(xTics) != len(pos):
            raise ValueError('Lists must have the same size!')
        self.xTics = xTics
        self.posXTics = pos
        self.update()

    def setYTics(self, yTics, pos):
        """
        Sets y tics on positions

        Args:
            yTics (list): y tics text to render
            pos (list): position as list with [[x, y, z], [x, y, z], ...] coordinate
        """
        if len(yTics) != len(pos):
            raise ValueError('Lists must have the same size!')
        self.yTics = yTics
        self.posYTics = pos
        self.update()

    def setZTics(self, zTics, pos):
        """
        Sets z tics on positions

        Args:
            zTics (list): z tics text to render
            pos (list): position as list with [[x, y, z], [x, y, z], ...] coordinate
        """
        if len(zTics) != len(pos):
            raise ValueError('Lists must have the same size!')
        self.zTics = zTics
        self.posZTics = pos
        self.update()

    def setXLabel(self, text, pos):
        """
        Sets x label on position

        Args:
            text (str): x axis text to render
            pos (list): position as list with [x, y, z] coordinate
        """
        self.xlabel = text
        self.posXLabel = pos
        self.update()

    def setYLabel(self, text, pos):
        """
        Sets y label on position

        Args:
            text (str): y axis text to render
            pos (list): position as list with [x, y, z] coordinate
        """
        self.ylabel = text
        self.posYLabel = pos
        self.update()

    def setZLabel(self, text, pos):
        """
        Sets z label on position

        Args:
            text (str): z axis text to render
            pos (list): position as list with [x, y, z] coordinate
        """
        self.zlabel = text
        self.posZLabel = pos
        self.update()

    def paintGL(self, *args, **kwds):
        """
        Overrides painGL function to render the text labels
        """
        gl.GLViewWidget.paintGL(self, *args, **kwds)
        self.renderText(self.posXLabel[0],
                        self.posXLabel[1],
                        self.posXLabel[2],
                        self.xlabel)
        self.renderText(self.posYLabel[0],
                        self.posYLabel[1],
                        self.posYLabel[2],
                        self.ylabel)

        self.renderText(self.oldPosZLabel[0],
                        self.oldPosZLabel[1],
                        self.oldPosZLabel[2],
                        " " * 20)
        self.renderText(self.posZLabel[0],
                        self.posZLabel[1],
                        self.posZLabel[2],
                        self.zlabel)
        self.oldPosZLabel = cp.copy(self.posZLabel)

        if self.showAllTics:
            for i in range(len(self.xTics)):
                self.renderText(self.posXTics[i][0],
                                self.posXTics[i][1],
                                self.posXTics[i][2],
                                '{:.1f}'.format(self.xTics[i]))
            for i in range(len(self.yTics)):
                self.renderText(self.posYTics[i][0],
                                self.posYTics[i][1],
                                self.posYTics[i][2],
                                '{:.1f}'.format(self.yTics[i]))
            for i in range(len(self.zTics)):
                self.renderText(self.posZTics[i][0],
                                self.posZTics[i][1],
                                self.posZTics[i][2],
                                '{:.1f}'.format(self.zTics[i]))
        else:
            for i in range(len(self.xTics)):
                self.renderText(self.posXTics[i][0],
                                self.posXTics[i][1],
                                self.posXTics[i][2],
                                str())
            for i in range(len(self.yTics)):
                self.renderText(self.posYTics[i][0],
                                self.posYTics[i][1],
                                self.posYTics[i][2],
                                str())
            for i in range(len(self.zTics)):
                self.renderText(self.posZTics[i][0],
                                self.posZTics[i][1],
                                self.posZTics[i][2],
                                str())


class PgColorBarWidget(pg.GraphicsLayoutWidget):
    """
    OpenGL Widget that depends on GraphicsLayoutWidget and realizes an axis and a color bar
    """

    def __init__(self, colorMap):
        super(PgColorBarWidget, self).__init__()

        _min = 0
        _max = 1

        # values axis
        self.ax = pg.AxisItem('left')
        self.ax.setRange(_min, _max)
        self.addItem(self.ax)
        # color bar gradients
        self.gw = PgGradientWidget(colorMap=cm.get_cmap(colorMap))
        self.setCBRange(_min, _max)
        self.addItem(self.gw)

    def setCBRange(self, _min, _max):
        """
        Sets the minimal and maximal value of axis and color bar

        Args:
            _min (float): minimal value
            _max (float): maximal value
        """
        self.gw.setRange(_min, _max)
        self.ax.setRange(_min, _max)

