from __future__ import division
import numpy as np
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl

__author__ = 'Stefan Ecklebe'
colors = ["r", "g", "b", "c", "m", "y", "k", "w"]


def create_colormap(cnt):
    """
    create a colormap containing cnt values
    :param cnt:
    :return:
    """
    col_map = pg.ColorMap(np.array([0, .5, 1]),
                          np.array([[0, 0, 1., 1.], [0, 1., 0, 1.], [1., 0, 0, 1.]]))
    indexes = np.linspace(0, 1, cnt)
    return col_map.map(indexes, mode="qcolor")


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


class DataPlot:
    """
    base class for all plotting related classes
    """
    def __init__(self, data):

        # just to be sure
        assert isinstance(data, list) or isinstance(data, EvalData)
        if isinstance(data, EvalData):
            data = [data]
        else:
            assert isinstance(data[0], EvalData)

        self._data = data
        self._dt = data[0].input_data[0][1] - data[0].input_data[0][0]


class PgDataPlot(DataPlot, QtCore.QObject):
    """
    base class for all pyqtgraph plotting related classes
    """
    def __init__(self, data):
        QtCore.QObject.__init__(self)
        DataPlot.__init__(self, data)


class PgAnimatedPlot(PgDataPlot):
    """
    wrapper that shows an updating one dimensional plot. of n-curves discretized in t time steps and z spatial steps
    It is assumed that time propagates along axis1 and and location along axis2 of values
    values are therefore expected to be a array of shape (n, t, z)
    playback set can be set via "dt" which is the real world step size. default is playback in realtime
    """
    # TODO generalize to n-d spatial domain

    def __init__(self, data, title="", dt=None):
        PgDataPlot.__init__(self, data)

        self._pw = pg.plot(title=title)
        self._pw.addLegend()
        self._pw.showGrid(x=True, y=True, alpha=0.5)

        time_data = [data_set.input_data[0] for data_set in self._data]
        max_times = [max(data) for data in time_data]
        self._longest_idx = max_times.index(max(max_times))
        if dt is not None:
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
            self._plot_data_items.append(pg.PlotDataItem(pen=colors[idx], name=data_set.name))
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


class PgSurfacePlot(PgDataPlot):
    """
    plot as 3d surface
    """
    def __init__(self, data):
        PgDataPlot.__init__(self, data)
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


class PgSlicePlot(PgDataPlot):
    """
    plot selected slice of given DataSets
    """
    # TODO think about a nice slice strategy see pyqtgraph for inspiration
    def __init__(self, data, title=None):
        PgDataPlot.__init__(self, data)
        self.dim = self._data[0].output_data.shape

        self.win = QtGui.QMainWindow()
        self.win.resize(800, 800)
        self.win.setWindowTitle("PgSlicePlot: {}".format(title))
        self.cw = QtGui.QWidget()
        self.win.setCentralWidget(self.cw)
        self.l = QtGui.QGridLayout()
        self.cw.setLayout(self.l)
        self.image_view = pg.ImageView(name="img_view")
        self.l.addWidget(self.image_view, 0, 0)
        self.slice_view = pg.PlotWidget(name="slice")
        self.l.addWidget(self.slice_view)
        self.win.show()

        # self.imv2 = pg.ImageView()
        # self.l.addWidget(self.imv2, 1, 0)

        self.roi = pg.LineSegmentROI([[0, self.dim[1]-1], [self.dim[0]-1, self.dim[1]-1]], pen='r')
        self.image_view.addItem(self.roi)
        self.image_view.setImage(self._data[0].output_data)
        #
        # self.plot_window.showGrid(x=True, y=True, alpha=.5)
        # self.plot_window.addLegend()
        #
        # input_idx = 0 if self.data_slice.shape[0] > self.data_slice.shape[1] else 0
        # for data_set in data:
        #     self.plot_window.plot(data_set.input_data[input_idx], data_set.output_data[self.data_slice],
        #                           name=data.name)


class PgLinePlot3d(PgDataPlot):
    """
    plots a series of n-lines of the systems state.
    scaling in z-direction can be changed with the scale setting
    """

    def __init__(self, data, n=50, scale=1):
        PgDataPlot.__init__(self, data)

        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.show()
        self.w.setWindowTitle(data[0].name)

        # grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.w.addItem(gz)

        res = self._data[0]
        z_vals = res.input_data[1][::-1]*scale

        t_subsets = np.linspace(0, res.input_data[0].size, n, endpoint=False, dtype=int)

        for t_idx, t_val in enumerate(t_subsets):
            t_vals = np.array([res.input_data[0][t_val]]*len(z_vals))
            pts = np.vstack([t_vals, z_vals, res.output_data[t_val, :]]).transpose()
            plt = gl.GLLinePlotItem(pos=pts, color=pg.glColor((t_idx, n*1.3)),
                                    # width=(t_idx + 1) / 10.,
                                    width=2,
                                    antialias=True)
            self.w.addItem(plt)


class MplSurfacePlot(DataPlot):
    """
    plot as 3d surface
    """
    def __init__(self, data, keep_aspect=True, show=False):
        DataPlot.__init__(self, data)

        for i in range(len(self._data)):

            disc = 3
            axespad = 0.02
            mpl.rcParams['font.size'] = 15
            mpl.rcParams['ytick.major.pad'] = 0
            mpl.rcParams['ytick.minor.pad'] = 0
            mpl.rcParams['axes.labelsize'] = 25
            x = self._data[i].input_data[1]
            y = self._data[i].input_data[0]
            z = (self._data[i].output_data)
            x_min, x_max = (int(x.min()), int(x.max()))
            y_min, y_max = (int(y.min()), int(y.max()))
            z_min, z_max = (z.min(), z.max())
            xx, yy = np.meshgrid(x, y)

            fig = plt.figure(figsize=(12, 12), facecolor='white')
            ax = fig.add_subplot(111, projection='3d')
            if keep_aspect:
                ax.set_aspect('equal', 'box')
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            # plt.title(r'$ $')
            ax.set_ylabel('\n'+r'$t$')
            ax.set_xlabel('\n'+r'$z$')
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel('\t'+r'$x(z,t)$', rotation=0)
            ax.set_xlim((x_min+0.025, x_max-axespad))
            plt.xticks(np.linspace(x_min, x_max, disc))
            ax.set_ylim((y_min+axespad, y_max-axespad))
            plt.yticks(np.linspace(y_min, y_max, disc))
            # # ax.w_xaxis.gridlines.set_lw(3.0)
            # # ax.w_yaxis.gridlines.set_lw(3.0)
            # # ax.w_zaxis.gridlines.set_lw(3.0)
            ax.w_xaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5)}})
            ax.w_yaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5)}})
            ax.w_zaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5)}})

            ax.plot_wireframe(xx, yy, z, rstride=2, cstride=4, color="#222222")
            # ax.view_init(elev=10, azim=-45)

        if show:
            plt.show()