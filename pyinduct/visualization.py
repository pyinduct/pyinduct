from __future__ import division
import numpy as np
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
from numbers import Number
from types import NoneType
import time
import pyqtgraph.opengl as gl
import scipy.interpolate as si
# axes3d not explicit used but needed
from mpl_toolkits.mplot3d import axes3d


colors = ["g", "c", "m", "b", "y", "k", "w", "r"]

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

    # TODO: add scipy n-D-interp function, min+max

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

    def interpolation_handle(self, desired_coordinates):
        return si.interpn(tuple(self.input_data),
                          self.output_data,
                          tuple(desired_coordinates))


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

    # TODO default realtime, kwarg: T

    def __init__(self, data, title="", dt=None):
        PgDataPlot.__init__(self, data)

        # TODO: choose not simply the discretisation from the first ut.EvalData object but from that with the
        interp_funcs = [si.interp2d(data.input_data[1], data.input_data[0], data.output_data) for data in self._data]
        time_data = [self._data[0].input_data[0] for _ in self._data]
        spatial_data = [self._data[0].input_data[1] for _ in self._data]
        state_data = [interp_func(spatial_data[0], time_data[0]) for interp_func in interp_funcs]

        # TODO: remove the next 4 lines and replace (from here) self._data with state_data, time_data and spatial_data
        for i in range(len(self._data)):
            self._data[i].input_data[0] = time_data[0]
            self._data[i].input_data[1] = spatial_data[0]
            self._data[i].output_data = state_data[i]

        self._pw = pg.plot(title=time.strftime("%H:%M:%S") + ' - ' + title)
        self._pw.addLegend()
        self._pw.showGrid(x=True, y=True, alpha=0.5)

        max_times = [max(data) for data in time_data]
        self._longest_idx = max_times.index(max(max_times))
        if dt is not None:
            self._dt = dt

        spat_min = np.min([np.min(data) for data in spatial_data])
        spat_max = np.max([np.max(data) for data in spatial_data])
        self._pw.setXRange(spat_min, spat_max)

        state_min = np.min([np.min(data) for data in state_data])
        state_max = np.max([np.max(data) for data in state_data])
        self._pw.setYRange(state_min, state_max)

        self._time_text = pg.TextItem('t= 0')
        self._pw.addItem(self._time_text)
        self._time_text.setPos(.9 * spat_max, .9 * state_min)

        self._plot_data_items = []
        for idx, data_set in enumerate(self._data):
            self._plot_data_items.append(pg.PlotDataItem(pen=colors[idx], name=data_set.name))
            self._pw.addItem(self._plot_data_items[-1])

        self._curr_frame = 0
        self._timer = pg.QtCore.QTimer()
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(1e3 * self._dt)

    def _update_plot(self):
        """
        update plot window
        """
        for idx, data_set in enumerate(self._data):
            frame = min(self._curr_frame, data_set.output_data.shape[0] - 1)
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

    def __init__(self, data, grid_height=None, title=""):
        PgDataPlot.__init__(self, data)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setWindowTitle(time.strftime("%H:%M:%S") + ' - ' + title)
        self.gl_widget.show()

        if grid_height is None:
            grid_height = max([data.output_data.max() for data in self._data])
        max_0 = max([max(data.input_data[0]) for data in self._data])
        max_1 = max([max(data.input_data[1]) for data in self._data])

        # because gl.GLGridItem.setSize() is broken gl.GLGridItem.scale() must be used
        grid_height_s = grid_height / 20
        max_0_s = max_0 / 20
        max_1_s = max_1 / 20

        for n in range(len(self._data)):
            plot_item = gl.GLSurfacePlotItem(x=self._data[n].input_data[0],
                                             y=np.flipud(self._data[n].input_data[1]),
                                             z=self._data[n].output_data,
                                             shader='normalColor')
            plot_item.translate(-max_0 / 2, -max_1 / 2, -grid_height / 2)
            self.gl_widget.addItem(plot_item)

        self._xgrid = gl.GLGridItem()
        self._xgrid.scale(x=max_0_s, y=max_1_s, z=grid_height_s)
        self._xgrid.translate(0, 0, -grid_height / 2)
        self.gl_widget.addItem(self._xgrid)

        self._ygrid = gl.GLGridItem()
        self._ygrid.scale(x=grid_height_s, y=max_1_s, z=0)
        self._ygrid.rotate(90, 0, 1, 0)
        self._ygrid.translate(max_0 / 2, 0, 0)
        self.gl_widget.addItem(self._ygrid)

        self._zgrid = gl.GLGridItem()
        self._zgrid.scale(x=max_0_s, y=grid_height_s, z=max_1)
        self._zgrid.rotate(90, 1, 0, 0)
        self._zgrid.translate(0, max_1 / 2, 0)
        self.gl_widget.addItem(self._zgrid)


# TODO: alpha
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

        self.roi = pg.LineSegmentROI([[0, self.dim[1] - 1], [self.dim[0] - 1, self.dim[1] - 1]], pen='r')
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


# TODO: alpha
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
        z_vals = res.input_data[1][::-1] * scale

        t_subsets = np.linspace(0, res.input_data[0].size, n, endpoint=False, dtype=int)

        for t_idx, t_val in enumerate(t_subsets):
            t_vals = np.array([res.input_data[0][t_val]] * len(z_vals))
            pts = np.vstack([t_vals, z_vals, res.output_data[t_val, :]]).transpose()
            plt = gl.GLLinePlotItem(pos=pts, color=pg.glColor((t_idx, n * 1.3)),
                                    # width=(t_idx + 1) / 10.,
                                    width=2,
                                    antialias=True)
            self.w.addItem(plt)


class MplSurfacePlot(DataPlot):
    """
    plot as 3d surface
    """

    def __init__(self, data, keep_aspect=False, fig_size=(12, 8), zlabel='$\quad x(z,t)$'):
        DataPlot.__init__(self, data)

        for i in range(len(self._data)):

            # data
            x = self._data[i].input_data[1]
            y = self._data[i].input_data[0]
            z = self._data[i].output_data
            xx, yy = np.meshgrid(x, y)

            # figure
            fig = plt.figure(figsize=fig_size, facecolor='white')
            ax = fig.add_subplot(111, projection='3d')
            if keep_aspect:
                ax.set_aspect('equal', 'box')
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            # labels
            ax.set_ylabel('$t$')
            ax.set_xlabel('$z$')
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel(zlabel, rotation=0)

            # grid
            ax.w_xaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.5)}})
            ax.w_yaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.5)}})
            ax.w_zaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.5)}})

            ax.plot_surface(xx, yy, z, rstride=2, cstride=2, cmap=plt.cm.cool, antialiased=False)


class MplSlicePlot(PgDataPlot):
    """
    Get list (eval_data_list) of ut.EvalData objects and plot the temporal/spatial slice, by spatial_point/time_point,
    from each ut.EvalData object, in one plot.
    For now: only ut.EvalData objects with len(input_data) == 2 supported
    """

    def __init__(self, eval_data_list, time_point=None, spatial_point=None, ylabel="",
                 legend_label=None, legend_location=1, figure_size=(10, 6)):

        if not ((isinstance(time_point, Number) ^ isinstance(spatial_point, Number)) and \
                (isinstance(time_point, NoneType) ^ isinstance(spatial_point, NoneType))):
            raise TypeError("Only one kwarg *_point can be passed,"
                            "which has to be an instance from type numbers.Number")

        DataPlot.__init__(self, eval_data_list)

        plt.figure(facecolor='white', figsize=figure_size)
        plt.ylabel(ylabel)
        plt.grid(True)

        # TODO: move to ut.EvalData
        len_data = len(self._data)
        interp_funcs = [si.interp2d(eval_data.input_data[1], eval_data.input_data[0], eval_data.output_data)
                        for eval_data in eval_data_list]

        if time_point is None:
            slice_input = [data_set.input_data[0] for data_set in self._data]
            slice_data = [interp_funcs[i](spatial_point, slice_input[i]) for i in range(len_data)]
            plt.xlabel(u'$t$')
        elif spatial_point is None:
            slice_input = [data_set.input_data[1] for data_set in self._data]
            slice_data = [interp_funcs[i](slice_input[i], time_point) for i in range(len_data)]
            plt.xlabel(u'$z$')
        else:
            raise TypeError

        if legend_label is None:
            show_leg = False
            legend_label = [evald.name for evald in eval_data_list]
        else:
            show_leg = True

        for i in range(0, len_data):
            plt.plot(slice_input[i], slice_data[i], label=legend_label[i])

        if show_leg:
            plt.legend(loc=legend_location)


def mpl_activate_latex():
    """
    Activate full (label, ticks, ...) latex printing in matplotlib plots
    :return:
    """
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              'font.size': 15,
              'font.family': 'lmodern',
              'text.latex.unicode': True,
              }
    plt.rcParams.update(params)


def mpl_3d_remove_margins():
    """
    Remove thin margins in matplotlib 3d plots
    form here: http://stackoverflow.com/questions/16488182/
    :return:
    """
    # ##source code patch start## #
    from mpl_toolkits.mplot3d.axis3d import Axis

    if not hasattr(Axis, "_get_coord_info_old"):
        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs

        Axis._get_coord_info_old = Axis._get_coord_info
        Axis._get_coord_info = _get_coord_info_new
    # ##source code patch end## #
