"""
Here are some frequently used plot types with the packages :py:mod:`pyqtgraph`
and/or :py:mod:`matplotlib` implemented. The respective
:py:mod:`pyinduct.visualization` plotting function get an :py:class:`.EvalData`
object whose definition also placed in this module.
A :py:class:`.EvalData`-object in turn can easily generated from simulation
data. The function :py:func:`pyinduct.simulation.simulate_system` for example
already provide the simulation result as EvalData object.
"""

import platform
import warnings
import numpy as np
import time
import os
import scipy.interpolate as si
import pyqtgraph as pg
import pyqtgraph.exporters
import pyqtgraph.opengl as gl
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numbers import Number
# axes3d not explicit used but needed
from mpl_toolkits.mplot3d import axes3d

from .registry import deregister_base
from .core import complex_wrapper, EvalData, Domain, Function
from .utils import create_animation, create_dir
from .tests import show_plots

__all__ = ["show", "tear_down", "surface_plot",
           "PgAnimatedPlot", "PgSurfacePlot",
           "MplSurfacePlot", "MplSlicePlot",
           "create_colormap", "visualize_roots", "visualize_functions"]

colors = ["g", "c", "m", "b", "y", "k", "w", "r"]
color_map = "viridis"

FORCE_MPL_ON_WINDOWS = True

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def show(show_pg=True, show_mpl=True):
    """
    Shortcut to show all pyqtgraph and matplotlib plots / animations.

    Args:
        show_pg (bool): Show matplotlib plots? Default: True
        show_mpl (bool): Show pyqtgraph plots? Default: True
    """
    if show_plots:
        if show_pg:
            pg.QAPP.exec_()
        if show_mpl:
            plt.show()


def tear_down(labels, plots=None):
    """
    Deregister labels and delete plots.

    Args:
        labels (array-like): All labels to deregister.
        plots (array-like): All plots to delete.
    """

    for label in labels:
        deregister_base(label)

    del plots


def create_colormap(cnt):
    """
    Create a colormap containing cnt values.

    Args:
        cnt (int): Number of colors in the map.

    Return:
        List of `QColor` instances.
    """
    col_map = pg.ColorMap(np.array([0, .5, 1]),
                          np.array([[0, 0, 1., 1.], [0, 1., 0, 1.], [1., 0, 0, 1.]]))
    indexes = np.linspace(0, 1, cnt)
    return col_map.map(indexes, mode="qcolor")


def visualize_functions(functions, points=100, return_window=False):
    """
    Visualizes a set of :py:class:`.Function` s on
    their domain.

    Parameters:
        functions (iterable): collection of
            :py:class:`.Function` s to display.
        points (int): Points to use for sampling
            the domain.
        return_window (bool): If True the graphics window is not shown directly.
            In this case, a reference to the plot window is returned.

    Returns: A PgPlotWindow if *delay_exec* is True.
    """
    # convenience
    if isinstance(functions, Function):
        functions = [functions]

    # evaluate
    _data = []
    for idx, func in enumerate(functions):
        x_vals = []
        y_vals = []
        for dom in func.domain:
            x_vals.append(Domain(bounds=dom, num=points).points)
            y_vals.append(func(x_vals[-1]))
        x_values = np.array(x_vals)
        y_values = np.array(y_vals)
        _data.append((x_values, np.real(y_values), np.imag(y_values)))

    data = np.array(_data)

    # plot
    cmap = cm.get_cmap(color_map)
    pg.mkQApp()
    pw = pg.GraphicsLayoutWidget()
    pw.setWindowTitle("function set visualization")

    lbl = pg.LabelItem(text="Real Part",
                       angle=-90,
                       bold=True,
                       size="10pt")
    pw.addItem(lbl)

    p_real = pg.PlotItem()
    p_real.addLegend()
    for idx, func_data in enumerate(data):
        c = cmap(idx/len(functions), bytes=True)
        for x_vals, y_vals in zip(func_data[0], func_data[1]):
            p_real.plot(x_vals, y_vals,
                        name="vector {}".format(idx),
                        pen=c)
    pw.addItem(p_real)

    if not np.allclose(data[:, 2, :], 0):
        # complex data is present
        pw.nextRow()
        lbl = pg.LabelItem(text="Imaginary Part",
                           angle=-90,
                           bold=True,
                           size="10pt")
        pw.addItem(lbl)

        p_imag = pg.PlotItem()
        # p_imag.addLegend()
        for idx, func_data in enumerate(data):
            c = cmap(idx/len(functions), bytes=True)
            for x_vals, y_vals in zip(func_data[0], func_data[1]):
                p_imag.plot(x_vals, y_vals,
                            name="vector {}".format(idx),
                            pen=c)
        pw.addItem(p_imag)

    pw.show()
    if not return_window:
        pg.QAPP.exec_()
    else:
        return pw


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


class PgDataPlot(DataPlot, pg.QtCore.QObject):
    """
    Base class for all pyqtgraph plotting related classes.
    """

    def __init__(self, data):
        DataPlot.__init__(self, data)
        pg.mkQApp()
        pg.QtCore.QObject.__init__(self)


class PgAnimatedPlot(PgDataPlot):
    """
    Wrapper that shows an updating one dimensional plot of n-curves discretized
    in t time steps and z spatial steps. It is assumed that time propagates
    along axis 0 and and location along axis 1 of values. Values are therefore
    expected to be a array of shape (n, t, z).

    Args:
        data ((iterable of) :py:class:`.EvalData`): results to animate
        title (basestring): Window title.
        refresh_time (int): Time in msec to refresh the window must be greater
            than zero
        replay_gain (float): Values above 1 acc- and below 1 decelerate the
            playback process, must be greater than zero
        save_pics (bool): Export snapshots for animation purposes.
        labels (dict): Axis labels for the plot that are passed to
            :py:class:`pyqtgraph.PlotItem` .

    """

    _res_path = "animation_output"

    def __init__(self, data, title="", refresh_time=40, replay_gain=1,
                 save_pics=False, create_video=False, labels=None):
        PgDataPlot.__init__(self, data)

        self.time_data = [np.atleast_1d(data_set.input_data[0])
                          for data_set in self._data]
        self.spatial_data = [np.atleast_1d(data_set.input_data[1])
                             for data_set in self._data]
        self.state_data = [data_set.output_data for data_set in self._data]

        self._time_stamp = time.strftime("%H:%M:%S")

        self._pw = pg.plot(title="-".join([self._time_stamp,
                                           title,
                                           "at",
                                           str(replay_gain)]),
                           labels=labels)
        self._pw.addLegend()
        self._pw.showGrid(x=True, y=True, alpha=0.5)

        min_times = [min(data) for data in self.time_data]
        max_times = [max(data) for data in self.time_data]
        self._start_time = min(min_times)
        self._end_time = max(max_times)
        self._longest_idx = max_times.index(self._end_time)

        assert refresh_time > 0
        self._tr = refresh_time
        assert replay_gain > 0
        self._t_step = self._tr / 1000 * replay_gain

        spat_min = np.min([np.min(data) for data in self.spatial_data])
        spat_max = np.max([np.max(data) for data in self.spatial_data])
        self._pw.setXRange(spat_min, spat_max)

        state_min = np.min([np.min(data) for data in self.state_data])
        state_max = np.max([np.max(data) for data in self.state_data])
        self._pw.setYRange(state_min, state_max)

        self.save_pics = save_pics
        self.create_video = create_video and save_pics
        self._export_complete = False
        self._exported_files = []

        if self.save_pics:
            self._exporter = pg.exporters.ImageExporter(self._pw.plotItem)
            self._exporter.parameters()['width'] = 1e3

            picture_path = create_dir(self._res_path)
            export_digits = int(np.abs(np.round(np.log10(self._end_time // self._t_step), 0)))
            # ffmpeg uses c-style format strings
            ff_name = "_".join(
                [title.replace(" ", "_"), self._time_stamp.replace(":", "_"), "%0{}d".format(export_digits), ".png"])
            file_name = "_".join(
                [title.replace(" ", "_"), self._time_stamp.replace(":", "_"), "{" + ":0{}d".format(export_digits) + "}",
                 ".png"])
            self._ff_mask = os.sep.join([picture_path, ff_name])
            self._file_mask = os.sep.join([picture_path, file_name])
            self._file_name_counter = 0

        self._time_text = pg.TextItem('t= 0')
        self._pw.addItem(self._time_text)
        self._time_text.setPos(.9 * spat_max, .9 * state_min)

        self._plot_data_items = []
        self._plot_indexes = []
        cls = create_colormap(len(self._data))
        for idx, data_set in enumerate(self._data):
            self._plot_indexes.append(0)
            self._plot_data_items.append(pg.PlotDataItem(pen=pg.mkPen(cls[idx], width=2), name=data_set.name))
            self._pw.addItem(self._plot_data_items[-1])

        self._curr_frame = 0
        self._t = self._start_time

        self._timer = pg.QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(self._tr)

    def _update_plot(self):
        """
        Update plot window.
        """
        new_indexes = []
        for idx, data_set in enumerate(self._data):
            # find nearest time index (0th order interpolation)
            t_idx = (np.abs(self.time_data[idx] - self._t)).argmin()
            new_indexes.append(t_idx)

            # TODO draw grey line if value is outdated

            # update data
            self._plot_data_items[idx].setData(x=self.spatial_data[idx], y=self.state_data[idx][t_idx])

        self._time_text.setText('t= {0:.2f}'.format(self._t))
        self._t += self._t_step
        self._pw.win.setWindowTitle('t= {0:.2f}'.format(self._t))

        if self._t > self._end_time:
            self._t = self._start_time
            if self.save_pics:
                self._export_complete = True
                print("saved pictures using mask: " + self._ff_mask)
                if self.create_video:
                    create_animation(input_file_mask=self._ff_mask)

        if self.save_pics and not self._export_complete:
            if new_indexes != self._plot_indexes:
                # only export a snapshot if the data changed
                f_name = self._file_mask.format(self._file_name_counter)
                self._exporter.export(f_name)
                self._exported_files.append(f_name)
                self._file_name_counter += 1

        self._plot_indexes = new_indexes

    @property
    def exported_files(self):
        if self._export_complete:
            return self._exported_files
        else:
            return None


class PgSurfacePlot(PgDataPlot):
    """
    Plot 3 dimensional data as a surface using OpenGl.

    Args:
        data (:py:class:`.EvalData`): Data to display, if the the input-vector
            has length of 2, a 3d surface is plotted, if has length 3, this
            surface is animated. Hereby, the time axis is assumed to be the
            first entry of the input vector.
        scales (tuple): Factors to scale the displayed data, each entry
            corresponds to an axis in the input vector with one additional scale
            for the *output_data*. It therefore must be of the size:
            `len(input_data) + 1` . If no scale is given, all axis are scaled
            uniformly.
        animation_axis (int): Index of the axis to use for animation.
            Not implemented, yet and therefore defaults to 0 by now.
        title (str): Window title to display.

    Note:
        For animation this object spawns a `QTimer` which needs an running
        event loop. Therefore remember to store a reference to this object.
    """

    def __init__(self, data, scales=None, animation_axis=None, title=""):
        """

        :type data: object
        """
        PgDataPlot.__init__(self, data)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setWindowTitle(time.strftime("%H:%M:%S") + ' - ' + title)

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
                    _scales.append(1/value)
            self.scales = np.array(_scales)
        else:
            self.scales = scales

        # setup color map
        cmap = cm.get_cmap(color_map)
        norm = mpl.colors.Normalize(vmin=self.extrema[0, -1],
                                    vmax=self.extrema[1, -1])
        self.mapping = cm.ScalarMappable(norm, cmap)

        # add plots
        self.plot_items = []
        for idx, data_set in enumerate(self._data):
            if len(data_set.input_data) == 3:
                if animation_axis is None:
                    raise ValueError("animation_axis has to be provided.")

                # crop scale arrays
                if len(self.scales) != len(data_set.input_data):
                    # only remove time scaling if user provided one
                    self.scales = np.delete(self.scales, animation_axis)

                self.deltas = np.delete(self.deltas, animation_axis)
                self.extrema = np.delete(self.extrema, animation_axis, axis=1)

                # move animation axis to the end
                self._data[idx].input_data.append(
                    self._data[idx].input_data.pop(animation_axis))
                self._data[idx].output_data = np.moveaxis(
                    self._data[idx].output_data,
                    animation_axis,
                    -1)
                x_data = np.atleast_1d(self._data[idx].input_data[0])
                y_data = np.flipud(np.atleast_1d(self._data[idx].input_data[1]))
                z_data = self._data[idx].output_data[..., 0]
                mapped_colors = self.mapping.to_rgba(z_data)
                plot_item = gl.GLSurfacePlotItem(x_data,
                                                 y_data,
                                                 z_data,
                                                 computeNormals=False,
                                                 colors=mapped_colors)
            else:
                # 1d system over time -> static
                x_data = np.atleast_1d(self._data[idx].input_data[0])
                y_data = np.flipud(np.atleast_1d(self._data[idx].input_data[1]))
                z_data = self._data[idx].output_data
                mapped_colors = self.mapping.to_rgba(z_data)
                plot_item = gl.GLSurfacePlotItem(x_data,
                                                 y_data,
                                                 z_data,
                                                 computeNormals=False,
                                                 colors=mapped_colors)

            plot_item.scale(*self.scales)
            plot_item.translate(*[-self.extrema[0][i]*self.scales[i]
                                  for i in range(3)])
            self.gl_widget.addItem(plot_item)
            self.plot_items.append(plot_item)

        if animation_axis is not None:
            self.t_idx = 0
            self._timer = pg.QtCore.QTimer(self)
            self._timer.timeout.connect(self._update_plot)
            self._timer.start(100)

        # setup grids
        sc_deltas = self.deltas * self.scales
        self._xygrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._xygrid.setSpacing(sc_deltas[0]/10, sc_deltas[1]/10, 0)
        self._xygrid.setSize(1.2 * sc_deltas[0], 1.2 * sc_deltas[1], 1)
        self._xygrid.translate(
            .5 * sc_deltas[0],
            .5 * sc_deltas[1],
            -.1 * sc_deltas[2],
        )
        self.gl_widget.addItem(self._xygrid)

        self._xzgrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._xzgrid.setSpacing(sc_deltas[0]/10, sc_deltas[2]/10, 0)
        self._xzgrid.setSize(1.2 * sc_deltas[0], 1.2 * sc_deltas[2], 1)
        self._xzgrid.rotate(90, 1, 0, 0)
        self._xzgrid.translate(
            .5 * sc_deltas[0],
            1.1 * sc_deltas[1],
            .5 * sc_deltas[2]
        )
        self.gl_widget.addItem(self._xzgrid)

        self._yzgrid = gl.GLGridItem(size=pg.QtGui.QVector3D(1, 1, 1))
        self._yzgrid.setSpacing(sc_deltas[1]/10, sc_deltas[2]/10, 0)
        self._yzgrid.setSize(1.2 * sc_deltas[1], 1.2 * sc_deltas[2], 1)
        self._yzgrid.rotate(90, 1, 0, 0)
        self._yzgrid.rotate(90, 0, 0, 1)
        self._yzgrid.translate(
            1.1 * sc_deltas[0],
            .5 * sc_deltas[1],
            .5 * sc_deltas[2],
        )
        self.gl_widget.addItem(self._yzgrid)

        center_point = pg.Vector(sc_deltas / 2)
        self.gl_widget.setCameraPosition(pos=center_point,
                                         distance=3,
                                         azimuth=-135
                                         )
        # This fixes Issue #481 of pyqtgraph
        self.gl_widget.opts["center"] = center_point
        self.gl_widget.show()

    def _update_plot(self):
        """
        Update the rendering
        """
        for idx, item in enumerate(self.plot_items):
            z_data = self._data[idx].output_data[..., self.t_idx]
            mapped_colors = self.mapping.to_rgba(z_data)
            item.setData(z=z_data, colors=mapped_colors)

        self.t_idx += 1

        # TODO check if every array has enough timestamps in it
        if self.t_idx >= len(self._data[0].input_data[0]):
            self.t_idx = 0

# TODO: alpha
class PgSlicePlot(PgDataPlot):
    """
    Plot selected slice of given DataSets.
    """

    # TODO think about a nice slice strategy see pyqtgraph for inspiration
    def __init__(self, data, title=None):
        PgDataPlot.__init__(self, data)
        self.dim = self._data[0].output_data.shape

        self.win = pg.QtGui.QMainWindow()
        self.win.resize(800, 800)
        self.win.setWindowTitle("PgSlicePlot: {}".format(title))
        self.cw = pg.QtGui.QWidget()
        self.win.setCentralWidget(self.cw)
        self.l = pg.QtGui.QGridLayout()
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
    Ulots a series of n-lines of the systems state.
    Scaling in z-direction can be changed with the scale setting.
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

        t_subsets = np.linspace(0, np.array(res.input_data[0]).size, n, endpoint=False, dtype=int)

        for t_idx, t_val in enumerate(t_subsets):
            t_vals = np.array([res.input_data[0][t_val]] * len(z_vals))
            pts = np.vstack([t_vals, z_vals, res.output_data[t_val, :]]).transpose()
            plt = gl.GLLinePlotItem(pos=pts, color=pg.glColor((t_idx, n * 1.3)), # width=(t_idx + 1) / 10.,
                                    width=2, antialias=True)
            self.w.addItem(plt)


class MplSurfacePlot(DataPlot):
    """
    Plot as 3d surface.
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
            ax = fig.gca(projection='3d')
            if keep_aspect:
                ax.set_aspect('equal', 'box')
            ax.w_xaxis.set_pane_color((1, 1, 1, 1))
            ax.w_yaxis.set_pane_color((1, 1, 1, 1))
            ax.w_zaxis.set_pane_color((1, 1, 1, 1))

            # labels
            ax.set_ylabel('$t$')
            ax.set_xlabel('$z$')
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel(zlabel, rotation=0)

            ax.plot_surface(xx, yy, z, rstride=2, cstride=2, cmap=plt.cm.cool, antialiased=False)


class MplSlicePlot(PgDataPlot):
    """
    Get list (eval_data_list) of ut.EvalData objects and plot the temporal/spatial slice, by spatial_point/time_point,
    from each ut.EvalData object, in one plot.
    For now: only ut.EvalData objects with len(input_data) == 2 supported
    """

    def __init__(self, eval_data_list, time_point=None, spatial_point=None, ylabel="", legend_label=None,
                 legend_location=1, figure_size=(10, 6)):

        if not ((isinstance(time_point, Number) ^ isinstance(spatial_point, Number)) and (
            isinstance(time_point, type(None)) ^ isinstance(spatial_point, type(None)))):
            raise TypeError("Only one kwarg *_point can be passed,"
                            "which has to be an instance from type numbers.Number")

        DataPlot.__init__(self, eval_data_list)

        plt.figure(facecolor='white', figsize=figure_size)
        plt.ylabel(ylabel)
        plt.grid(True)

        # TODO: move to ut.EvalData
        len_data = len(self._data)
        interp_funcs = [si.interp2d(eval_data.input_data[1], eval_data.input_data[0], eval_data.output_data) for
                        eval_data in eval_data_list]

        if time_point is None:
            slice_input = [data_set.input_data[0] for data_set in self._data]
            slice_data = [interp_funcs[i](spatial_point, slice_input[i]) for i in range(len_data)]
            plt.xlabel('$t$')
        elif spatial_point is None:
            slice_input = [data_set.input_data[1] for data_set in self._data]
            slice_data = [interp_funcs[i](slice_input[i], time_point) for i in range(len_data)]
            plt.xlabel('$z$')
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
    Activate full (label, ticks, ...) latex printing in matplotlib plots.
    """
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True, 'font.size': 15, 'font.family': 'lmodern', 'text.latex.unicode': True,}
    plt.rcParams.update(params)


def mpl_3d_remove_margins():
    """
    Remove thin margins in matplotlib 3d plots.
    The Solution is from `Stackoverflow`_.

    .. _Stackoverflow:
        http://stackoverflow.com/questions/16488182/
    """

    from mpl_toolkits.mplot3d.axis3d import Axis

    if not hasattr(Axis, "_get_coord_info_old"):
        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs

        Axis._get_coord_info_old = Axis._get_coord_info
        Axis._get_coord_info = _get_coord_info_new


def save_2d_pg_plot(plot, filename):
    """
    Save a given pyqtgraph plot in the folder <current path>.pictures_plot
    under the given filename :py:obj:`filename`.

    Args:
        plot (:py:class:`pyqtgraph.plotItem`): Pyqtgraph plot.
        filename (str): Png picture filename.

    Return:
        tuple of 2 str's: Path with filename and path only.
    """

    path = create_dir('pictures_plot') + os.path.sep
    path_filename = path + filename + '.png'
    exporter = pg.exporters.ImageExporter(plot.plotItem)
    exporter.parameters()['width'] = 1e3
    exporter.export(path_filename)
    return path_filename, path


def visualize_roots(roots, grid, func, cmplx=False, return_window=False):
    """
    Visualize a given set of roots by examining the output
    of the generating function.

    Args:
        roots (array like): Roots to display, if `None` is given, no roots will
            be displayed, this is useful to get a view of *func* and choosing
            an appropriate `grid`.
        grid (list): List of arrays that form the grid, used for
            the evaluation of the given *func*.
        func (callable): Possibly vectorial function handle
            that will take input of of the shape ('len(grid)', ).
        cmplx (bool): If True, the complex valued *func* is
            handled as a vectorial function returning [Re(func), Im(func)].
        return_window (bool): If True the graphics window is not shown directly.
            In this case, a reference to the plot window is returned.

    Returns: A PgPlotWindow if `delay_exec` is True.
    """
    if roots is not None:
        roots = np.atleast_1d(roots)
    if isinstance(grid[0], Number):
        grid = [grid]

    dim = len(grid)
    assert dim < 3

    if cmplx:
        assert dim == 2
        func = complex_wrapper(func)
        if np.iscomplexobj(roots):
            roots = np.array([np.real(roots), np.imag(roots)]).T

    grids = np.meshgrid(*[row for row in grid])
    values = np.vstack([arr.flatten() for arr in grids]).T

    components = []
    absolute = []
    for val in values:
        components.append(func(val))
        absolute.append(np.linalg.norm(components[-1]))

    comp_values = np.array(components)
    abs_values = np.array(absolute)

    # plot roots
    pg.mkQApp()
    pw = pg.GraphicsLayoutWidget()
    pw.setWindowTitle("Root Visualization")

    if dim == 1:
        # plot function with roots
        pl = pw.addPlot()
        if roots is not None:
            pl.plot(roots, np.zeros(roots.shape[0]),
                    pen=None, symbolPen=pg.mkPen("g"))
        pl.plot(np.squeeze(values), np.squeeze(comp_values), pen=pg.mkPen("b"))
    else:
        # plot function components
        rect = pg.QtCore.QRectF(grid[0][0],
                                grid[1][0],
                                grid[0][-1] - grid[0][0],
                                grid[1][-1] - grid[1][0])
        for idx in range(comp_values.shape[1]):
            lbl = pg.LabelItem(text="Component {}".format(idx),
                               angle=-90,
                               bold=True,
                               size="10pt")
            pw.addItem(lbl)

            p_img = pw.addPlot()
            img = pg.ImageItem()
            img.setImage(comp_values[:, idx].reshape(grids[0].shape).T)
            img.setRect(rect)
            p_img.addItem(img)

            # add roots on top
            if roots is not None:
                p_img.plot(roots[:, 0], roots[:, 1],
                           pen=None,
                           symbolPen=pg.mkPen("g"))

            hist = pg.HistogramLUTItem()
            hist.setImageItem(img)
            pw.addItem(hist)

            pw.nextRow()

        # plot absolute value of function
        lbl = pg.LabelItem(text="Absolute Value",
                           angle=-90,
                           bold=True,
                           size="10pt")
        pw.addItem(lbl)
        p_abs = pw.addPlot()
        img = pg.ImageItem()
        img.setImage(abs_values.reshape(grids[0].shape).T)
        img.setRect(rect)
        p_abs.addItem(img)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        pw.addItem(hist)

        # add roots on top
        if roots is not None:
            p_abs.plot(roots[:, 0], roots[:, 1],
                       pen=None, symbolPen=pg.mkPen("g"))

    pw.show()
    if not return_window:
        pg.QAPP.exec_()
    else:
        return pw


def surface_plot(data, **kwargs):
    """
    Compatibility wrapper for PgSurfacePLot and MplSurfacePlot

    Since OpenGL suffers under some problems in current windows versions,
    the matplotlib implementation is used there.
    """
    pg_kwargs = ["scales", "animation_axis", "title"]
    mpl_kwargs = ["keep_aspect", "fig_size", "zlabel"]
    os = platform.system()
    pg_wanted = any([arg in kwargs for arg in pg_kwargs])
    mpl_wanted = any([arg in kwargs for arg in mpl_kwargs])
    if pg_wanted and mpl_wanted:
        raise ValueError("Provided kwargs are ambigous")
    if pg_wanted:
        if os == "Windows" and FORCE_MPL_ON_WINDOWS:
            warnings.warn("Compatibility wrapper: Enforcing use of "
                          "MplSurfacePlot")
            for arg in pg_kwargs:
                if arg in kwargs:
                    warnings.warn("Ignoring argument '{}'".format(arg))
                    kwargs.pop(arg)
            p = MplSurfacePlot(data, **kwargs)
        else:
            p = PgSurfacePlot(data, **kwargs)
    else:
        p = MplSurfacePlot(data, **kwargs)

    return p
