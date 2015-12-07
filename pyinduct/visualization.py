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
from matplotlib import cm
import matplotlib.lines as mlines
from numbers import Number
from types import NoneType
import time
import scipy.interpolate as si

__author__ = 'Stefan Ecklebe'
colors = ["r", "g", "b", "c", "m", "y", "k", "w"]

# matplotlib parameters
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 15,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
mpl.rcParams['axes.labelpad'] = 25
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['legend.borderaxespad'] = 0.3

## form here: http://stackoverflow.com/questions/16488182/
###source code patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###source code patch end###


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


        len_data = len(self._data)
        interp_funcs = [si.interp2d(data.input_data[1], data.input_data[0], data.output_data) for data in self._data]
        time_data = [self._data[0].input_data[0] for data_set in self._data]
        spatial_data = [self._data[0].input_data[1] for data_set in self._data]
        state_data = [interp_func(spatial_data[0], time_data[0]) for interp_func in interp_funcs]

        # TODO: remove the next 4 lines and replace (from here) self._data with state_data, time_data and spatial_data
        for i in range(len(self._data)):
            self._data[i].input_data[0] = time_data[0]
            self._data[i].input_data[1] = spatial_data[0]
            self._data[i].output_data = state_data[i]

        self._pw = pg.plot(title=time.strftime("%H:%M:%S")+' - '+title)
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
    def __init__(self, data, title=""):
        PgDataPlot.__init__(self, data)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setWindowTitle(time.strftime("%H:%M:%S")+' - '+title)
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
    def __init__(self, data, hack_xdata=None, hack_ydata=None, keep_aspect=True, show=False, own_cmap=plt.cm.Greys,
                 dpi=80, azim = -60, elev = 30, nbins=3, wire_f=False, left_corner=False, zticks=None,
                 fig_size=(12, 8), zlabel='$\quad x(z,t)$', zpadding=0):
        DataPlot.__init__(self, data)

        disc = 3
        ticklabelsize = 30
        mpl.rcParams['axes.labelpad'] = 40
        mpl.rcParams['axes.labelsize'] = 30

        for i in range(len(self._data)):

            x = self._data[i].input_data[1]
            if hack_xdata != None:
                x[-1] = hack_xdata
            y = self._data[i].input_data[0]
            if hack_ydata != None:
                y[-1] = hack_ydata
            z = (self._data[i].output_data)
            x_min, x_max = (x.min(), x.max())
            y_min, y_max = (y.min(), y.max())
            z_min, z_max = (z.min(), z.max())
            xx, yy = np.meshgrid(x, y)

            fig = plt.figure(figsize=fig_size, dpi=dpi)
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
            ax.set_zlabel(zlabel, rotation=0, labelpad=50+zpadding)

            # ticks
            ax.set_xlim((x_min, x_max))
            if not left_corner:
                plt.xticks(np.linspace(x_min, x_max, disc), ha='right')
            else:
                plt.xticks(np.linspace(x_min, x_max, disc), ha='left')
            ax.set_ylim((y_min, y_max))
            if not left_corner:
                plt.yticks(np.linspace(y_min, y_max, disc), ha='left')
            else:
                plt.yticks(np.linspace(y_min, y_max, disc), ha='right')
            for tick in ax.get_zticklabels():
                tick.set_verticalalignment('bottom')
            plt.locator_params(axis='z', nbins=nbins)
            ax.tick_params(axis='z', pad=12+zpadding)
            ax.tick_params(width=10, length=10, size=10)
            ax.tick_params(labelsize=ticklabelsize)
            if zticks != None:
                ax.set_zticks(zticks)

            # grid
            ax.w_xaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5)}})
            ax.w_yaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5)}})
            ax.w_zaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5)}})

            if wire_f:
                ax.plot_wireframe(xx, yy, z, rstride=2, cstride=2, color="#222222")
            else:
                ax.plot_surface(xx, yy, z, rstride=2, cstride=2, cmap=own_cmap, antialiased=False)

            # default: azim=-60, elev=30
            ax.view_init(elev=elev, azim=azim)

        if show:
            plt.show()

class MplComparePlot(PgDataPlot):
    """
    get one desired EvalData-object and up to five simulation/experiment EvalData-object's
    """

    def __init__(self, eval_data_list, time_point=None, spatial_point=None, ylabel="",
                 leg_lbl=None, show=False, leg_pos=1, fig_size=(10, 6)):

        if not ((isinstance(time_point, Number) ^ isinstance(spatial_point, Number)) and \
                (isinstance(time_point, NoneType) ^ isinstance(spatial_point, NoneType))):
            raise TypeError("Only one kwarg *_point can be passed, which has to be an instance from type numbers.Number")

        DataPlot.__init__(self, eval_data_list)
        len_data = len(self._data)

        if len(self._data[0].input_data) == 1:
            point_input = [data_set.input_data[0] for data_set in self._data]
            point_data = [data_set.output_data for data_set in self._data]
        elif len(self._data[0].input_data) == 2:
            interp_funcs = [si.interp2d(self._data[i].input_data[1], self._data[i].input_data[0],
                                        self._data[i].output_data)
                            for i in range(len_data)]
            if time_point == None:
                point_input = [data_set.input_data[0] for data_set in self._data]
                point_data = [interp_funcs[i](spatial_point, point_input[i]) for i in range(len_data)]
            elif spatial_point == None:
                point_input = [data_set.input_data[1] for data_set in self._data]
                point_data = [interp_funcs[i](point_input[i], time_point) for i in range(len_data)]
            else:
                raise TypeError
        else:
            NotImplementedError

        fig = plt.figure(figsize=fig_size)

        xlabelpad=10
        if time_point == None:
            plt.xlabel(u'$t$', size=30, labelpad=xlabelpad)
        elif spatial_point == None:
            plt.xlabel(u'$z$', size=30, labelpad=xlabelpad)
        else:
            raise StandardError
        plt.ylabel(ylabel, size=25, rotation='horizontal', ha='right', labelpad=10)
        plt.yticks(va='bottom')
        plt.xticks(ha='left')
        plt.grid(True, which='both', color='0.0',linestyle='--')

        if leg_lbl == None:
            show_leg = False
            leg_lbl = ['1', '2', '3', '4', '5']
        else:
            show_leg = True
        ls = ['-', '--', '-.', ':']
        m_reserve = [ u'1', u'x', u'+', u'|', u'd',  u's']
        m_filled = [ u'o', u'^']
        marker_point_input = [np.linspace(data[0], data[-1], 8) for data in point_input]
        marker_point_data = [np.interp(marker_point_input[i], point_input[i], point_data[i].flatten())
                             for i in range(len(point_data))]
        lines = []
        for i in range(0, len_data):
            if i < 2:
                plt.plot(point_input[i], point_data[i], ls=ls[i], lw=2, c='black')
                lines.append(mlines.Line2D([], [], ls=ls[i], lw=2, c='black', label=leg_lbl[i]))
            elif False:
                plt.plot(point_input[i], point_data[i], ls='None', lw=3, marker=m_filled[1-i], ms=10, c='black',
                         label=leg_lbl[i])
            else:
                plt.plot(point_input[i], point_data[i], ls='-', lw=1, c='black', label=leg_lbl[i])
                plt.plot(marker_point_input[i], marker_point_data[i], ls='None', lw=1, marker=m_filled[2-i], ms=10,
                         mfc='None', mew=1, c='black', label=leg_lbl[i])
                lines.append(mlines.Line2D([], [], ls='-', lw=1, c='black', label=leg_lbl[i],
                                           marker=m_filled[2-i], ms=10, mfc='None', mew=1))

        plt.margins(x=0.0, y=0.05)
        if show_leg:
            plt.legend(fontsize=23, handles=lines, loc=leg_pos)
        fig.tight_layout()

        if show:
            plt.show()
