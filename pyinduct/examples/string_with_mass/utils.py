import pyinduct as pi
import numpy as np
import sympy as sp
import time
import os
import pyqtgraph as pg
import matplotlib.pyplot as plt
from pyinduct.visualization import PgDataPlot

# matplotlib configuration
plt.rcParams.update({'text.usetex': True})


def pprint(expression="\n\n\n"):
    if isinstance(expression, np.ndarray):
        expression = sp.Matrix(expression)
    sp.pprint(expression, num_columns=180)


def get_primal_eigenvector(according_paper=False):
    if according_paper:
        # some condensed parameters
        alpha = beta = sym.c / 2
        tau0 = 1 / sp.sqrt(sym.a * sym.b)
        w = tau0 * sp.sqrt((sym.lam + alpha) ** 2 - beta ** 2)
        # matrix exponential
        expm_A = sp.Matrix([
            [sp.cosh(w * sym.z),
             (sym.lam + sym.c) / sym.b / w * sp.sinh(w * sym.z)],
            [sym.lam / sym.a / w * sp.sinh(w * sym.z),
             sp.cosh(w * sym.z)]
        ])

    else:
        # matrix
        A = sp.Matrix([[sp.Float(0), (sym.lam + sym.c) / sym.b],
                       [sym.lam/sym.a, sp.Float(0)]])
        # matrix exponential
        expm_A = sp.exp(A * sym.z)

    # inital values at z=0 (scaled by xi(s))
    phi0 = sp.Matrix([[sp.Float(1)], [sym.lam / sym.d]])
    # solution
    phi = expm_A * phi0

    return phi


def plot_eigenvalues(eigenvalues):
    plt.figure(facecolor="white")
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues))
    ax = plt.gca()
    ax.set_xlabel(r"$Re(\lambda)$")
    ax.set_ylabel(r"$Im(\lambda)$")
    plt.show()


def check_eigenvalues(sys_fem_lbl, obs_fem_lbl, obs_modal_lbl, ceq, ss):
    # check eigenvalues of the approximation
    A_sys = (-ceq[0].dynamic_forms[sys_fem_lbl].e_n_pb_inv @
             ceq[0].dynamic_forms[sys_fem_lbl].matrices["E"][0][1])
    A_obs = (-ceq[1].dynamic_forms[obs_fem_lbl].e_n_pb_inv @
             ceq[1].dynamic_forms[obs_fem_lbl].matrices["E"][0][1])
    A_modal_obs = (-ceq[2].dynamic_forms[obs_modal_lbl].e_n_pb_inv @
                   ceq[2].dynamic_forms[obs_modal_lbl].matrices["E"][0][1])
    pprint()
    pprint("Eigenvalues [{}, {}, {}]".format(sys_fem_lbl, obs_fem_lbl, obs_modal_lbl))
    pprint([np.linalg.eigvals(A_) for A_ in (A_sys, A_obs, A_modal_obs)])


def find_eigenvalues(n):
    def characteristic_equation(om):
        return om * (np.sin(om) + param.m * om * np.cos(om))

    eig_om = pi.find_roots(
        characteristic_equation, np.linspace(0, np.pi * n, 5 * n), n)

    eig_vals = list(sum([(1j * ev, -1j * ev) for ev in eig_om], ()))

    return eig_om, sort_eigenvalues(eig_vals)


def sort_eigenvalues(eigenvalues):
    imag_ev = list()
    real_ev = list()
    for ev in eigenvalues:
        if np.isclose(np.imag(ev), 0):
            real_ev.append(0 if np.isclose(ev, 0) else np.real(ev))
        else:
            imag_ev.append(ev)

    eig_vals = list(np.flipud(sorted(real_ev)))
    for ev in np.array(imag_ev)[np.argsort(np.abs(np.imag(imag_ev)))]:
        eig_vals.append(ev)

    if len(eigenvalues) != len(eig_vals):
        raise ValueError(
            "Something went wrong! (only odd number of eigenvalues considered)"
        )

    return np.array(eig_vals)


class SwmPgAnimatedPlot(PgDataPlot):
    """
    Animation for the string with mass example.
    Compare with :py:class:`.PgAnimatedPlot`.

    Args:
        data ((iterable of) :py:class:`.EvalData`): results to animate
        title (basestring): window title
        refresh_time (int): time in msec to refresh the window must be greater
            than zero
        replay_gain (float): values above 1 acc- and below 1 decelerate the
            playback process, must be greater than zero
        save_pics (bool):
        labels:

    Return:
    """

    _res_path = "animation_output"

    def __init__(self, data, title="", refresh_time=40, replay_gain=1, save_pics=False, create_video=False,
                 labels=None):
        PgDataPlot.__init__(self, data)

        self.time_data = [np.atleast_1d(data_set.input_data[0]) for data_set in self._data]
        self.spatial_data = [np.atleast_1d(data_set.input_data[1]) for data_set in self._data]
        self.state_data = [data_set.output_data for data_set in self._data]

        self._time_stamp = time.strftime("%H:%M:%S")

        self._pw = pg.plot(title="-".join([self._time_stamp, title, "at", str(replay_gain)]), labels=labels)
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

            from pyinduct.visualization import create_dir
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
        cls = pi.create_colormap(len(self._data))
        for idx, data_set in enumerate(self._data):
            self._plot_indexes.append(0)
            self._plot_data_items.append(pg.PlotDataItem(pen=pg.mkPen(cls[idx], width=2), name=data_set.name))
            self._pw.addItem(self._plot_data_items[-1])

        angles = np.linspace(0, 2 * np.pi, 1000)
        self.x_circle = .01 * (spat_max - spat_min) * np.cos(angles)
        self.y_circle = .01 * (state_max - state_min) * np.sin(angles)
        for idx, data_set in enumerate(self._data):
            self._plot_indexes.append(0)
            self._plot_data_items.append(pg.PlotDataItem(pen=pg.mkPen(cls[idx], width=2)))
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

            # circles
            self._plot_data_items[idx + len(self._data)].setData(
                x=self.x_circle + self.spatial_data[idx][0],
                y=self.y_circle + self.state_data[idx][t_idx][0])

        self._time_text.setText('t= {0:.2f}'.format(self._t))
        self._t += self._t_step
        self._pw.win.setWindowTitle('t= {0:.2f}'.format(self._t))

        if self._t > self._end_time:
            self._t = self._start_time
            if self.save_pics:
                self._export_complete = True
                print("saved pictures using mask: " + self._ff_mask)
                if self.create_video:
                    from pyinduct.visualization import create_animation
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


class Parameters:
    def __init__(self):
        pass


# parameters
param = Parameters()
param.m = 1
param.tau = 1
param.sigma = 1
obs_gain = Parameters()
obs_gain.k0 = 90
obs_gain.k1 = 100
obs_gain.alpha = 0
ctrl_gain = Parameters()
ctrl_gain.k0 = 2
ctrl_gain.k1 = 2
ctrl_gain.alpha = 0

# symbols
sym = Parameters()
sym.m, sym.lam, sym.tau, sym.om, sym.theta, sym.z, sym.t, sym.u, sym.yt, sym.tau, sym.sigma = [
    sp.Symbol(sym, real=True) for sym in (r"m", r"lambda", r"tau", r"omega", r"theta", r"z", r"t", r"u", r"\tilde{y}", r"tau", r"sigma")]
subs_list = [(sym.m, param.m)]
