from pyinduct.tests import test_examples

if __name__ == "__main__" or test_examples:
    import numpy as np
    import pyinduct as pi
    import matplotlib.pyplot as plt

    class ReversedRobinEigenfunction(pi.SecondOrderRobinEigenfunction):
        def __init__(self, om, param, l, scale=1, max_der_order=2):
            a2, a1, a0, alpha, beta = param
            _param = a2, -a1, a0, beta, alpha
            pi.SecondOrderRobinEigenfunction.__init__(self, om, _param, l,
                                                      scale, max_der_order)

            self.function_handle = self.function_handle_factory(
                self.function_handle, l)
            self.derivative_handles = [
                self.function_handle_factory(handle, l, ord + 1) for
                ord, handle in enumerate(self.derivative_handles)]

        def function_handle_factory(self, old_handle, l, der_order=0):
            def new_handle(z):
                return old_handle(l - z) * (-1) ** der_order

            return new_handle

        @staticmethod
        def eigfreq_eigval_hint(param, l, n_roots, show_plot=False):
            a2, a1, a0, alpha, beta = param
            _param = a2, -a1, a0, beta, alpha
            return pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
                _param, l, n_roots, show_plot=show_plot)

    n = 5
    temporal_domain = pi.Domain(bounds=(0, 1), num=500)
    param = [1, 0, 6, 0, 0]
    param_t = [1, 0, -6, 0, 0]

    eig_val, init_eig_funcs = pi.SecondOrderRobinEigenfunction.solve_evp_hint(
        param, 1, n=n)
    eig_funcs = pi.normalize_base(init_eig_funcs)
    _, eig_val_t = pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
        param_t, 1, n)
    eig_funcs_o = eig_funcs
    scale_o_t = [func(1) for func in eig_funcs_o]
    _, eig_funcs_o_t = ReversedRobinEigenfunction.solve_evp_hint(
        param_t, 1, eig_val=eig_val, scale=scale_o_t)

    pi.register_base("eig_base", eig_funcs)
    pi.register_base("eig_base_o_t", eig_funcs_o_t)

    A = np.matrix(np.diag(np.real_if_close(eig_val)))
    l = np.matrix([ft.derive(1)(0) - f.derive(1)(0) for ft, f in zip(eig_funcs_o_t, eig_funcs_o)])
    p0 = np.matrix([p(0) for p in eig_funcs_o])

    At = A + l.T @ p0
    print(At)
    print(eig_val_t)
    print(np.flipud(np.sort(np.linalg.eig(At)[0])))
    import pyinduct.parabolic as parabolic
    Atd = pi.create_state_space(pi.parse_weak_formulation(parabolic.get_parabolic_robin_weak_form("eig_base_o_t", "eig_base_o_t", None, param_t, (0, 1))[0])).A[1]
    print(Atd)
    np.testing.assert_array_almost_equal(np.asarray(At), Atd.T, decimal=1)

    # import pyinduct.parabolic as parabolic
    # obs_rad_pde, obs_base_labels = parabolic.get_parabolic_robin_weak_form(
    #     obs_lbl,
    #     obs_lbl,
    #     system_input,
    #     param,
    #     spatial_domain.bounds)

