import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    _sigma_sc = np.loadtxt('article_data_plots/sigma_air_sigma_sc_wl_295nm_circ_pol.txt', skiprows=0, delimiter=',')
    plt.figure(figsize=(4, 2.5), dpi=300)
    plt.plot(_sigma_sc[:, 0], _sigma_sc[:, 2], label='SPP', color='orange', lw=3)
    plt.plot(_sigma_sc[:, 0], _sigma_sc[:, 1], label='Air', color='blue', lw=3)
    plt.plot(_sigma_sc[:, 0], _sigma_sc[:, 1] + _sigma_sc[:, 2], label='Total', lw=3, c='black')
    plt.ylabel('$\\sigma_{sc}$, norm')
    plt.xlabel('$\\lambda$, nm')
    plt.grid()
    plt.legend(loc='lower left')
    plt.xlim(_sigma_sc[0, 0], _sigma_sc[-1, 0])
    plt.ylim(0, 22)
    #plt.savefig('article_plots/scat_sc_air_spp.svg', bbox_inches='tight')
    plt.show()
    return np, plt


@app.cell
def _(np, plt):
    _sigma_sc = np.loadtxt('article_data_plots/sigma_air_sigma_sc_R_900nm_circ_pol.txt', skiprows=0, delimiter=',')
    plt.figure(figsize=(4, 2.5), dpi=300)
    plt.plot(_sigma_sc[:, 0], _sigma_sc[:, 2] / (np.pi * (_sigma_sc[:, 0] * 1e-09) ** 2), label='SPP', color='orange', lw=3)
    plt.plot(_sigma_sc[:, 0], _sigma_sc[:, 1] / (np.pi * (_sigma_sc[:, 0] * 1e-09) ** 2), label='Air', color='blue', lw=3)
    plt.plot(_sigma_sc[:, 0], (_sigma_sc[:, 1] + _sigma_sc[:, 2]) / (np.pi * (_sigma_sc[:, 0] * 1e-09) ** 2), label='Total', lw=3, c='black')
    plt.ylabel('$\\sigma_{sc}$, norm')
    plt.xlabel('$R$, nm')
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlim(_sigma_sc[0, 0], _sigma_sc[-26, 0])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare Scattring SC
    """)
    return


@app.cell
def _(np, sweep_res):
    sigma_dips_comsol = np.loadtxt('article_data_plots/scat multipoles.txt', skiprows=8, delimiter=',')

    sigma_int_comsol = np.loadtxt('article_data_plots/sc integr.txt', skiprows=8, delimiter=',')

    c_const = 299792458
    eps0_const = 1/(4*np.pi*c_const**2)*1e7
    mu0_const = 4*np.pi * 1e-7

    wl = 900 * 1e-9

    px = sweep_res["px"].apply(lambda x: x.magnitude).to_numpy()
    py = sweep_res["py"].apply(lambda x: x.magnitude).to_numpy()
    pz = sweep_res["pz"].apply(lambda x: x.magnitude).to_numpy()

    mx = sweep_res["mx"].apply(lambda x: x.magnitude).to_numpy()
    my = sweep_res["my"].apply(lambda x: x.magnitude).to_numpy()
    mz = sweep_res["mz"].apply(lambda x: x.magnitude).to_numpy()

    R = sweep_res["R"].apply(lambda x: x.magnitude).to_numpy()

    k0 = 2*np.pi/wl

    const_sigma_sc = k0**4/(6*np.pi*eps0_const**2)

    ED = np.abs(px)**2+ np.abs(py)**2+np.abs(pz)**2

    MD = (np.abs(mx)**2+ np.abs(my)**2+np.abs(mz)**2)/c_const**2

    sigma_ED = ED *const_sigma_sc
    sigma_MD = MD * const_sigma_sc
    return (
        R,
        mu0_const,
        sigma_ED,
        sigma_MD,
        sigma_dips_comsol,
        sigma_int_comsol,
    )


@app.cell
def _(R, np, plt, sigma_ED, sigma_MD, sigma_dips_comsol, sigma_int_comsol):
    plt.figure(figsize=(4,2.5), dpi=300)


    plt.plot(sigma_int_comsol[:,0], (sigma_int_comsol[:,2])/(np.pi*(sigma_int_comsol[:,0]*1e-9)**2), label='Comsol Integration', lw=2, c='red')

    plt.plot(sigma_dips_comsol[:,0], (sigma_dips_comsol[:,5])/(np.pi*(sigma_dips_comsol[:,0]*1e-9)**2), label='Comsol Multipole Formula', lw=2, c='orange')

    plt.plot(R, (sigma_MD+sigma_ED)/(np.pi*(R*1e-9)**2), label='Analytical Dipole Formula', color='blue', lw=2)

    plt.ylabel(r'$\sigma_{sc}$, norm')
    plt.xlabel('$R$, nm')
    plt.grid()
    plt.legend(loc='upper left')
    plt.xlim(sigma_int_comsol[0,0], sigma_int_comsol[-1,0])
    # plt.ylim(0,22)
    #plt.savefig('article_plots/scat_sc_compare_coms_th.png', bbox_inches='tight')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Directivity heatmaps and pattern
    """)
    return


@app.cell
def _(np, plt):
    from MieSppForce.simulation import SimulationConfig, SweepRunner, DipoleCalculator, OpticalForceCalculator, DiagramCalculator, CylindricalGrid
    from scipy.integrate import trapezoid
    from pint import UnitRegistry
    ureg = UnitRegistry()

    def get_mean_angle(I, phi):
        vec_sum = np.sum(I * np.exp(1j * phi))
        return np.angle(vec_sum)

    def add_axis_line(ax, theta, label, color='k'):
        r_arrow = ax.get_rmax()
        ax.plot([theta, theta], [0, r_arrow], color=color, lw=2, ls='--')
        ax.text(theta + 0.1, r_arrow * 1.1, label, ha='center', va='center', fontsize=12, fontweight='bold', color=color)
    base_config = SimulationConfig(wl=900 * ureg.nanometer, R=125 * ureg.nanometer, dist=2 * ureg.nanometer, angle=np.deg2rad(25), psi=np.pi / 4, chi=np.pi / 8, show_warnings=False, two_beam_setup=False)
    I0 = base_config.c_const.magnitude * base_config.eps0_const.magnitude / 2
    phi_arr = np.linspace(0, 2 * np.pi, 100)
    gridCylXoY = CylindricalGrid(base_config.wl, phi_arr * ureg.rad, base_config.R + base_config.dist)
    gridCylXoY_SPP = CylindricalGrid(base_config.wl * 5, phi_arr * ureg.rad, 0 * ureg.nm)
    Diag_res_reg = DiagramCalculator(base_config, gridCylXoY, normalize=None).compute('reg')
    Diag_res_sc = DiagramCalculator(base_config, gridCylXoY, normalize=None).compute('sc')
    Diag_res_air = DiagramCalculator(base_config, gridCylXoY, normalize=None).compute('air')
    Diag_res_spp = DiagramCalculator(base_config, gridCylXoY_SPP, normalize=None).compute('spp')
    Force_res = OpticalForceCalculator(base_config).compute()
    Fx_spp = Force_res.as_dict()['Fxspp']
    Fy_spp = Force_res.as_dict()['Fyspp']
    Fxe0 = Force_res.as_dict()['Fxe0']
    Fxm0 = Force_res.as_dict()['Fxm0']
    Fx = Force_res.as_dict()['Fx']
    Fy = Force_res.as_dict()['Fy']
    Fx_reg = Fx - Fxe0 - Fxm0 - Fx_spp
    Fy_reg = Fy - Fy_spp
    phi_vals_reg = Diag_res_reg.as_array()[:, 0]
    intensity_reg = Diag_res_reg.as_array()[:, 1]
    _fig = plt.figure(dpi=300, figsize=(2.5, 2.5))
    ax = _fig.add_subplot(polar=True)
    ax.plot(Diag_res_reg.as_array()[:, 0], Diag_res_reg.as_array()[:, 1], color='b', lw=2, label='$D_{reg}$')
    ax.plot(Diag_res_air.as_array()[:, 0], Diag_res_air.as_array()[:, 1], color='r', lw=2, label='$D_{air}$')
    ax.plot(Diag_res_sc.as_array()[:, 0], Diag_res_sc.as_array()[:, 1], color='b', lw=2, label='$D_{sc}$')
    ax.plot(Diag_res_spp.as_array()[:, 0], Diag_res_spp.as_array()[:, 1], color='orange', lw=2, label='$D_{spp}$')
    ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.2))
    plt.tight_layout()
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_rlabel_position(-22.5)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    plt.show()
    return (
        CylindricalGrid,
        DiagramCalculator,
        SimulationConfig,
        SweepRunner,
        UnitRegistry,
        ax,
        base_config,
        ureg,
    )


@app.cell
def _(DiagramCalculator, base_config, np, ureg):
    from MieSppForce.simulation import SphericalGrid

    gridSphericalXoZ = SphericalGrid(base_config.wl*5,
                                     np.linspace(-np.pi/2, np.pi/2, 100)*ureg.rad,
                                     0*ureg.rad)



    Diag_res_reg_XoZ = DiagramCalculator(base_config, gridSphericalXoZ, normalize=None).compute('reg')

    Diag_res_air_XoZ = DiagramCalculator(base_config, gridSphericalXoZ, normalize=None).compute('air')

    Diag_res_sc_XoZ = DiagramCalculator(base_config, gridSphericalXoZ, normalize=None).compute('sc')

    Diag_res_spp_XoZ = DiagramCalculator(base_config, gridSphericalXoZ, normalize=None).compute('spp')
    return (
        Diag_res_air_XoZ,
        Diag_res_reg_XoZ,
        Diag_res_sc_XoZ,
        Diag_res_spp_XoZ,
    )


@app.cell
def _(
    Diag_res_air_XoZ,
    Diag_res_reg_XoZ,
    Diag_res_sc_XoZ,
    Diag_res_spp_XoZ,
    ax,
    np,
    plt,
):
    _fig, ax1, ax2 = plt.subfigures(1, 2, dpi=300, figsize=(5, 2.5))
    ax1 = _fig.add_subplot(polar=True)
    ax.plot(Diag_res_reg_XoZ.as_array()[:, 0], Diag_res_reg_XoZ.as_array()[:, 1], color='b', lw=2, label='$D_{reg}$')
    ax.plot(Diag_res_air_XoZ.as_array()[:, 0], Diag_res_air_XoZ.as_array()[:, 1], color='r', lw=2, label='$D_{air}$')
    ax.plot(Diag_res_sc_XoZ.as_array()[:, 0], Diag_res_sc_XoZ.as_array()[:, 1], color='g', lw=2, label='$D_{sc}$')
    ax.plot(Diag_res_spp_XoZ.as_array()[:, 0], Diag_res_spp_XoZ.as_array()[:, 1], color='orange', lw=2, label='$D_{spp}$')
    ax.legend()
    ax.set_xlim(-np.pi, np.pi)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_title(f'XoZ,')
    return


@app.cell
def _(SweepRunner, base_config, np, ureg):
    sweep_res, diagrams, _ = SweepRunner(base_config, 'R', np.linspace(100, 165, 200)*ureg.nm, True, True, True, False, None, 'spp').run(n_jobs=-1)
    return diagrams, sweep_res


@app.cell
def _(diagrams, np, plt):
    diagrams['phi'] = diagrams['phi'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams['R'] = diagrams['R'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    _pivot = diagrams.pivot(index='phi', columns='R', values='D')
    plt.figure(figsize=(4, 3), dpi=300)
    plt.imshow(_pivot.values.astype(float), aspect='auto', origin='lower', extent=[float(_pivot.columns.min()), float(_pivot.columns.max()), float(_pivot.index.min()), float(_pivot.index.max())], cmap='hot', vmin=0)
    _phi_values = _pivot.index.values.astype(float)
    _R_values = _pivot.columns.values.astype(float)
    max_phi_for = []
    for _r in _R_values:
        _col = _pivot[_r].values.astype(float)
        _complex_sum = np.sum(_col * np.exp(1j * _phi_values))
        _phi_mean = np.angle(_complex_sum)
        if _phi_mean < 0:
            _phi_mean = _phi_mean + 2 * np.pi
        max_phi_for.append(_phi_mean)
    plt.plot(_R_values, max_phi_for, color='lightgray', linewidth=2, label='$D_{spp}^{max}$', ls='--')
    plt.colorbar(label='$D_{spp}$')
    plt.xlabel('$R$ (nm)')
    plt.ylabel('$\\varphi$ (rad)')
    _yticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    _yticklabels = ['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$']
    plt.yticks(_yticks, _yticklabels)
    plt.ylim(0, 2 * np.pi)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(CylindricalGrid, SweepRunner, base_config, np, ureg):
    _gridCylXoY_reg = CylindricalGrid(base_config.wl * 5, np.linspace(0, 2 * np.pi, 100) * ureg.rad, base_config.R + base_config.dist)
    sweep_res_1, diagrams_reg, _ = SweepRunner(base_config, 'R', np.linspace(100, 165, 200) * ureg.nanometer, True, True, True, False, _gridCylXoY_reg, 'reg').run(n_jobs=-1)
    sweep_res_1, diagrams_air, _ = SweepRunner(base_config, 'R', np.linspace(100, 165, 200) * ureg.nanometer, True, True, True, False, _gridCylXoY_reg, 'air').run(n_jobs=-1)
    sweep_res_1, diagrams_sc, _ = SweepRunner(base_config, 'R', np.linspace(100, 165, 200) * ureg.nanometer, True, True, True, False, _gridCylXoY_reg, 'sc').run(n_jobs=-1)
    return diagrams_air, diagrams_reg, diagrams_sc


@app.cell
def _(diagrams_air, diagrams_reg, diagrams_sc, np, plt):
    diagrams_reg['phi'] = diagrams_reg['phi'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams_reg['R'] = diagrams_reg['R'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams_air['phi'] = diagrams_air['phi'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams_air['R'] = diagrams_air['R'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams_sc['phi'] = diagrams_sc['phi'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams_sc['R'] = diagrams_sc['R'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    _pivot = diagrams_air.pivot(index='phi', columns='R', values='D')
    _pivot = _pivot
    _R_values = _pivot.columns.values.astype(float)
    _phi_values = _pivot.index.values.astype(float)
    max_phi_for_r = []
    for _r in _R_values:
        _col = _pivot[_r].values.astype(float)
        _complex_sum = np.sum(_col * np.exp(1j * _phi_values))
        _phi_mean = np.angle(_complex_sum)
        if _phi_mean < 0:
            _phi_mean = _phi_mean + 2 * np.pi
        max_phi_for_r.append(_phi_mean)
    plt.figure(figsize=(4, 3), dpi=300)
    plt.imshow(_pivot.values.astype(float), aspect='auto', origin='lower', extent=[float(_pivot.columns.min()), float(_pivot.columns.max()), float(_pivot.index.min()), float(_pivot.index.max())], cmap='hot', vmin=0)
    plt.plot(_R_values, max_phi_for_r, color='lightgray', linewidth=2, label='$D_{air}^{max}$', ls='--')
    plt.colorbar(label='$D_{air}$')
    plt.xlabel('$R$ (nm)')
    plt.ylabel('$\\varphi$ (rad)')
    _yticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    _yticklabels = ['$0$', '$\\pi/2$', '$\\pi$', '$3\\i/2$', '$2\\pi$']
    plt.yticks(_yticks, _yticklabels)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 2
    """)
    return


@app.cell
def _(SimulationConfig, np, ureg):
    base_config_1 = SimulationConfig(wl=900 * ureg.nanometer, R=295 / 2 * ureg.nanometer, dist=2 * ureg.nanometer, angle=np.deg2rad(25), psi=0, chi=np.pi / 4, show_warnings=False, two_beam_setup=False)
    return (base_config_1,)


@app.cell
def _(CylindricalGrid, SweepRunner, base_config_1, np, ureg):
    _gridCylXoY_reg = CylindricalGrid(base_config_1.wl, np.linspace(0, 2 * np.pi, 100) * ureg.rad, base_config_1.R + base_config_1.dist)
    sweep_res_2, diagrams_spp, _ = SweepRunner(base_config_1, 'R', np.linspace(100, 165, 100) * ureg.nanometer, True, True, True, False, None, 'spp').run(n_jobs=-1)
    _, diagrams_air_1, _ = SweepRunner(base_config_1, 'R', np.linspace(100, 165, 100) * ureg.nanometer, True, True, True, False, _gridCylXoY_reg, 'air').run(n_jobs=-1)
    return diagrams_air_1, diagrams_spp, sweep_res_2


@app.cell
def _(base_config_1, diagrams_spp, np, plt, sweep_res_2):
    import pandas as pd

    def circular_mean_phi(df):
        wl_unique = df['R'].unique()
        result = []
        for wl_val in wl_unique:
            sub = df[df['R'] == wl_val]
            phi = sub['phi'].values.astype(float)
            D = sub['D'].values.astype(float)
            _complex_sum = np.sum(D * np.exp(1j * phi))
            _phi_mean = np.angle(_complex_sum)
            if _phi_mean < 0:
                _phi_mean = _phi_mean + 2 * np.pi
            result.append([wl_val, _phi_mean])
        return pd.DataFrame(result, columns=['R', 'phi_mean'])
    diagrams_spp['phi'] = diagrams_spp['phi'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams_spp['R'] = diagrams_spp['R'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    phi_mean_spp = circular_mean_phi(diagrams_spp)
    Fx_1 = sweep_res_2['Fx'].apply(lambda x: x.to('N').magnitude).to_numpy()
    Fy_1 = sweep_res_2['Fy'].apply(lambda x: x.to('N').magnitude).to_numpy()
    R_1 = sweep_res_2['R'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x).to_numpy()
    Fxe0_1 = sweep_res_2['Fxe0'].apply(lambda x: x.to('N').magnitude).to_numpy()
    Fxm0_1 = sweep_res_2['Fxm0'].apply(lambda x: x.to('N').magnitude).to_numpy()
    Fx_nosc = Fx_1 - Fxe0_1 - Fxm0_1
    Fxspp = sweep_res_2['Fxspp'].apply(lambda x: x.to('N').magnitude).to_numpy()
    Fyspp = sweep_res_2['Fyspp'].apply(lambda x: x.to('N').magnitude).to_numpy()
    Fy_sc = Fy_1 - Fyspp
    Fx_sc = Fx_1 - Fxspp - Fxe0_1 - Fxm0_1
    theta_F = np.arctan2(Fy_1, Fx_1)
    theta_F_nosc = np.arctan2(Fy_1, Fx_nosc)
    theta_F_spp = np.arctan2(Fyspp, Fxspp)
    theta_F_sc = np.arctan2(Fy_sc, Fx_sc)
    df_forces = pd.DataFrame({'R': R_1, 'thetaFspp': theta_F_spp, 'thetaFnosc': theta_F_nosc, 'theta_F': theta_F, 'theta_F_sc': theta_F_sc})
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(phi_mean_spp['R'], np.unwrap(phi_mean_spp['phi_mean']) + 2 * np.pi, label='$\\varphi(D_{spp}^{max})$', lw=3)
    plt.plot(df_forces['R'], np.unwrap(df_forces['thetaFspp']) + 3 * np.pi, label='$\\varphi(F_{spp})+\\pi$', lw=3)
    _yticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    _yticklabels = ['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$']
    plt.yticks(_yticks, _yticklabels)
    plt.xlim(phi_mean_spp['R'].to_numpy()[0], phi_mean_spp['R'].to_numpy()[-1])
    plt.xlabel('$R$ (nm)')
    plt.ylabel('$\\varphi$ (rad)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.title(f'SPP Force: $\\psi=${np.rad2deg(base_config_1.psi):.2f}° rad, $\\chi$={np.rad2deg(base_config_1.chi):.2f}° rad')
    plt.show()
    return (
        Fx_1,
        Fx_sc,
        Fxe0_1,
        Fxm0_1,
        Fxspp,
        Fy_1,
        Fy_sc,
        Fyspp,
        R_1,
        circular_mean_phi,
        df_forces,
        pd,
        phi_mean_spp,
    )


@app.cell
def _(
    base_config_1,
    circular_mean_phi,
    df_forces,
    diagrams_air_1,
    np,
    phi_mean_spp,
    plt,
):
    diagrams_air_1['phi'] = diagrams_air_1['phi'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    diagrams_air_1['R'] = diagrams_air_1['R'].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
    phi_mean_air = circular_mean_phi(diagrams_air_1)
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(phi_mean_air['R'], np.unwrap(phi_mean_air['phi_mean']) - 2 * np.pi, label='$\\varphi (D_{air}^{max})$', lw=3)
    plt.plot(df_forces['R'], np.unwrap(df_forces['theta_F_sc']) + np.pi, label='$\\varphi(F_{sc})+\\pi$', lw=3)
    _yticks = [0, np.pi / 2, np.pi]
    _yticklabels = ['$0$', '$\\pi/2$', '$\\pi$']
    plt.yticks(_yticks, _yticklabels)
    plt.xlim(phi_mean_spp['R'].to_numpy()[0], phi_mean_spp['R'].to_numpy()[-1])
    plt.xlabel('$R$ (nm)')
    plt.ylabel('$\\varphi$ (rad)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.title(f'Scatt Force: $\\psi=${np.rad2deg(base_config_1.psi):.2f}° rad, $\\chi$={np.rad2deg(base_config_1.chi):.2f}° rad')
    plt.show()
    return


@app.cell
def _(SimulationConfig, SweepRunner, base_config_1, np, ureg):
    #Free Space
    base_config_free_space = SimulationConfig(wl=base_config_1.wl, R=295 / 2 * ureg.nanometer, dist=2 * ureg.nanometer, angle=np.deg2rad(0), a_angle=0, phase=0, substrate='Air')
    sweep_res_free_space, _, _ = SweepRunner(base_config_free_space, 'R', np.linspace(100, 160, 185) * ureg.nanometer, True, False, True, False).run()
    F0 = np.max(np.abs(sweep_res_free_space['Fz'].to_numpy()))
    return (F0,)


@app.cell
def _(np):
    f_coms_tot = np.loadtxt('article_data_plots/tot_force.txt', delimiter=',', skiprows=8)
    f_coms_press = np.loadtxt('article_data_plots/press_force.txt', delimiter=',', skiprows=8)
    return f_coms_press, f_coms_tot


@app.cell
def _(F0, Fy_1, Fy_sc, Fyspp, df_forces, f_coms_tot, phi_mean_spp, plt):
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(df_forces['R'], Fyspp / F0, label='SPP', c='orange', lw=2)
    #plt.plot(df_plot["R"], np.unwrap(df_plot["phi_mean"])-2*np.pi, label="$\varphi_D^{max}$", lw=4)
    plt.plot(df_forces['R'], Fy_sc / F0, label='Air', c='blue', lw=2)
    #plt.plot(phi_mean_air["R"], np.unwrap(df_plot["phi_mean"])+2*np.pi, label="$\varphi (D_{air}^{max})$", lw=3)
    plt.plot(f_coms_tot[:, 0], f_coms_tot[:, 2] / F0, label='Total (Numerical)', c='black', lw=1, ls='--')
    plt.plot(df_forces['R'], Fy_1 / F0, label='Total', c='black', lw=2)
    plt.xlim(phi_mean_spp['R'].to_numpy()[0], phi_mean_spp['R'].to_numpy()[-1])
    plt.xlabel('$R$ (nm)')
    plt.ylabel('$F_y/F_0^{max}$ (1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('article_plots/Fy_vs_R_air.svg', dpi=300)
    plt.show()
    return


@app.cell
def _(
    F0,
    Fx_1,
    Fx_sc,
    Fxe0_1,
    Fxm0_1,
    Fxspp,
    df_forces,
    f_coms_press,
    f_coms_tot,
    phi_mean_spp,
    plt,
):
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(df_forces['R'], Fxspp / F0, label='SPP', c='orange', lw=2)
    #plt.plot(df_plot["R"], np.unwrap(df_plot["phi_mean"])-2*np.pi, label="$\varphi_D^{max}$", lw=4)
    plt.plot(df_forces['R'], Fx_sc / F0, label='Air', c='blue', lw=2)
    #plt.plot(phi_mean_air["R"], np.unwrap(df_plot["phi_mean"])+2*np.pi, label="$\varphi (D_{air}^{max})$", lw=3)
    plt.plot(df_forces['R'], (Fx_1 - Fxe0_1 - Fxm0_1) / F0, label='Total', c='black', lw=2)
    plt.plot(f_coms_tot[:, 0], (f_coms_tot[:, 1] - f_coms_press[:, 1] - f_coms_press[:, 2]) / F0, label='Total (Numerical)', c='black', lw=1, ls='--')
    plt.xlim(phi_mean_spp['R'].to_numpy()[0], phi_mean_spp['R'].to_numpy()[-1])
    plt.xlabel('$R$ (nm)')
    plt.ylabel('$F_x^{\text{no press}}/F_0^{max}$ (1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-0.55, 0.1)
    #plt.savefig('article_plots/Fx_vs_R_air.svg', dpi=300)
    plt.show()
    return


@app.cell
def _(SimulationConfig, SweepRunner, UnitRegistry, np):
    ureg_1 = UnitRegistry()
    base_config_2 = SimulationConfig(wl=900 * ureg_1.nanometer, R=295 / 2 * ureg_1.nanometer, dist=2 * ureg_1.nanometer, angle=np.deg2rad(25), a_angle=np.pi / 4, phase=np.pi / 2, show_warnings=False, two_beam_setup=True)
    polar_param = np.linspace(0, np.pi / 2, 200)
    sweep_res_from_a, _, _ = SweepRunner(base_config_2, 'a_angle', polar_param, True, False, True).run(n_jobs=1)
    return polar_param, sweep_res_from_a, ureg_1


@app.cell
def _(np, plt, polar_param, sweep_res_from_a):
    Fxe0_2 = sweep_res_from_a['Fxe0'].apply(lambda x: x.to('N').magnitude).to_numpy()
    Fxm0_2 = sweep_res_from_a['Fxm0'].apply(lambda x: x.to('N').magnitude).to_numpy()
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(polar_param, Fxe0_2, label='$F_x^{e0}$')
    plt.plot(polar_param, Fxm0_2, label='$F_x^{m0}$')
    plt.plot(polar_param, Fxe0_2 + Fxm0_2, label='$F_x^{0}$')
    plt.grid()
    plt.axvline(np.pi / 4)
    plt.xlabel('a angle (rad)')
    # plt.axvline(polar_param[np.argmin(np.abs(Fxe0+Fxm0))])
    # print(polar_param[np.argmin(np.abs(Fxe0+Fxm0))])
    plt.legend()
    return Fxe0_2, Fxm0_2


@app.cell
def _(np):
    force_comsol = np.loadtxt('article_data_plots/tot force comsol.txt', skiprows=8, delimiter=',')
    return (force_comsol,)


@app.cell
def _(Fx_1, R_1, force_comsol, plt):
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(R_1, Fx_1, label='Analytical', lw=3)
    plt.plot(force_comsol[:, 0], force_comsol[:, 1], label='Comsol', lw=3)
    plt.legend()
    plt.xlabel('R (nm)')
    plt.ylabel('$F_x$ (N)')
    plt.grid()
    plt.tight_layout()
    plt.xlim(10, 100)
    plt.ylim(0, 1e-25)
    #plt.savefig('article_plots/Fx_vs_R_comsol_compare_scaled.png', dpi=300)
    plt.show()
    return


@app.cell
def _(Fy_1, R_1, force_comsol, plt):
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(R_1, Fy_1, label='Analytical', lw=3)
    plt.plot(force_comsol[:, 0], force_comsol[:, 2], label='Comsol', lw=3)
    plt.legend()
    plt.xlabel('R (nm)')
    plt.ylabel('$F_y$ (N)')
    plt.grid()
    plt.tight_layout()
    plt.xlim(10, 110)
    plt.ylim(-1e-25, 1e-25)
    #plt.savefig('article_plots/Fy_vs_R_comsol_compare_scaled.png', dpi=300)
    plt.show()
    return


@app.cell
def _(np):
    from MieSppForce.force import field_dx
    from MieSppForce.dipoles import initial_field
    from MieSppForce import frenel
    p_coms = np.loadtxt('article_data_plots/p_dips.txt', skiprows=8, delimiter=',')
    m_coms = np.loadtxt('article_data_plots/m_dips.txt', skiprows=8, delimiter=',')
    px_1 = p_coms[:, 1] - 1j * p_coms[:, 2]
    py_1 = p_coms[:, 3] - 1j * p_coms[:, 4]
    pz_1 = p_coms[:, 5] - 1j * p_coms[:, 6]
    mx_1 = m_coms[:, 1] - 1j * m_coms[:, 2]
    my_1 = m_coms[:, 3] - 1j * m_coms[:, 4]
    mz_1 = m_coms[:, 5] - 1j * m_coms[:, 6]
    epsAu = frenel.get_interpolate('Au')
    dist = 2
    R_coms = p_coms[:, 0]
    dEdx = np.zeros((len(R_coms), 3), dtype=complex)
    dHdx = np.zeros((len(R_coms), 3), dtype=complex)
    for i in range(len(R_coms)):
        dEdx[i, :], dHdx[i, :] = field_dx(initial_field, 900, np.deg2rad(25), 1, epsAu, point=[0, 0, dist + R_coms[i]], phase=np.pi / 2, a_angle=np.pi / 4)
    return R_coms, dEdx, dHdx, mx_1, my_1, mz_1, px_1, py_1, pz_1


@app.cell
def _(dEdx, dHdx, mu0_const, mx_1, my_1, mz_1, np, px_1, py_1, pz_1):
    Fx_press_e = 0.5 * np.real(px_1.conj() * dEdx[:, 0] + py_1.conj() * dEdx[:, 1] + pz_1.conj() * dEdx[:, 2])
    Fx_press_m = 0.5 * np.real(mx_1.conj() * dHdx[:, 0] + my_1.conj() * dHdx[:, 1] + mz_1.conj() * dHdx[:, 2]) * mu0_const
    return Fx_press_e, Fx_press_m


@app.cell
def _(Fx_press_e, Fx_press_m, Fxe0_2, Fxm0_2, R_1, R_coms, plt):
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(R_coms, Fx_press_e + Fx_press_m, label='Comsol', lw=3)
    plt.plot(R_1, Fxe0_2 + Fxm0_2, label='Analytical', lw=3)
    plt.legend()
    plt.grid()
    plt.xlabel('R (nm)')
    plt.ylabel('$F_x^{press}$ (N)')
    plt.tight_layout()
    plt.xlim(10, 180)
    #plt.savefig('article_plots/Fx_press_vs_R_comsol_compare.png', dpi=300)
    plt.show()
    return


@app.cell
def _(
    Fx_1,
    Fx_press_e,
    Fx_press_m,
    Fxe0_2,
    Fxm0_2,
    R_1,
    R_coms,
    force_comsol,
    plt,
):
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(R_coms, force_comsol[:, 1] - Fx_press_e - Fx_press_m, label='Comsol', lw=3)
    plt.plot(R_1, Fx_1 - Fxe0_2 - Fxm0_2, label='Analytical', lw=3)
    plt.legend()
    plt.grid()
    plt.xlabel('R (nm)')
    plt.ylabel('$F_x^{sc}$ (N)')
    plt.tight_layout()
    plt.xlim(10, 180)
    #plt.savefig('article_plots/Fx_sc_vs_R_comsol_compare.png', dpi=300)
    plt.show()
    return


@app.cell
def _(SimulationConfig, np, ureg_1):
    from MieSppForce.simulation import SweepRunner2D
    base_config_3 = SimulationConfig(wl=900 * ureg_1.nanometer, R=295 / 2 * ureg_1.nanometer, dist=2 * ureg_1.nanometer, angle=np.deg2rad(25), a_angle=np.pi / 4, phase=np.pi / 2, show_warnings=False, two_beam_setup=False)
    runner = SweepRunner2D(base_config_3, primary_param='R', primary_values=np.linspace(100, 165, 200) * ureg_1.nm, secondary_param='a_angle', secondary_values=np.linspace(-np.pi / 2, np.pi / 2, 200), compute_dipoles=True, compute_diagram=False, compute_force=True, compute_fields=False, parallel_param='primary', enable_parallel=True)
    df_summary, df_diagrams, field_results = runner.run(n_jobs=-1)  # or 'secondary' / None
    return SweepRunner2D, df_summary


@app.cell
def _(df_summary, np, plt):
    from matplotlib.colors import TwoSlopeNorm

    def _to_numeric(value, unit=None):
        if hasattr(value, 'to'):
            return value.to(unit).magnitude if unit else value.magnitude
        if hasattr(value, 'magnitude'):
            return value.magnitude
        return float(value)
    heatmap_df = df_summary.copy()
    heatmap_df['R_nm'] = heatmap_df['R'].apply(lambda v: _to_numeric(v, 'nanometer'))
    heatmap_df['a_angle_rad'] = heatmap_df['a_angle'].apply(_to_numeric)
    heatmap_df['Fy_N_full'] = heatmap_df['Fy'].apply(lambda v: _to_numeric(v, 'newton'))
    heatmap_df['Fy_N_spp'] = heatmap_df['Fyspp'].apply(lambda v: _to_numeric(v, 'newton'))
    heatmap_df['Fy_N'] = heatmap_df['Fy_N_full'] - heatmap_df['Fy_N_spp']
    pivot_map = heatmap_df.pivot(index='a_angle_rad', columns='R_nm', values='Fy_N')
    pivot_map = pivot_map.sort_index().sort_index(axis=1)
    _vmax = np.nanmax(np.abs(pivot_map.values))
    _norm = TwoSlopeNorm(vmin=-_vmax, vcenter=0.0, vmax=_vmax)
    _fig, ax_1 = plt.subplots(figsize=(5, 4), dpi=300)
    _im = ax_1.imshow(pivot_map.values, origin='lower', aspect='auto', extent=[pivot_map.columns.min(), pivot_map.columns.max(), pivot_map.index.min(), pivot_map.index.max()], cmap='bwr', norm=_norm)
    _cbar = plt.colorbar(_im, ax=ax_1, label='$F_y$ (N)')
    ax_1.set_xlabel('$R$ (nm)')
    ax_1.set_ylabel('$\\beta$ (rad)')
    ax_1.set_title('$F_y$ heatmap')
    ax_1.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    ax_1.set_yticklabels(['$-\\pi/2$', '$-\\pi/4$', '0', '$\\pi/4$', '$\\pi/2$'])
    plt.tight_layout()
    plt.show()
    return (TwoSlopeNorm,)


@app.cell
def _(np):
    from scipy.optimize import fsolve

    def beta_delta_from_psichi(psi, chi):
        if not -np.pi / 4 <= chi <= np.pi / 4:
            raise ValueError('chi вне диапазона [-pi/4, pi/4]')  # Проверка диапазонов
        if not 0 <= psi <= np.pi:
            raise ValueError('psi вне диапазона [0, pi]')
        if abs(chi) < 1e-12:
            beta = psi
            delta = 0
            return (beta, delta)  # Линейная поляризация — отдельный физический случай
        A = np.tan(2 * psi)
        B = np.sin(2 * chi)
      # фаза не определена и не нужна
        def F(vars):
            beta, delta = vars
            return [A - np.tan(2 * beta) * np.cos(delta), B - np.sin(2 * beta) * np.sin(delta)]  # Численное решение в общем случае
        sol = fsolve(F, [psi / 2, 0.0])
        beta, delta = sol
        return (beta, delta)
    psi = 1.0
    chi = 0.0
    beta, delta = beta_delta_from_psichi(psi, chi)
    # Пример
    print('beta:', beta, 'delta:', delta)
    return


@app.cell
def _(np):
    from scipy.optimize import least_squares

    def beta_delta_from_psichi_1(psi, chi):
        """
        Численное решение уравнений:
            tan(2*psi) = tan(2*beta) * cos(delta)
            sin(2*chi) = sin(2*beta) * sin(delta)
        с ограничением:
            0 <= beta <= pi/2
            0 <= delta <= 2*pi
        """
        if not -np.pi / 4 <= chi <= np.pi / 4:
            raise ValueError('chi вне диапазона [-pi/4, pi/4]')
        if not 0 <= psi <= np.pi:
            raise ValueError('psi вне диапазона [0, pi]')

        def equations(vars):  # # Начальное приближение
            beta, delta = vars  # beta0 = min(psi, np.pi/2)
            eq1 = np.tan(2 * psi) - np.tan(2 * beta) * np.cos(delta)  # delta0 = (2*chi) % (2*np.pi)
            eq2 = np.sin(2 * chi) - np.sin(2 * beta) * np.sin(delta)
            return [eq1, eq2]
        res = least_squares(equations, x0=[0, 0], bounds=([0, 0], [np.pi / 2, 2 * np.pi]))
        beta, delta = res.x
        return (beta, delta)  # Используем least_squares с жесткими границами
    return


@app.cell
def _(np):
    def beta_delta_from_psichi_2(psi, chi, tol=1e-12):
        if not 0.0 <= psi <= np.pi:
            raise ValueError('psi вне диапазона [0, π]')
        if not -np.pi / 4 <= chi <= np.pi / 4:
            raise ValueError('chi вне диапазона [-π/4, π/4]')
        cos2beta = np.clip(np.cos(2 * psi) * np.cos(2 * chi), -1.0, 1.0)
        beta = 0.5 * np.arccos(cos2beta)
        sin2beta = np.sqrt(max(0.0, 1.0 - cos2beta ** 2))
        if sin2beta < tol:
            return (beta, 0.0)
        sin_delta = np.clip(np.sin(2 * chi) / sin2beta, -1.0, 1.0)  # линейная поляризация
        cos_delta = np.clip(np.tan(2 * psi) * cos2beta / sin2beta, -1.0, 1.0)
        delta = np.arctan2(sin_delta, cos_delta) % (2 * np.pi)
        return (beta, delta)
    return (beta_delta_from_psichi_2,)


@app.cell
def _(beta_delta_from_psichi_2, display, np, plt):
    import ipywidgets as widgets
    psi_slider = widgets.FloatSlider(min=0, max=np.pi, step=0.01, value=0, description='ψ')
    chi_slider = widgets.FloatSlider(min=-np.pi / 4, max=np.pi / 4, step=0.01, value=np.pi / 8, description='χ')
    out = widgets.Output()
    _fig, ax_2 = plt.subplots(figsize=(3, 3), dpi=300)
    plt.close(_fig)

    def draw_ellipse(psi, chi):
        beta, delta = beta_delta_from_psichi_2(psi, chi)
        t = np.linspace(0, 2 * np.pi, 400)
        Ex = np.cos(beta) * np.cos(t)
        Ey = np.sin(beta) * np.cos(t + delta)
        ax_2.clear()
        ax_2.plot(Ex, Ey, color='blue', lw=2, label='Эллипс поляризации')
        idx = 60
        ax_2.annotate('', xy=(Ex[idx - 4], Ey[idx - 4]), xytext=(Ex[idx], Ey[idx]), arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        handedness = 'правое' if chi > 0 else 'левое' if chi < 0 else 'линейное'
        ax_2.text(0.05, 0.9, f'Направление: {handedness}', transform=ax_2.transAxes, color='purple', fontsize=11)
        a = np.cos(chi)
        b = np.sin(abs(chi))
        major_axis = np.array([a * np.cos(psi), a * np.sin(psi)])
        minor_axis = np.array([-b * np.sin(psi), b * np.cos(psi)])
        ax_2.plot([0, major_axis[0]], [0, major_axis[1]], 'r', lw=2, label='Большая ось')
        ax_2.plot([0, minor_axis[0]], [0, minor_axis[1]], 'g', lw=2, label='Малая ось')
        ax_2.axhline(0, color='gray', lw=0.5)
        ax_2.axvline(0, color='gray', lw=0.5)
        ax_2.set_xlim(-1, 1)
        ax_2.set_ylim(-1, 1)
        ax_2.set_aspect('equal', adjustable='box')
        ax_2.set_title(f'ψ={psi:.3f}, χ={chi:.3f}, β={beta:.3f}, δ={delta:.3f}')
        ax_2.set_xlabel('TM')
        ax_2.set_ylabel('TE')
        ax_2.grid(True)
        with out:
            out.clear_output(wait=True)
            display(_fig)

    def on_change(_):
        draw_ellipse(psi_slider.value, chi_slider.value)
    for slider in (psi_slider, chi_slider):
        slider.observe(on_change, names='value')
    ui = widgets.VBox([widgets.HBox([psi_slider, chi_slider]), out])
    display(ui)
    draw_ellipse(psi_slider.value, chi_slider.value)
    return


@app.cell
def _(SimulationConfig, SweepRunner2D, UnitRegistry, np):
    ureg_2 = UnitRegistry()
    _R_values = np.linspace(100, 165, 200) * ureg_2.nanometer
    chi_values = np.linspace(-np.pi / 4, np.pi / 4, 200)
    psi_fixed = 0
    base_config_R_psi = SimulationConfig(wl=900 * ureg_2.nanometer, R=_R_values[0], dist=2 * ureg_2.nanometer, angle=np.deg2rad(25), chi=chi_values[0], psi=psi_fixed, show_warnings=False, two_beam_setup=False)
    runner_R_psi = SweepRunner2D(base_config_R_psi, primary_param='R', primary_values=_R_values, secondary_param='chi', secondary_values=chi_values, compute_dipoles=True, compute_diagram=False, compute_force=True, compute_fields=False, parallel_param='primary', enable_parallel=True)
    df_sweep_R_psi, _, _ = runner_R_psi.run(n_jobs=-1)
    df_sweep_R_psi.head()
    return df_sweep_R_psi, ureg_2


@app.cell
def _(TwoSlopeNorm, df_sweep_R_psi, np, plt):
    def _to_numeric(value, unit=None):
        if hasattr(value, 'to'):
            return value.to(unit).magnitude if unit else value.magnitude
        if hasattr(value, 'magnitude'):
            return value.magnitude
        return float(value)
    sweep_par_y = 'chi'
    heatmap_r_psi = df_sweep_R_psi.copy()
    heatmap_r_psi['R_nm'] = heatmap_r_psi['R'].apply(lambda v: _to_numeric(v, 'nanometer'))
    heatmap_r_psi[sweep_par_y + '_rad'] = heatmap_r_psi[sweep_par_y].apply(_to_numeric)
    heatmap_r_psi['Fy_N_full'] = heatmap_r_psi['Fy'].apply(lambda v: _to_numeric(v, 'newton'))
    heatmap_r_psi['Fy_N_spp'] = heatmap_r_psi['Fyspp'].apply(lambda v: _to_numeric(v, 'newton'))
    heatmap_r_psi['Fy_N_air'] = heatmap_r_psi['Fy_N_full'] - heatmap_r_psi['Fy_N_spp']
    plt_comp = 'spp'
    if plt_comp == 'air':
        pivot_r_psi = heatmap_r_psi.pivot(index=sweep_par_y + '_rad', columns='R_nm', values='Fy_N_air')
    elif plt_comp == 'spp':
        pivot_r_psi = heatmap_r_psi.pivot(index=sweep_par_y + '_rad', columns='R_nm', values='Fy_N_spp')
    elif plt_comp == 'tot':
        pivot_r_psi = heatmap_r_psi.pivot(index=sweep_par_y + '_rad', columns='R_nm', values='Fy_N_full')
    pivot_r_psi_full = heatmap_r_psi.pivot(index=sweep_par_y + '_rad', columns='R_nm', values='Fy_N_full')
    pivot_r_psi_full = pivot_r_psi_full.sort_index().sort_index(axis=1)
    pivot_r_psi = pivot_r_psi.sort_index().sort_index(axis=1)
    _vmax = np.nanmax(np.abs(pivot_r_psi_full.values)) or 1.0
    _norm = TwoSlopeNorm(vmin=-_vmax, vcenter=0.0, vmax=_vmax)
    _fig, ax_3 = plt.subplots(figsize=(5, 4), dpi=300)
    _im = ax_3.imshow(pivot_r_psi.values, origin='lower', aspect='auto', extent=[pivot_r_psi.columns.min(), pivot_r_psi.columns.max(), pivot_r_psi.index.min(), pivot_r_psi.index.max()], cmap='bwr', norm=_norm)
    _cbar = plt.colorbar(_im, ax=ax_3, label='$F_y^{spp}$ (N)')
    ax_3.set_xlabel('$R$ (nm)')
    ax_3.set_ylabel(f'$\\{sweep_par_y}$ (rad)')
    ax_3.set_title(f'$F_y^{{{plt_comp}}}$ vs $R$ and $\\chi$')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fixed R changing psi
    """)
    return


@app.cell
def _(np, plt):
    def plot_directivity_heatmap(df, param_col, param_label, max_method='complex', val_col='D', title_suffix='', cmap='hot'):
        df_clean = df.copy()
        for _col in [param_col, 'phi', val_col]:
            if _col in df_clean.columns:
                df_clean[_col] = df_clean[_col].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)
        _pivot = df_clean.pivot(index='phi', columns=param_col, values=val_col)
        param_values = _pivot.columns.values.astype(float)
        _phi_values = _pivot.index.values.astype(float)
        max_phi_list = []
        for p_val in param_values:
            col_data = _pivot[p_val].values.astype(float)
            if max_method == 'complex':
                _complex_sum = np.sum(col_data * np.exp(1j * _phi_values))
                _phi_mean = np.angle(_complex_sum)
            elif max_method == 'max':
                np.argmax_val = np.argmax(col_data)
                _phi_mean = _phi_values[np.argmax_val]
            if _phi_mean < 0:
                _phi_mean = _phi_mean + 2 * np.pi
            max_phi_list.append(_phi_mean)
        _fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
        _im = ax.imshow(_pivot.values.astype(float), aspect='auto', origin='lower', extent=[float(param_values.min()), float(param_values.max()), float(_phi_values.min()), float(_phi_values.max())], cmap=cmap, vmin=0)
        ax.plot(param_values, max_phi_list, color='cyan', linewidth=2, ls='--', label='$\\varphi_{max}$')
        _cbar = plt.colorbar(_im, ax=ax)
        _cbar.set_label(f'${val_col}$')
        ax.set_xlabel(param_label)
        ax.set_ylabel('$\\varphi$ (rad)')
        ax.set_title(f'Directivity Map {title_suffix}')
        _yticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        _yticklabels = ['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$']
        ax.set_yticks(_yticks)
        ax.set_yticklabels(_yticklabels)
        ax.set_ylim(0, 2 * np.pi)
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
    return (plot_directivity_heatmap,)


@app.cell
def _(SimulationConfig, np, ureg_2):
    base_config_psi = SimulationConfig(wl=900 * ureg_2.nanometer, R=295 / 2 * ureg_2.nanometer, dist=2 * ureg_2.nanometer, angle=np.deg2rad(25), chi=np.pi / 4, psi=0, show_warnings=False, two_beam_setup=False)
    return (base_config_psi,)


@app.cell
def _(CylindricalGrid, SweepRunner, base_config_psi, np, ureg_2):
    gridCylXoY_reg_psi = CylindricalGrid(base_config_psi.wl, np.linspace(0, 2 * np.pi, 100) * ureg_2.rad, base_config_psi.R + base_config_psi.dist)
    sweep_psi, diagrams_psi_spp, _ = SweepRunner(base_config_psi, 'chi', np.linspace(-np.pi / 4, np.pi / 4, 100), True, True, True, False, None, 'spp').run(n_jobs=-1)
    sweep_psi_reg, diagrams_psi_reg, _ = SweepRunner(base_config_psi, 'chi', np.linspace(-np.pi / 4, np.pi / 4, 100), True, True, True, False, gridCylXoY_reg_psi, 'air').run(n_jobs=-1)
    return diagrams_psi_reg, sweep_psi


@app.cell
def _(diagrams_psi_reg, plot_directivity_heatmap):
    plot_directivity_heatmap(diagrams_psi_reg, param_col='chi', param_label='$\\chi$ (rad)', title_suffix='(Air vs $\\chi$)', max_method='complex')
    return


@app.cell
def _(np, pd, plt):
    def plot_force_directivity_comparison(diagrams_df, sweep_df, param_col, param_label, component='spp', title_suffix='', max_method='complex', complex_par=1, ymax=np.pi / 2, ymin=0, ytick_step=np.pi / 8):
        from fractions import Fraction
        df_diag = diagrams_df.copy()
        for _col in [param_col, 'phi', 'D']:
            if _col in df_diag.columns:
                df_diag[_col] = df_diag[_col].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x)

        def get_angle_metric(df):
            unique_vals = df[param_col].unique()
            result = []
            for val in unique_vals:
                sub = df[df[param_col] == val]
                phi = sub['phi'].values.astype(float)
                D = sub['D'].values.astype(float)
                if max_method == 'complex':
                    _complex_sum = np.sum(D ** complex_par * np.exp(1j * phi))
                    _phi_mean = np.angle(_complex_sum)
                elif max_method == 'max':
                    idx = np.argmax(D)
                    _phi_mean = phi[idx]
                else:
                    raise ValueError(f'Unknown max_method: {max_method}')
                if _phi_mean < 0:
                    _phi_mean = _phi_mean + 2 * np.pi
                result.append([val, _phi_mean])
            return pd.DataFrame(result, columns=[param_col, 'phi_mean']).sort_values(by=param_col)
        df_phi_mean = get_angle_metric(df_diag)
        force_data = sweep_df.copy()

        def get_val(col):
            return force_data[_col].apply(lambda x: x.to('N').magnitude if hasattr(x, 'to') else x).to_numpy()
        param_vals = force_data[param_col].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x).to_numpy()
        Fx = get_val('Fx')
        Fy = get_val('Fy')
        Fxe0 = get_val('Fxe0')
        Fxm0 = get_val('Fxm0')
        Fxspp = get_val('Fxspp')
        Fyspp = get_val('Fyspp')
        if component == 'spp':
            Fx_target = Fxspp
            Fy_target = Fyspp
            label_D = 'D_{spp}^{max}'
            label_F = 'F_{spp}'
        elif component == 'sc' or component == 'air':
            Fx_target = Fx - Fxspp - Fxe0 - Fxm0
            Fy_target = Fy - Fyspp
            label_D = 'D_{air}^{max}'
            label_F = 'F_{sc}'
        else:
            Fx_target = Fx
            Fy_target = Fy
            label_D = 'D_{tot}^{max}'
            label_F = 'F_{tot}'
        theta_F = np.arctan2(Fy_target, Fx_target)
        df_forces = pd.DataFrame({param_col: param_vals, 'theta_F': theta_F})
        df_plot = pd.merge(df_phi_mean, df_forces, on=param_col, how='inner')
        x = df_plot[param_col].values
        y_D = np.unwrap(df_plot['phi_mean'].values)
        y_F = np.unwrap(df_plot['theta_F'].values)
        plt.figure(figsize=(4, 3), dpi=300)
        plt.plot(x, y_D, label=f'$\\varphi({label_D})$', lw=3)
        plt.plot(x, y_F + np.pi, label=f'$\\varphi({label_F}) + \\pi$', lw=3)
        _yticks = np.arange(ymin, ymax + ytick_step, ytick_step)
        _yticklabels = [f'${Fraction(tick / np.pi).limit_denominator()}\\pi$' if tick != 0 else '$0$' for tick in _yticks]
        plt.yticks(_yticks, _yticklabels)
        plt.xlim(x[0], x[-1])
        plt.xlabel(param_label)
        plt.ylabel('$\\varphi$ (rad)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.title(f'Force vs Directivity {title_suffix}')
        plt.show()
    return (plot_force_directivity_comparison,)


@app.cell
def _(diagrams_psi_reg, np, plot_force_directivity_comparison, sweep_psi):
    plot_force_directivity_comparison(diagrams_psi_reg, sweep_psi, param_col='chi', param_label='$\\chi$ (rad)', component='air', title_suffix='(Air vs Psi)', max_method='complex', complex_par=1, ymax=5 * np.pi / 2, ymin=0, ytick_step=np.pi / 4)
    return


@app.cell
def _(plt):
    def plot_force_xy(sweep_df, param_col, param_label, component='spp', title_suffix=''):
        force_data = sweep_df.copy()

        def get_val(col):
            return force_data[_col].apply(lambda x: x.to('N').magnitude if hasattr(x, 'to') else x).to_numpy()
        param_vals = force_data[param_col].apply(lambda x: x.magnitude if hasattr(x, 'magnitude') else x).to_numpy()
        Fx = get_val('Fx')
        Fy = get_val('Fy')
        Fxe0 = get_val('Fxe0')
        Fxm0 = get_val('Fxm0')
        Fxspp = get_val('Fxspp')
        Fyspp = get_val('Fyspp')
        if component == 'spp':
            Fx_target = Fxspp
            Fy_target = Fyspp
        elif component == 'sc' or component == 'air':
            Fx_target = Fx - Fxspp - Fxe0 - Fxm0
            Fy_target = Fy - Fyspp
        else:
            Fx_target = Fx
            Fy_target = Fy
        plt.figure(figsize=(4, 3), dpi=300)
        plt.plot(param_vals, Fx_target, label=f'$F_x^{{{component}}}$', lw=3)
        plt.plot(param_vals, Fy_target, label=f'$F_y^{{{component}}}$', lw=3)
        plt.xlim(param_vals[0], param_vals[-1])
        plt.xlabel(param_label)
        plt.ylabel('$F$ (N)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.title(f'Force {title_suffix}')
        plt.show()
    return (plot_force_xy,)


@app.cell
def _(plot_force_xy, sweep_psi):
    plot_force_xy(
            sweep_psi, 
            param_col='chi', 
            param_label='$\\chi$ (rad)', 
            component='tot',
            title_suffix="(Air vs Psi)",
        )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
