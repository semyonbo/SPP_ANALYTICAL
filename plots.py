import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from MieSppForce.simulation import SimulationConfig, SweepRunner,  DipoleCalculator, OpticalForceCalculator, DiagramCalculator, CylindricalGrid, SphericalGrid
    import matplotlib.pyplot as plt
    from scipy.integrate import trapezoid
    import numpy as np
    from numpy import pi, sin, cos, abs, exp, sum
    from pint import UnitRegistry
    ureg = UnitRegistry()
    import matplotlib.style
    import marimo as mo
    return (
        CylindricalGrid,
        DiagramCalculator,
        SimulationConfig,
        SphericalGrid,
        mo,
        np,
        pi,
        plt,
        ureg,
    )


@app.cell
def _(mo):
    get_chi, set_chi = mo.state(0)
    get_psi, set_psi = mo.state(0)
    get_R, set_R = mo.state(147.5)
    return get_R, get_chi, get_psi, set_R, set_chi, set_psi


@app.cell
def _(get_R, get_chi, get_psi, mo, set_R, set_chi, set_psi):
    slider_chi = mo.ui.slider(-45, 45, step=0.1, label="chi", value=get_chi(), on_change=set_chi)
    slider_psi = mo.ui.slider(0, 180, step=0.1, label="psi", value=get_psi(), on_change=set_psi)
    slider_R = mo.ui.slider(80, 165, step=0.5, label="R", value=get_R(), on_change=set_R)
    return slider_R, slider_chi, slider_psi


@app.cell
def _(get_R, get_chi, get_psi, mo, set_R, set_chi, set_psi):
    input_chi = mo.ui.number(value=get_chi(), on_change=lambda x: set_chi(x),step=0.01, start=-45, stop=45)
    input_psi = mo.ui.number(value=get_psi(), on_change=lambda x: set_psi(x), step=0.01, start=0, stop=180)
    input_R = mo.ui.number(value=get_R(), on_change=lambda x: set_R(x), step=0.5, start=80, stop=165)
    return input_R, input_chi, input_psi


@app.cell
def _(SimulationConfig, np, slider_R, slider_chi, slider_psi, ureg):
    def get_mean_angle(I, phi):
        vec_sum = np.sum(I * np.exp(1j * phi))
        return np.angle(vec_sum)

    def add_axis_line(ax, theta, label, color="k"):
        r_arrow = ax.get_rmax()
        ax.plot([theta, theta], [0, r_arrow], color=color, lw=2, ls='--')
        ax.text(theta+0.1, r_arrow*1.1, label, ha='center', va='center', fontsize=12, fontweight='bold', color=color)



    base_config = SimulationConfig(
        wl = 900 * ureg.nanometer,
        R =  slider_R.value * ureg.nanometer,
        dist = 2 * ureg.nanometer,
        angle = np.deg2rad(25),
        psi = np.deg2rad(slider_psi.value),
        chi = np.deg2rad(slider_chi.value),
        show_warnings=False,
        two_beam_setup=False
    )
    return (base_config,)


@app.cell
def _(CylindricalGrid, SphericalGrid, base_config, np, pi, ureg):
    farfield_grid_multiplyer = 1


    XOZ = SphericalGrid(base_config.wl*farfield_grid_multiplyer/2,
                         np.linspace(-np.pi/2, np.pi/2, 50)*ureg.rad,
                         0*ureg.rad)

    YOZ = SphericalGrid(base_config.wl*farfield_grid_multiplyer/2,
                        np.linspace(-np.pi/2, np.pi/2, 50)*ureg.rad,
                        pi/2*ureg.rad)

    XOY_SPP = CylindricalGrid(phi=np.linspace(0, 2*pi, 50)*ureg.rad, 
                              z=0*ureg.nm, 
                              r=base_config.wl*farfield_grid_multiplyer)

    XOY_AIR = CylindricalGrid(phi=np.linspace(0, 2*pi, 50)*ureg.rad, 
                              z=5*(base_config.dist+base_config.R),
                              r=base_config.wl*1)
    return XOY_AIR, XOY_SPP, XOZ, YOZ


@app.cell
def _(np):
    def plot_diagram(fig, ax, Diagram_res, label):
        ax.plot(Diagram_res.as_array()[:,0], Diagram_res.as_array()[:,1], lw=2, label=label)

    def plot_params(fig, ax, proj, lims_x=None, lims_y=None):
        ax.legend(loc='lower left')
        if proj == 'XOZ' or proj == 'YOZ':
            ax.set_title(proj)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N") 
        else: 
            ax.set_title(proj)
            ax.set_xlim(0, 2*np.pi)
    return plot_diagram, plot_params


@app.cell(hide_code=True)
def _(DiagramCalculator, XOY_AIR, XOY_SPP, XOZ, YOZ, base_config):
    Diag_reg_XOZ = DiagramCalculator(base_config, XOZ, normalize=None).compute('reg')
    Diag_air_XOZ = DiagramCalculator(base_config, XOZ, normalize=None).compute('air')
    Diag_sc_XOZ = DiagramCalculator(base_config, XOZ, normalize=None).compute('sc')
    Diag_spp_XOZ = DiagramCalculator(base_config, XOZ, normalize=None).compute('spp')

    Diag_reg_YOZ = DiagramCalculator(base_config, YOZ, normalize=None).compute('reg')
    Diag_air_YOZ = DiagramCalculator(base_config, YOZ, normalize=None).compute('air')
    Diag_sc_YOZ = DiagramCalculator(base_config, YOZ, normalize=None).compute('sc')
    Diag_spp_YOZ = DiagramCalculator(base_config, YOZ, normalize=None).compute('spp')

    Diag_reg_XOY = DiagramCalculator(base_config, XOY_AIR, normalize=None).compute('reg')
    Diag_air_XOY = DiagramCalculator(base_config, XOY_AIR, normalize=None).compute('air')
    Diag_sc_XOY = DiagramCalculator(base_config, XOY_AIR, normalize=None).compute('sc')
    Diag_spp_XOY = DiagramCalculator(base_config, XOY_SPP, normalize=None).compute('spp')
    return (
        Diag_air_XOY,
        Diag_air_XOZ,
        Diag_air_YOZ,
        Diag_reg_XOY,
        Diag_reg_XOZ,
        Diag_reg_YOZ,
        Diag_sc_XOY,
        Diag_sc_XOZ,
        Diag_sc_YOZ,
        Diag_spp_XOY,
        Diag_spp_XOZ,
        Diag_spp_YOZ,
    )


@app.cell
def _(input_R, input_chi, input_psi, mo, slider_R, slider_chi, slider_psi):
    chi_hor = mo.hstack([
        slider_chi,
        input_chi,
        mo.md('chi'),
    ])

    psi_hor = mo.hstack([
        slider_psi,
        input_psi,
        mo.md('psi'),
    ])

    R_hor = mo.hstack([
        slider_R,
        input_R,
        mo.md('R'),
    ])


    params_mgmt = mo.vstack(
        [psi_hor,
         chi_hor,
         R_hor
        ])
    return (params_mgmt,)


@app.cell
def _(
    Diag_air_XOY,
    Diag_air_XOZ,
    Diag_air_YOZ,
    Diag_reg_XOY,
    Diag_reg_XOZ,
    Diag_reg_YOZ,
    Diag_sc_XOY,
    Diag_sc_XOZ,
    Diag_sc_YOZ,
    Diag_spp_XOY,
    Diag_spp_XOZ,
    Diag_spp_YOZ,
    plot_diagram,
    plot_params,
    plt,
):
    fig, [ax1, ax2, ax3]= plt.subplots(1,3, dpi=150, figsize = (14,7), subplot_kw={'projection': 'polar'})


    plot_diagram(fig, ax1, Diag_air_XOZ, '$I_{air}$')
    plot_diagram(fig, ax1, Diag_reg_XOZ, '$I_{reg}$')
    plot_diagram(fig, ax1, Diag_sc_XOZ, '$I_{sc}$')
    plot_diagram(fig, ax1, Diag_spp_XOZ,'$I_{spp}$')

    plot_diagram(fig, ax2, Diag_air_YOZ, '$I_{air}$')
    plot_diagram(fig, ax2, Diag_reg_YOZ, '$I_{reg}$')
    plot_diagram(fig, ax2, Diag_sc_YOZ, '$I_{sc}$')
    plot_diagram(fig, ax2, Diag_spp_YOZ,'$I_{spp}$')

    plot_diagram(fig, ax3, Diag_air_XOY, '$I_{air}$')
    plot_diagram(fig, ax3, Diag_reg_XOY, '$I_{reg}$')
    plot_diagram(fig, ax3, Diag_sc_XOY, '$I_{sc}$')
    plot_diagram(fig, ax3, Diag_spp_XOY,'$I_{spp}$')

    plot_params(fig, ax1, 'XOZ')
    plot_params(fig, ax2, 'YOZ')
    plot_params(fig, ax3, 'XOY')

    plt.show()
    return


@app.cell
def _(params_mgmt):
    params_mgmt
    return


@app.cell
def _():
    # sweep_reg, diagrams_reg, _ = SweepRunner(base_config, 'R', np.linspace(100, 165, 100)*ureg.nanometer, True, True, True, False, XOY_AIR, 'sc', diagram_normalize=None).run(n_jobs=-1)
    return


@app.cell
def _():
    # diagrams_reg["phi"] = diagrams_reg["phi"].apply(lambda x: x.magnitude if hasattr(x, "magnitude") else x)
    # diagrams_reg["R"]  = diagrams_reg["R"].apply(lambda x: x.magnitude if hasattr(x, "magnitude") else x)

    # pivot = diagrams_reg.pivot(index="phi", columns="R", values="D")
    # pivot = pivot

    # R_values = pivot.columns.values.astype(float)
    # phi_values = pivot.index.values.astype(float)

    # max_phi_for_r = []
    # for r in R_values:
    #     col = pivot[r].values.astype(float)
    #     # Интегральное круговое среднее:
    #     # complex_sum = np.sum(col * np.exp(1j * phi_values))
    #     # phi_mean = np.angle(complex_sum)
    #     phi_mean = phi_values[np.argmax(col)]
    #     if phi_mean < 0:
    #         phi_mean += 2 * np.pi
    #     max_phi_for_r.append(phi_mean)


    # # Построение тепловой карты
    # plt.figure(figsize=(4, 3), dpi=300)
    # plt.imshow(
    #     pivot.values.astype(float),
    #     aspect="auto",
    #     origin="lower",
    #     extent=[
    #         float(pivot.columns.min()), float(pivot.columns.max()),
    #         float(pivot.index.min()),   float(pivot.index.max())
    #     ],
    #     cmap="hot",
    #     vmin=0
    # )

    # plt.plot(R_values, max_phi_for_r, color="lightgray", linewidth=2, label="$D_{air}^{max}$", ls='--')

    # plt.colorbar(label="$D_{air}$")
    # plt.xlabel("$R$ (nm)")
    # plt.ylabel("$\\varphi$ (rad)")

    # # Установка делений по оси Y в радианах
    # yticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    # yticklabels = ["$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"]
    # plt.yticks(yticks, yticklabels)

    # plt.legend(loc="lower left")
    # plt.tight_layout()
    # #plt.savefig('article_plots/Dair_R.svg', dpi=300)
    # plt.show()
    return


if __name__ == "__main__":
    app.run()
