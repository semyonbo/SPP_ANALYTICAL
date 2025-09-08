from tqdm import tqdm
import numpy as np
import pandas as pd
import pint
from MieSppForce import frenel, dipoles, force, fields
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union
import numpy as np
from scipy.integrate import trapezoid

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
pint.set_application_registry(ureg)
ureg.setup_matplotlib(True)


@dataclass
class CylindricalGrid:
    r: Q_
    phi: Q_
    z: Q_

    def generate_points(self) -> np.ndarray:
        r_vals = self.r.to('nm').magnitude
        phi_vals = self.phi.to('rad').magnitude
        z_vals = self.z.to('nm').magnitude

        rr, pp, zz = np.meshgrid(r_vals, phi_vals, z_vals, indexing='ij')
        points = np.stack([rr.ravel(), pp.ravel(), zz.ravel()], axis=1)
        return points
    

class SimulationConfig:
    def __init__(self, wl, R, dist, angle, a_angle, phase, substrate='Au', particle='Si', STOP=35):
        self.wl = wl
        self.R = R
        self.dist = dist
        self.angle = angle
        self.a_angle = a_angle
        self.phase = phase
        self.substrate = substrate
        self.particle = particle
        self.STOP = STOP
        self.amplitude = 1.0
        self.eps_particle = frenel.get_interpolate(particle)
        
        
        if substrate == 'Air':
            self.eps_substrate = lambda wl: 1 + 0j
        else:
            self.eps_substrate = frenel.get_interpolate(substrate)

        self.c_const = 299792458 * ureg.meter / ureg.second
        self.eps0_const = 1/(4*np.pi*self.c_const**2)*1e7 * \
            ureg.farad / ureg.meter
        self.mu0_const = 4*np.pi * 1e-7 * ureg.newton / \
            (ureg.ampere**2)

    def k0(self):
        return (2 * np.pi / self.wl)

    def omega(self):
        return (2 * np.pi * self.c_const / self.wl)

    def point0(self):
        return [0, 0, (self.dist + self.R).to('nm').magnitude]

    def get_eps_particle(self):
        return frenel.get_interpolate(self.particle)(self.wl)

    def get_eps_substrate(self):
        return frenel.get_interpolate(self.substrate)(self.wl)


class SimulationResult:
    def __init__(self, config: SimulationConfig, data: pd.DataFrame):
        self.config = config
        self.data = data

    def to_csv(self, filename: str):
        df_serialized = self.data.copy()
        for col in df_serialized.columns:
            if isinstance(df_serialized[col].iloc[0], ureg.Quantity):
                df_serialized[col] = df_serialized[col].apply(lambda x: f"{x.magnitude} {x.units}")
        df_serialized.to_csv(filename, index=False)

    @classmethod
    def from_csv(cls, config: SimulationConfig, filename: str):
        raw_df = pd.read_csv(filename)
        df_parsed = raw_df.copy()

        for col in raw_df.columns:
            try:
                first_val = raw_df[col].iloc[0]
                if isinstance(first_val, str) and any(u in first_val for u in ureg):
                    df_parsed[col] = raw_df[col].apply(lambda x: Q_(x))
            except Exception:
                pass

        return cls(config, df_parsed)

    def __repr__(self):
        summary = self.data.describe(include='all')
        return f"SimulationResult Summary:\n{summary}"


@dataclass
class DipoleResult:
    p: np.ndarray  # shape (3,), in C·m
    m: np.ndarray  # shape (3,), in A·m²

    def as_dict(self) -> dict:
        return {
            'px': Q_(self.p[0], ureg.coulomb * ureg.meter),
            'py': Q_(self.p[1], ureg.coulomb * ureg.meter),
            'pz': Q_(self.p[2], ureg.coulomb * ureg.meter),
            'mx': Q_(self.m[0], ureg.ampere * ureg.meter**2),
            'my': Q_(self.m[1], ureg.ampere * ureg.meter**2),
            'mz': Q_(self.m[2], ureg.ampere * ureg.meter**2),
        }

    def __repr__(self):
        d = self.as_dict()
        return "\n".join([f"{k}: {v.magnitude.real:.3e} + {v.magnitude.imag:.3e}j [{v.units:~}]" for k, v in d.items()])
    
@dataclass
class DiagramResult:
    phi: np.ndarray
    D: np.ndarray
    
    def as_dict(self) -> dict:
        result = []
        for i in range(len(self.phi)):
            row = {
                'phi': self.phi[i],
                'D': self.D[i]
            }
            result.append(row)
        return pd.DataFrame(result)

    def as_array(self) -> np.ndarray:
        res = np.zeros((len(self.phi),2))
        res[:,0] = self.phi
        res[:,1] = self.D
        return res

@dataclass
class OpticalForceResult:
    Fx: np.ndarray
    Fy: np.ndarray
    Fz: np.ndarray
    Fx0: np.ndarray
    Fy0: np.ndarray
    Fxspp: np.ndarray
    Fyspp: np.ndarray

    def as_dict(self) -> dict:
        N = ureg.newton
        return {
            'Fx': Q_(self.Fx[0], N),
            'Fxe0': Q_(self.Fx[1], N),
            'Fxe1': Q_(self.Fx[2], N),
            'Fxe2': Q_(self.Fx[3], N),
            'Fxm0': Q_(self.Fx[4], N),
            'Fxm1': Q_(self.Fx[5], N),
            'Fxm2': Q_(self.Fx[6], N),
            'Fxcross': Q_(self.Fx[7], N),
            
            'Fy': Q_(self.Fy[0], N),
            'Fye0': Q_(self.Fy[1], N),
            'Fye1': Q_(self.Fy[2], N),
            'Fye2': Q_(self.Fy[3], N),
            'Fym0': Q_(self.Fy[4], N),
            'Fym1': Q_(self.Fy[5], N),
            'Fym2': Q_(self.Fy[6], N),
            'Fycross': Q_(self.Fy[7], N),
            
            'Fz': Q_(self.Fz[0], N),
            'Fze0': Q_(self.Fz[1], N),
            'Fze1': Q_(self.Fz[2], N),
            'Fze2': Q_(self.Fz[3], N),
            'Fzm0': Q_(self.Fz[4], N),
            'Fzm1': Q_(self.Fz[5], N),
            'Fzm2': Q_(self.Fz[6], N),
            'Fzcross': Q_(self.Fz[7], N),
            
            
            'Fxspp': Q_(self.Fxspp[0], N),
            'Fxsppe1': Q_(self.Fxspp[2], N),
            'Fxsppe2': Q_(self.Fxspp[3], N),
            'Fxsppm1': Q_(self.Fxspp[5], N),
            'Fxsppm2': Q_(self.Fxspp[6], N),
            
            'Fyspp': Q_(self.Fyspp[0], N),
            'Fysppe1': Q_(self.Fyspp[2], N),
            'Fysppe2': Q_(self.Fyspp[3], N),
            'Fysppm1': Q_(self.Fyspp[5], N),
            'Fysppm2': Q_(self.Fyspp[6], N)
        }

    def __repr__(self):
        d = self.as_dict()
        return "\n".join([f"{k}: {v.magnitude.real:.3e} + {v.magnitude.imag:.3e}j [{v.units:~}]" for k, v in d.items()])
    
@dataclass
class FieldResult:
    df: pd.DataFrame

    def to_csv(self, filename: str):
        df_serialized = self.df.copy()
        for col in df_serialized.columns:
            if isinstance(df_serialized[col].iloc[0], ureg.Quantity):
                df_serialized[col] = df_serialized[col].apply(lambda x: f"{x.magnitude} {x.units}")
        df_serialized.to_csv(filename, index=False)

    @classmethod
    def from_csv(cls, filename: str):
        df = pd.read_csv(filename)
        for col in df.columns:
            try:
                val = df[col].iloc[0]
                if isinstance(val, str) and any(u in val for u in ureg):
                    df[col] = df[col].apply(lambda x: Q_(x))
            except Exception:
                pass
        return cls(df)

    def __repr__(self):
        return f"FieldResult: {self.df.shape[0]} points\nColumns: {list(self.df.columns)}"


class DipoleCalculator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute(self):
        p, m = dipoles.calc_dipoles_v2(wl = self.config.wl.to('nm').magnitude, 
                                       eps_Au = self.config.eps_substrate, 
                                       point = self.config.point0(), 
                                       R = self.config.R.to('nm').magnitude, 
                                       eps_Si = self.config.eps_particle, 
                                       alpha = self.config.angle, 
                                       amplitude = self.config.amplitude, 
                                       phase = self.config.phase, 
                                       a_angle = self.config.a_angle, 
                                       stop = self.config.STOP)

        p_vec = p[:, 0]
        m_vec = m[:, 0]

        return DipoleResult(p=p_vec, m=m_vec)


class OpticalForceCalculator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute(self):
        fx, fy, fz = force.F(wl=self.config.wl.to('nm').magnitude,
                             eps_Au=self.config.eps_substrate,
                             point=self.config.point0(),
                             R=self.config.R.to('nm').magnitude,
                             eps_si=self.config.eps_particle,
                             alpha=self.config.angle,
                             amplitude=1,
                             phase=self.config.phase,
                             a_angle=self.config.a_angle,
                             stop=self.config.STOP,
                             full_output=True,
                             stop_dipoles=self.config.STOP)
        
        fx0, fy0, _ = force.F(wl=self.config.wl.to('nm').magnitude,
                             eps_Au=self.config.eps_substrate,
                             point=self.config.point0(),
                             R=self.config.R.to('nm').magnitude,
                             eps_si=self.config.eps_particle,
                             alpha=self.config.angle,
                             amplitude=1,
                             phase=self.config.phase,
                             a_angle=self.config.a_angle,
                             stop=1,
                             full_output=True,
                             stop_dipoles=self.config.STOP)
        
        fxspp, fyspp = fx-fx0, fy-fy0

        return OpticalForceResult(Fx=fx, Fy=fy, Fz=fz, Fx0 = fx0, Fy0 = fy0, Fxspp=fxspp, Fyspp=fyspp)
    
class FieldsCalculator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute(self, grid: CylindricalGrid, field_type=None) -> FieldResult:

        points = grid.generate_points()
        results = []

        for r_val, phi_val, z_val in tqdm(points):
            E, H = fields.get_field(
                wl=self.config.wl.to('nm').magnitude,
                eps_interp=self.config.eps_substrate,
                alpha=self.config.angle,
                phase=self.config.phase,
                a_angle=self.config.a_angle,
                stop=self.config.STOP,
                eps_particle=self.config.eps_particle,
                R=self.config.R.to('nm').magnitude,
                r=r_val,
                phi=phi_val,
                z=z_val,
                z0=(self.config.dist + self.config.R).to('nm').magnitude,
                field_type=field_type
            )
            
            
            sin_phi = np.sin(phi_val)
            cos_phi = np.cos(phi_val)

            Hphi = -sin_phi * H[0] + cos_phi * H[1]
            Hphi_abs2 = np.abs(Hphi)**2  # |Hphi|^2

            row = {
                'r': Q_(r_val, 'nm'),
                'phi': Q_(phi_val, 'rad'),
                'z': Q_(z_val, 'nm'),
                'Ex': Q_(E[0], 'V/m'),
                'Ey': Q_(E[1], 'V/m'),
                'Ez': Q_(E[2], 'V/m'),
                'Hx': Q_(H[0], 'A/m'),
                'Hy': Q_(H[1], 'A/m'),
                'Hz': Q_(H[2], 'A/m'),
                'Hphi_abs2': Q_(Hphi_abs2, 'A^2/m^2')
            }
            results.append(row)

        df = pd.DataFrame(results)
        return FieldResult(df)
    
class DiagramCalculator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = CylindricalGrid(self.config.wl*20, 
                                    np.linspace(0,2*np.pi,300)*ureg.rad,
                                    np.array([0])*ureg.nm)
    
    def compute(self):
        FarField = FieldsCalculator(self.config).compute(self.grid, 'sub')
        Hphi_abs2 = FarField.df['Hphi_abs2'].apply(lambda x: x.magnitude)
        phi = FarField.df['phi'].apply(lambda x: x.magnitude)
        integr = trapezoid(Hphi_abs2, phi)
        D = Hphi_abs2.apply(lambda x: 2*np.pi*x/integr)
        return DiagramResult(phi, D)
    

class SweepRunner:
    def __init__(
        self,
        base_config: SimulationConfig,
        sweep_param: str,
        sweep_values,
        compute_dipoles: bool = True,
        compute_diagram: bool = True,
        compute_force: bool = False,
        compute_fields: bool = False,
        grid: CylindricalGrid = None
    ):
        self.base_config = base_config
        self.param = sweep_param
        self.values = sweep_values
        self.compute_dipoles = compute_dipoles or compute_force
        self.compute_diagram = compute_diagram
        self.compute_force = compute_force
        self.compute_fields = compute_fields
        self.grid = grid

        if self.compute_fields and self.grid is None:
            raise ValueError("Для вычисления полей необходимо передать `grid`.")

    def run(self):
        summary_results = []
        diagrams_records = []
        fields_results = {}

        for val in tqdm(self.values, desc=f"Sweeping '{self.param}'", unit="step"):
            setattr(self.base_config, self.param, val)
            row = {self.param: val}

            if self.compute_dipoles:
                dip_res = DipoleCalculator(self.base_config).compute()
                row.update(dip_res.as_dict())
                
            if self.compute_diagram:
                diag_res = DiagramCalculator(self.base_config).compute()
                df_diag = diag_res.as_dict()
                df_diag[self.param] = val
                diagrams_records.append(df_diag)

            if self.compute_force:
                force_result = OpticalForceCalculator(self.base_config).compute()
                row.update(force_result.as_dict())

            if self.compute_fields:
                field_result = FieldsCalculator(self.base_config).compute(self.grid)
                fields_results[val] = field_result

            summary_results.append(row)

        df_summary = pd.DataFrame(summary_results)
        df_diagrams = pd.concat(diagrams_records, ignore_index=True) if self.compute_diagram else None

        return df_summary, df_diagrams, fields_results if self.compute_fields else None

class Visualizer:
    @staticmethod
    def _extract_label_with_units(series: pd.Series, label: str) -> str:
        if isinstance(series.iloc[0], ureg.Quantity):
            unit = series.iloc[0].units
            return f"${label}$ [{unit:~}]"
        return f"${label}$"

    @staticmethod
    def _magnitude(series: pd.Series) -> np.ndarray:
        if hasattr(series.iloc[0], 'magnitude'):
            return series.apply(lambda x: x.magnitude), f" [{series.iloc[0].units:~L}]"
        else:
            return series, ""

    @staticmethod
    def plot_component(df: pd.DataFrame, sweep_param: str, component: str):
        import matplotlib.pyplot as plt
        y_vals, unit_y = Visualizer._magnitude(df[component])
        x_vals, unit_x = Visualizer._magnitude(df[sweep_param])

        plt.figure(figsize=(8, 5))

        if np.iscomplexobj(y_vals.iloc[0]):
            plt.plot(x_vals, y_vals.apply(np.real), label=f'{component} (Re)')
            plt.plot(x_vals, y_vals.apply(np.imag), '--', label=f'{component} (Im)')
        else:
            plt.plot(x_vals, y_vals, label=component)

        plt.xlabel(f"{sweep_param} ${unit_x}$")
        plt.ylabel(f"{component} ${unit_y}$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def list_components(df: pd.DataFrame, include_real_only=False):
        if include_real_only:
            return [col for col in df.columns if not np.iscomplexobj(df[col])]
        return list(df.columns)
    

    from typing import Union
    @staticmethod
    def plot_components(df: pd.DataFrame, sweep_param: str, components: Union[str, list]):
        import matplotlib.pyplot as plt

        if isinstance(components, str):
            components = components.split()

        x_vals, unit_x = Visualizer._magnitude(df[sweep_param])

        y_unit = None
        plt.figure(figsize=(10, 6))

        for comp in components:
            if comp not in df.columns:
                raise ValueError(f"Компонента '{comp}' отсутствует в DataFrame.")

            series = df[comp]
            
            y_vals, unit = Visualizer._magnitude(series)

            if y_unit is None:
                y_unit = unit
            elif unit != y_unit:
                raise ValueError(f"Несовместимые размерности: '{comp}' имеет {unit}, ожидалось {y_unit}.")

            if np.iscomplexobj(y_vals.iloc[0]):
                plt.plot(x_vals, y_vals.apply(np.real), label=f'{comp} (Re)')
                plt.plot(x_vals, y_vals.apply(np.imag), '--', label=f'{comp} (Im)')
            else:
                plt.plot(x_vals, y_vals, label=comp)

        plt.xlabel(f"{sweep_param} ${unit_x}$")
        plt.ylabel(f"Значение ${y_unit}$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def list_components(df: pd.DataFrame, include_real_only=False):
        if include_real_only:
            return [col for col in df.columns if not np.iscomplexobj(df[col])]
        return list(df.columns)