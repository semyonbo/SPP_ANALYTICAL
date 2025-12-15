from tqdm import tqdm
import numpy as np
import pandas as pd
import pint
from MieSppForce import frenel, dipoles, force, fields
from dataclasses import dataclass
import numpy as np
from scipy.integrate import trapezoid
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import warnings
from scipy.integrate import IntegrationWarning


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
pint.set_application_registry(ureg)
ureg.setup_matplotlib(True)

@dataclass
class Grid:
    def generate_points(self) -> np.ndarray:
        return NotImplementedError


@dataclass
class CylindricalGrid(Grid):
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
    
@dataclass 
class SphericalGrid(Grid):
    r: Q_
    theta: Q_
    phi: Q_
    
    def translate_to_cylindrical(self) -> np.ndarray:
        r_vals = self.r.to('nm').magnitude
        theta_vals = self.theta.to('rad').magnitude
        phi_vals = self.phi.to('rad').magnitude
        
        rr, tt, pp = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')
        r_cyl = rr * np.sin(tt)
        z = rr * np.cos(tt)

        points = np.stack([r_cyl.ravel(), pp.ravel(), z.ravel()], axis=1)
        return points

    def generate_points(self) -> np.ndarray:
        return self.translate_to_cylindrical()
    

class SimulationConfig:
    def __init__(self, wl, R, dist, angle, 
                 psi=None, chi=None, beta=None, delta=None,
                 substrate='Au', particle='Si', stop=45 ,amplitude=1,show_warnings=True, initial_field_type='plane_wave'):
        self.wl = wl
        self.R = R
        self.dist = dist
        self.angle = angle
        
        self._psi = psi
        self._chi = chi
        
        # Determine beta and delta (internal storage)
        if psi is not None and chi is not None:
            self.polaris_param_type = 'ellipse'
            self.a_angle, self.phase = beta_delta_from_psichi(psi, chi)
        elif beta is not None and delta is not None:
            self.polaris_param_type = 'classic'
            self.a_angle = beta
            self.phase = delta
        else:
            raise ValueError("Polarization parameters not specified. Use (psi, chi) or (beta, delta).")
        
        self.substrate = substrate
        self.particle = particle
        self.amplitude = amplitude
        self.eps_particle = frenel.get_interpolate(particle)
        self.STOP=stop
        self.show_warnings = show_warnings 
        self.initial_field_type = initial_field_type
        
        if not self.show_warnings:
            warnings.filterwarnings("ignore", category=IntegrationWarning)
        else:
            warnings.filterwarnings("default", category=IntegrationWarning)
            
        if substrate == 'Air':
            self.eps_substrate = lambda wl: 1 + 0j
        else:
            self.eps_substrate = frenel.get_interpolate(substrate)

        self.c_const = 299792458 * ureg.meter / ureg.second
        self.eps0_const = 1/(4*np.pi*self.c_const**2)*1e7 * \
            ureg.farad / ureg.meter
        self.mu0_const = 4*np.pi * 1e-7 * ureg.newton / \
            (ureg.ampere**2)

    @property
    def psi(self):
        return self._psi
    
    @psi.setter
    def psi(self, value):
        self._psi = value
        if self._chi is not None:
            self.a_angle, self.phase = beta_delta_from_psichi(self._psi, self._chi)

    @property
    def chi(self):
        return self._chi
    
    @chi.setter
    def chi(self, value):
        self._chi = value
        if self._psi is not None:
            self.a_angle, self.phase = beta_delta_from_psichi(self._psi, self._chi)

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
    Fz0: np.ndarray
    Fxspp: np.ndarray
    Fyspp: np.ndarray
    Fzspp: np.ndarray

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
            'Fysppm2': Q_(self.Fyspp[6], N),
            
            'Fzspp': Q_(self.Fzspp[0], N)
            
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
    
    
def beta_delta_from_psichi(psi, chi, tol=1e-12):
    if not (0.0 <= psi <= np.pi):
        raise ValueError("psi вне диапазона [0, π]")
    if not (-np.pi/4 <= chi <= np.pi/4):
        raise ValueError("chi вне диапазона [-π/4, π/4]")

    cos2beta = np.clip(np.cos(2*psi) * np.cos(2*chi), -1.0, 1.0)
    beta = 0.5 * np.arccos(cos2beta)

    sin2beta = np.sqrt(max(0.0, 1.0 - cos2beta**2))
    if sin2beta < tol:  # линейная поляризация
        return beta, 0.0

    sin_delta = np.clip(np.sin(2*chi) / sin2beta, -1.0, 1.0)
    cos_delta = np.clip(np.tan(2*psi) * cos2beta / sin2beta, -1.0, 1.0)
    delta = (np.arctan2(sin_delta, cos_delta)) % (2*np.pi)
    return beta, delta


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
                                       initial_field_type=self.config.initial_field_type)

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
                             amplitude=self.config.amplitude,
                             phase=self.config.phase,
                             a_angle=self.config.a_angle,
                             stop=self.config.STOP,
                             full_output=True,
                             initial_field_type=self.config.initial_field_type)
        
        fx0, fy0, fz0 = force.F(wl=self.config.wl.to('nm').magnitude,
                             eps_Au=self.config.eps_substrate,
                             point=self.config.point0(),
                             R=self.config.R.to('nm').magnitude,
                             eps_si=self.config.eps_particle,
                             alpha=self.config.angle,
                             amplitude=self.config.amplitude,
                             phase=self.config.phase,
                             a_angle=self.config.a_angle,
                             stop=1,
                             full_output=True,
                             initial_field_type=self.config.initial_field_type)
        
        fxspp, fyspp, fzspp = fx-fx0, fy-fy0, fz-fz0

        return OpticalForceResult(Fx=fx, Fy=fy, Fz=fz, Fx0 = fx0, Fy0 = fy0, Fz0 = fz0, Fxspp=fxspp, Fyspp=fyspp, Fzspp = fzspp )
    
class FieldsCalculator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        
    # def _compute_point(self, r_val, phi_val, z_val, field_type):
    #     E, H = fields.get_field(
    #         wl=self.config.wl.to('nm').magnitude,
    #         eps_interp=self.config.eps_substrate,
    #         alpha=self.config.angle,
    #         phase=self.config.phase,
    #         a_angle=self.config.a_angle,
    #         stop=self.config.STOP,
    #         eps_particle=self.config.eps_particle,
    #         R=self.config.R.to('nm').magnitude,
    #         r=r_val,
    #         phi=phi_val,
    #         z=z_val,
    #         z0=(self.config.dist + self.config.R).to('nm').magnitude,
    #         field_type=field_type,
    #         amplitude=self.config.amplitude
    #     )

    #     sin_phi = np.sin(phi_val)
    #     cos_phi = np.cos(phi_val)

    #     Hphi = -sin_phi * H[0] + cos_phi * H[1]
    #     Hphi_abs2 = np.abs(Hphi)**2  # |Hphi|^2

    #     return {
    #         'r': Q_(r_val, 'nm'),
    #         'phi': Q_(phi_val, 'rad'),
    #         'z': Q_(z_val, 'nm'),
    #         'Ex': Q_(E[0], 'V/m'),
    #         'Ey': Q_(E[1], 'V/m'),
    #         'Ez': Q_(E[2], 'V/m'),
    #         'Hx': Q_(H[0], 'A/m'),
    #         'Hy': Q_(H[1], 'A/m'),
    #         'Hz': Q_(H[2], 'A/m'),
    #         'Hphi_abs2': Q_(Hphi_abs2, 'A^2/m^2')
    #     }


    # def compute(self, grid: Grid, field_type=None, n_jobs=-1) -> FieldResult:
    #     points = grid.generate_points()

    #     with tqdm_joblib(tqdm(total=len(points), desc="Computing fields", unit="pt")):
    #         results = Parallel(n_jobs=n_jobs)(
    #             delayed(self._compute_point)(r_val, phi_val, z_val, field_type)
    #             for r_val, phi_val, z_val in points
    #         )

    #     df = pd.DataFrame(results)
    #     return FieldResult(df)
    
    
    def compute(self, grid: Grid, field_type=None, internal_compute = False) -> FieldResult:

        points = grid.generate_points()
        results = []
        
        if internal_compute:
            smart_range = lambda x: x
        else:
            smart_range = tqdm

        for r_val, phi_val, z_val in smart_range(points):
            E, H = fields.get_field(
                wl=self.config.wl.to('nm').magnitude,
                eps_interp=self.config.eps_substrate,
                alpha=self.config.angle,
                phase=self.config.phase,
                a_angle=self.config.a_angle,
                eps_particle=self.config.eps_particle,
                R=self.config.R.to('nm').magnitude,
                r=r_val,
                phi=phi_val,
                z=z_val,
                z0=(self.config.dist + self.config.R).to('nm').magnitude,
                field_type=field_type,
                amplitude=self.config.amplitude,
                initial_field_type=self.config.initial_field_type
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
    def __init__(self, config: SimulationConfig, grid=None, normalize='directivity'):
        self.config = config
        self.normalize = normalize
        if grid == None:
            self.grid = CylindricalGrid(self.config.wl*10, 
                                    np.linspace(0,2*np.pi,300)*ureg.rad,
                                    np.array([0])*ureg.nm)
        else:
            self.grid=grid
            
    def radiation_pattern(grid: Grid , FarField : FieldResult):
        Ex = FarField.df['Ex'].apply(lambda x: x.magnitude).to_numpy()
        Ey = FarField.df['Ey'].apply(lambda x: x.magnitude).to_numpy()
        Ez = FarField.df['Ez'].apply(lambda x: x.magnitude).to_numpy()
        Hx = FarField.df['Hx'].apply(lambda x: x.magnitude).to_numpy()
        Hy = FarField.df['Hy'].apply(lambda x: x.magnitude).to_numpy()
        Hz = FarField.df['Hz'].apply(lambda x: x.magnitude).to_numpy()
        
        Sx = 0.5*np.real(Ey*Hz.conj() - Ez*Hy.conj())
        Sy = 0.5*np.real(Ez*Hx.conj() - Ex*Hz.conj())
        Sz = 0.5*np.real(Ex*Hy.conj() - Ey*Hx.conj())
        
        if type(grid) == SphericalGrid:
            phi = grid.phi.magnitude
            theta = grid.theta.magnitude
            I  = Sx * np.sin(theta) * np.cos(phi) + Sy * np.sin(theta) * np.sin(phi) + Sz * np.cos(theta)
        elif type(grid) == CylindricalGrid:
            theta = np.pi/2
            phi = grid.phi.magnitude
            I = Sx*np.cos(phi) + Sy*np.sin(phi)
        return I, theta, phi
        
        
        
    def compute(self, field_type=None, internal_compute = False):
    
        # if (type(self.grid) == CylindricalGrid and (field_type=='spp')):
        #     FarField = FieldsCalculator(self.config).compute(self.grid, 'spp', internal_compute=internal_compute)
        #     Hphi_abs2 = FarField.df['Hphi_abs2'].apply(lambda x: x.magnitude)
        #     phi = FarField.df['phi'].apply(lambda x: x.magnitude)
        #     if self.normalize == 'directivity':
        #         integr = trapezoid(Hphi_abs2, phi)
        #         pattern = Hphi_abs2.apply(lambda x: 2*np.pi*x/integr)
        #     elif self.normalize == None:
        #         pattern = Hphi_abs2
        #     return DiagramResult(phi, pattern)

        if type(self.grid) == SphericalGrid:
            FarField = FieldsCalculator(self.config).compute(self.grid, field_type=field_type, internal_compute=internal_compute)
            I, theta, phi = DiagramCalculator.radiation_pattern(self.grid, FarField)
            if self.normalize == 'directivity':
                integr = trapezoid(I , theta)
                pattern = 2*np.pi*I/integr
            elif self.normalize == None:
                pattern = I          
            return DiagramResult(theta, pattern)
        
        elif (type(self.grid) == CylindricalGrid):
            FarField = FieldsCalculator(self.config).compute(self.grid, field_type=field_type, internal_compute=internal_compute)
            I, theta, phi = DiagramCalculator.radiation_pattern(self.grid, FarField)
            if self.normalize == 'directivity':
                integr = trapezoid(I, phi)
                pattern = 2*np.pi*I/integr
            elif self.normalize == None:
                pattern = I   
            return DiagramResult(phi, pattern)
        else:
            return NotImplementedError    
        

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
        grid: Grid = None,
        field_type: str = None,
        diagram_normalize: str = 'directivity'
    ):
        self.base_config = base_config
        self.param = sweep_param
        self.values = sweep_values
        self.compute_dipoles = compute_dipoles or compute_force
        self.compute_diagram = compute_diagram
        self.compute_force = compute_force
        self.compute_fields = compute_fields
        self.grid = grid
        self.field_type = field_type
        self.diagram_normalize = diagram_normalize

        if self.compute_fields and self.grid is None:
            raise ValueError("Для вычисления полей необходимо передать `grid`.")

    # def run(self):
    #     summary_results = []
    #     diagrams_records = []
    #     fields_results = {}

    #     for val in tqdm(self.values, desc=f"Sweeping '{self.param}'", unit="step"):
    #         setattr(self.base_config, self.param, val)
    #         row = {self.param: val}

    #         if self.compute_dipoles:
    #             dip_res = DipoleCalculator(self.base_config).compute()
    #             row.update(dip_res.as_dict())
                
    #         if self.compute_diagram:
    #             diag_res = DiagramCalculator(self.base_config).compute(field_type=self.field_type)
    #             df_diag = diag_res.as_dict()
    #             df_diag[self.param] = val
    #             diagrams_records.append(df_diag)

    #         if self.compute_force:
    #             force_result = OpticalForceCalculator(self.base_config).compute()
    #             row.update(force_result.as_dict())

    #         if self.compute_fields:
    #             field_result = FieldsCalculator(self.base_config).compute(self.grid)
    #             fields_results[val] = field_result

    #         summary_results.append(row)

    #     df_summary = pd.DataFrame(summary_results)
    #     df_diagrams = pd.concat(diagrams_records, ignore_index=True) if self.compute_diagram else None

    #     return df_summary, df_diagrams, fields_results if self.compute_fields else None


    def _single_run(self, val):
        if not self.base_config.show_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                return self._do_single_run(val)
        else:
            return self._do_single_run(val)
        
        
    def _do_single_run(self, val):
        """Один шаг sweep-а"""
        setattr(self.base_config, self.param, val)
        row = {self.param: val}
        diagrams_records = []
        fields_results = {}

        if self.compute_dipoles:
            dip_res = DipoleCalculator(self.base_config).compute()
            row.update(dip_res.as_dict())

        if self.compute_diagram:
            diag_res = DiagramCalculator(self.base_config, grid=self.grid, normalize=self.diagram_normalize).compute(field_type=self.field_type, internal_compute=True)
            df_diag = diag_res.as_dict()
            df_diag[self.param] = val
            diagrams_records.append(df_diag)

        if self.compute_force:
            force_result = OpticalForceCalculator(self.base_config).compute()
            row.update(force_result.as_dict())

        if self.compute_fields:
            field_result = FieldsCalculator(self.base_config).compute(self.grid, field_type=self.field_type, internal_compute=True)
            fields_results[val] = field_result

        # logging.info(f"Done {self.param}={val}")
        return row, diagrams_records, fields_results



    def run_par(self, n_jobs=-1):
        with tqdm_joblib(tqdm(total=len(self.values))):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._single_run)(val) for val in self.values
            )

        return results

    def run(self, n_jobs=-1):
        summary_results, diagrams_records, fields_results = [], [], {}
        results = self.run_par(n_jobs=n_jobs)
            
        for row, diag, field in results:
            summary_results.append(row)
            diagrams_records.extend(diag)
            fields_results.update(field)

        df_summary = pd.DataFrame(summary_results)
        df_diagrams = (
            pd.concat(diagrams_records, ignore_index=True)
            if self.compute_diagram else None
        )
        return df_summary, df_diagrams, fields_results if self.compute_fields else None


class SweepRunner2D:
    def __init__(
        self,
        base_config: SimulationConfig,
        primary_param: str,
        primary_values,
        secondary_param: str,
        secondary_values,
        compute_dipoles: bool = True,
        compute_diagram: bool = True,
        compute_force: bool = False,
        compute_fields: bool = False,
        grid: Grid = None,
        field_type: str = None,
        diagram_normalize: str = 'directivity',
        parallel_param: str = 'primary',
        enable_parallel: bool = True,
    ):
        if primary_param == secondary_param:
            raise ValueError("primary_param and secondary_param must be different")

        self.base_config = base_config
        self.primary_param = primary_param
        self.secondary_param = secondary_param
        self.primary_values = list(primary_values)
        self.secondary_values = list(secondary_values)
        self.compute_dipoles = compute_dipoles or compute_force
        self.compute_diagram = compute_diagram
        self.compute_force = compute_force
        self.compute_fields = compute_fields
        self.grid = grid
        self.field_type = field_type
        self.diagram_normalize = diagram_normalize
        self.parallel_param = parallel_param
        self.enable_parallel = enable_parallel

        if self.compute_fields and self.grid is None:
            raise ValueError("Для вычисления полей необходимо передать `grid`.")

        if parallel_param not in {None, 'primary', 'secondary'}:
            raise ValueError("parallel_param must be None, 'primary', or 'secondary'")

    def _run_step(self, assignments, fields_key):
        effective_assignments = dict(assignments)

        for attr, value in effective_assignments.items():
            setattr(self.base_config, attr, value)

        row = dict(effective_assignments)
        diagrams_records = []
        fields_results = {}

        if self.compute_dipoles:
            dip_res = DipoleCalculator(self.base_config).compute()
            row.update(dip_res.as_dict())

        if self.compute_diagram:
            diag_res = DiagramCalculator(self.base_config, normalize=self.diagram_normalize).compute(field_type=self.field_type, internal_compute=True)
            df_diag = diag_res.as_dict()
            for attr, value in effective_assignments.items():
                df_diag[attr] = value
            diagrams_records.append(df_diag)

        if self.compute_force:
            force_result = OpticalForceCalculator(self.base_config).compute()
            row.update(force_result.as_dict())

        if self.compute_fields:
            field_result = FieldsCalculator(self.base_config).compute(self.grid, field_type=self.field_type, internal_compute=True)
            fields_results[fields_key] = field_result

        return row, diagrams_records, fields_results

    def _sequential_run(self, iterable_primary, iterable_secondary):
        summary_results, diagrams_records, fields_results = [], [], {}
        for val_primary in tqdm(iterable_primary, desc=f"{self.primary_param} sweep"):
            for val_secondary in iterable_secondary:
                assignments = {
                    self.primary_param: val_primary,
                    self.secondary_param: val_secondary,
                }
                key = (val_primary, val_secondary)
                if not self.base_config.show_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=IntegrationWarning)
                        row, diag, field = self._run_step(assignments, key)
                else:
                    row, diag, field = self._run_step(assignments, key)
                summary_results.append(row)
                diagrams_records.extend(diag)
                fields_results.update(field)
        return summary_results, diagrams_records, fields_results

    def _parallel_worker(self, fixed_value, sweep_over_secondary):
        summary_results, diagrams_records, fields_results = [], [], {}
        iterable = self.secondary_values if sweep_over_secondary else self.primary_values

        for varying_value in iterable:
            if sweep_over_secondary:
                assignments = {
                    self.primary_param: fixed_value,
                    self.secondary_param: varying_value,
                }
                key = (fixed_value, varying_value)
            else:
                assignments = {
                    self.primary_param: varying_value,
                    self.secondary_param: fixed_value,
                }
                key = (varying_value, fixed_value)

            if not self.base_config.show_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=IntegrationWarning)
                    row, diag, field = self._run_step(assignments, key)
            else:
                row, diag, field = self._run_step(assignments, key)

            summary_results.append(row)
            diagrams_records.extend(diag)
            fields_results.update(field)

        return summary_results, diagrams_records, fields_results

    def run(self, n_jobs=-1):
        if (
            not self.enable_parallel
            or self.parallel_param is None
            or n_jobs == 1
        ):
            summary_results, diagrams_records, fields_results = self._sequential_run(
                self.primary_values,
                self.secondary_values,
            )
        else:
            if self.parallel_param == 'primary':
                tasks = self.primary_values
                sweep_secondary = True
            else:
                tasks = self.secondary_values
                sweep_secondary = False

            with tqdm_joblib(tqdm(total=len(tasks))):
                results = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(self._parallel_worker)(val, sweep_secondary)
                    for val in tasks
                )

            summary_results, diagrams_records, fields_results = [], [], {}
            for summary, diagrams, fields in results:
                summary_results.extend(summary)
                diagrams_records.extend(diagrams)
                fields_results.update(fields)

        df_summary = pd.DataFrame(summary_results)
        df_diagrams = (
            pd.concat(diagrams_records, ignore_index=True)
            if self.compute_diagram else None
        )
        return df_summary, df_diagrams, fields_results if self.compute_fields else None

