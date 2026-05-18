"""
Оптимизация параметров поляризации для сортировки частиц по размеру.

Ищет параметры psi и chi, при которых угол φ = arctan(Fy/Fx) изменяется
плавно, монотонно и с максимальной амплитудой в диапазоне R ∈ [100, 165] нм.

Использует SimulationConfig с initial_field_type='two_beam' для расчета 
ПОЛНОЙ силы (не вычитая компоненты).
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from datetime import datetime
import json
import time

from pint import UnitRegistry
from tqdm import tqdm
import warnings

from MieSppForce.simulation import SimulationConfig, SweepRunner

warnings.filterwarnings('ignore')

ureg = UnitRegistry()

# === Конфигурация ===
R_MIN = 100  # нм
R_MAX = 165  # нм
N_POINTS_SCAN = 50  # точек для сканирования
N_POINTS_FINE = 100  # точек для финальной визуализации

PSI_RANGE = (0, np.pi)
CHI_RANGE = (-np.pi/4, np.pi/4)
N_PSI = 100
N_CHI = 100

OUTPUT_DIR = Path("optimization_results")

# Физические константы
c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


def compute_force_angle_curve(psi, chi, R_values, config_params):
    """
    Вычисляет кривую угла силы phi(R) для заданных psi и chi.
    Использует SimulationConfig с initial_field_type='two_beam' и SweepRunner.
    
    Args:
        psi: угол поляризации psi (рад)
        chi: угол поляризации chi (рад)
        R_values: массив радиусов (pint Quantity в nm)
        config_params: dict с параметрами конфигурации
    
    Returns:
        phi: массив углов
        Fx: массив компонент силы Fx (в N)
        Fy: массив компонент силы Fy (в N)
    """
    wl = config_params['wl']
    dist = config_params['dist']
    angle = config_params['angle']
    
    try:
        # Создаём конфигурацию с two_beam
        config = SimulationConfig(
            wl=wl,
            R=R_values[0],  # начальное значение, будет меняться в sweep
            dist=dist,
            angle=angle,
            psi=psi,
            chi=chi,
            show_warnings=False,
            initial_field_type='plane_wave',
        )
        
        # Запускаем sweep по R с вычислением силы (verbose=False чтобы не спамить)
        sweep_result, _, _ = SweepRunner(
            config, 
            sweep_param='R', 
            sweep_values=R_values,
            compute_dipoles=True,
            compute_diagram=False,
            compute_force=True,
            compute_fields=False,
            verbose=False,  # Отключаем внутренний tqdm
        ).run(n_jobs=1)  # n_jobs=1 для работы кэша
        
        # Извлекаем Fx и Fy из результата (полная сила!)
        Fx_arr = sweep_result.Fx.to('N').magnitude
        Fy_arr = sweep_result.Fy.to('N').magnitude
        
        # arctan2 возвращает углы в [-π, π]
        phi = np.arctan2(Fy_arr, Fx_arr)
        
        return phi, Fx_arr, Fy_arr
        
    except Exception as e:
        print(f"Error at psi={psi:.3f}, chi={chi:.3f}: {e}")
        return None, None, None


def angular_distance(phi1, phi2):
    """
    Вычисляет кратчайшее угловое расстояние между двумя углами.
    Учитывает периодичность: расстояние между -π и π равно 0.
    Возвращает значение в [-π, π].
    """
    diff = phi2 - phi1
    return np.arctan2(np.sin(diff), np.cos(diff))


def total_angular_path(phi: np.ndarray) -> float:
    """
    Вычисляет суммарный угловой путь - сумму всех угловых приращений.
    Это "длина пути" по окружности от первой до последней точки.
    """
    total = 0.0
    for i in range(1, len(phi)):
        delta = angular_distance(phi[i-1], phi[i])
        total += delta
    return total


def unwrap_angular(phi: np.ndarray) -> np.ndarray:
    """
    Разворачивает углы, убирая скачки на ±2π.
    (Для совместимости со старыми функциями)
    """
    return np.unwrap(phi)


def quality_metric(phi: np.ndarray) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════
    МЕТРИКА КАЧЕСТВА ДЛЯ СОРТИРОВКИ ЧАСТИЦ ПО РАЗМЕРУ
    ═══════════════════════════════════════════════════════════════════════
    
    ЦЕЛЬ: Найти параметры поляризации (psi, chi), при которых частицы 
    разных размеров R отклоняются в разные стороны под действием силы.
    
    Угол направления силы: φ = arctan2(Fy, Fx) ∈ [-π, π]
    
    ───────────────────────────────────────────────────────────────────────
    КРИТЕРИИ КАЧЕСТВА:
    ───────────────────────────────────────────────────────────────────────
    
    1. УГЛОВОЙ ПУТЬ (Angular Path) - главный критерий
       ─────────────────────────────────────────────────
       Сумма угловых приращений от R_min до R_max:
       
         Path = Σ Δφᵢ,  где Δφᵢ = φ(Rᵢ₊₁) - φ(Rᵢ) (с учетом периодичности)
       
       ЧЕМ БОЛЬШЕ |Path|, тем лучше:
       - Большой путь = большой разброс углов = хорошее разделение частиц
       - Path > 0: угол растёт с R
       - Path < 0: угол падает с R
       
    2. МОНОТОННОСТЬ - критерий однозначности
       ─────────────────────────────────────────────────
       Все приращения Δφᵢ должны быть одного знака.
       
       Если есть смены знака → одному углу могут соответствовать 
       несколько размеров R → сортировка неоднозначна!
       
       Штраф: -0.3 радиан за каждую смену знака производной
       
    3. ГЛАДКОСТЬ - отсутствие резких скачков
       ─────────────────────────────────────────────────
       Стандартное отклонение второй производной должно быть малым.
       
       Штраф: -2 × std(d²φ/dR²)
    
    ───────────────────────────────────────────────────────────────────────
    ИТОГОВАЯ ФОРМУЛА:
    ───────────────────────────────────────────────────────────────────────
    
      Quality = |Path| - 0.3×(смены знака) - 2×std(d²φ) + Бонус
      
      Бонус = +0.2×|Path| если кривая строго монотонна
    
    ───────────────────────────────────────────────────────────────────────
    ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТА:
    ───────────────────────────────────────────────────────────────────────
    
      Quality > 1.0  - Отличная сортировка (разброс > 60°)
      Quality > 0.5  - Хорошая сортировка (разброс > 30°)  
      Quality > 0.2  - Приемлемая сортировка
      Quality < 0.2  - Плохая сортировка
      
    ═══════════════════════════════════════════════════════════════════════
    """
    if phi is None or len(phi) < 3:
        return -np.inf
    
    # 1. УГЛОВОЙ ПУТЬ - суммарное изменение угла от R_min до R_max
    angular_path = total_angular_path(phi)
    abs_path = np.abs(angular_path)
    
    # 2. МОНОТОННОСТЬ - считаем смены знака производной
    deltas = np.array([angular_distance(phi[i-1], phi[i]) for i in range(1, len(phi))])
    
    # Игнорируем очень маленькие изменения (шум)
    significant_deltas = deltas[np.abs(deltas) > 0.01]
    if len(significant_deltas) < 2:
        sign_changes = 0
    else:
        sign_changes = np.sum(np.abs(np.diff(np.sign(significant_deltas))) > 0)
    
    monotonicity_penalty = sign_changes * 0.3
    
    # 3. ГЛАДКОСТЬ - стандартное отклонение второй производной
    if len(deltas) > 1:
        d2phi = np.diff(deltas)
        smoothness_penalty = np.std(d2phi) * 2
    else:
        smoothness_penalty = 0
    
    # 4. БОНУС за строгую монотонность
    is_monotonic = (sign_changes == 0) and (abs_path > 0.1)
    monotonicity_bonus = 0.2 * abs_path if is_monotonic else 0
    
    # Итоговая метрика
    quality = abs_path - monotonicity_penalty - smoothness_penalty + monotonicity_bonus
    
    return quality


def quality_metric_old(phi: np.ndarray) -> float:
    """Старая версия метрики (для сравнения)."""
    if phi is None or len(phi) < 3:
        return -np.inf
    
    phi_unwrap = unwrap_angular(phi)
    amplitude = np.abs(phi_unwrap[-1] - phi_unwrap[0])
    
    dphi = np.diff(phi_unwrap)
    sign_changes = np.sum(np.abs(np.diff(np.sign(dphi))) > 0)
    monotonicity_penalty = sign_changes * 0.2
    
    d2phi = np.diff(dphi)
    smoothness_penalty = np.std(d2phi) * 2
    
    R_idx = np.arange(len(phi_unwrap))
    corr = np.corrcoef(R_idx, phi_unwrap)[0, 1]
    if np.isnan(corr):
        corr = 0
    linearity_bonus = np.abs(corr) * 0.5
    
    quality = amplitude - monotonicity_penalty - smoothness_penalty + linearity_bonus
    return quality


def scan_parameter_space(R_values, config_params, psi_range, chi_range, n_psi, n_chi):
    """
    Сканирование пространства параметров psi и chi.
    Использует SimulationConfig с two_beam.
    
    Args:
        R_values: массив радиусов (pint Quantity)
    """
    psi_scan = np.linspace(*psi_range, n_psi)
    chi_scan = np.linspace(*chi_range, n_chi)
    param_grid = list(product(psi_scan, chi_scan))
    
    print(f"Сканирование {len(param_grid)} точек...")
    print(f"  psi ∈ [{np.rad2deg(psi_range[0]):.0f}°, {np.rad2deg(psi_range[1]):.0f}°]")
    print(f"  chi ∈ [{np.rad2deg(chi_range[0]):.0f}°, {np.rad2deg(chi_range[1]):.0f}°]")
    
    results = []
    
    for psi, chi in tqdm(param_grid, desc="Scanning parameters"):
        phi, Fx, Fy = compute_force_angle_curve(psi, chi, R_values, config_params)
        q = quality_metric(phi)
        
        result = {
            'psi': float(psi),
            'chi': float(chi),
            'quality': float(q) if not np.isinf(q) else None,
            'phi': phi.tolist() if phi is not None else None,
            'Fx': Fx.tolist() if Fx is not None else None,
            'Fy': Fy.tolist() if Fy is not None else None
        }
        results.append(result)
    
    valid_results = [r for r in results if r['quality'] is not None]
    print(f"\nВалидных точек: {len(valid_results)} / {len(results)}")
    
    return results, psi_scan, chi_scan


def run_optimization():
    """Запуск полной оптимизации с использованием SimulationConfig."""
    
    print("="*70)
    print("ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ПОЛЯРИЗАЦИИ")
    print("="*70)
    
    # Параметры конфигурации с единицами измерения pint
    wl = 900 * ureg.nanometer
    dist = 2 * ureg.nanometer
    angle = np.deg2rad(25)
    
    # Массив радиусов с единицами
    R_values = np.linspace(R_MIN, R_MAX, N_POINTS_SCAN) * ureg.nanometer
    
    config_params = {
        'wl': wl,
        'dist': dist,
        'angle': angle,
    }
    
    print(f"Параметры:")
    print(f"  wl = {wl}")
    print(f"  dist = {dist}")
    print(f"  angle = {np.rad2deg(angle):.1f}°")
    print(f"  R ∈ [{R_MIN}, {R_MAX}] nm, {N_POINTS_SCAN} точек")
    
    # Сканирование пространства параметров
    t_start = time.time()
    results, psi_scan, chi_scan = scan_parameter_space(
        R_values, config_params, PSI_RANGE, CHI_RANGE, N_PSI, N_CHI
    )
    t_scan = time.time() - t_start
    print(f"\nВремя сканирования: {t_scan:.1f} с")
    
    return results, psi_scan, chi_scan


def save_results(results: list, psi_scan: np.ndarray, chi_scan: np.ndarray, output_dir: Path):
    """Сохранение результатов оптимизации."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохранение всех результатов в JSON
    results_file = output_dir / f"optimization_results_{timestamp}_au.json"
    
    save_data = {
        'timestamp': timestamp,
        'config': {
            'R_min_nm': R_MIN,
            'R_max_nm': R_MAX,
            'N_points_scan': N_POINTS_SCAN,
            'psi_range_rad': list(PSI_RANGE),
            'chi_range_rad': list(CHI_RANGE),
            'N_psi': N_PSI,
            'N_chi': N_CHI,
        },
        'psi_scan': psi_scan.tolist(),
        'chi_scan': chi_scan.tolist(),
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nРезультаты сохранены: {results_file}")
    
    # Сохранение лучших параметров отдельно
    valid_results = [r for r in results if r['quality'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['quality'])
        top5 = sorted(valid_results, key=lambda x: x['quality'], reverse=True)[:5]
        
        best_file = output_dir / f"best_params_{timestamp}_au.json"
        best_data = {
            'timestamp': timestamp,
            'best': {
                'psi_rad': best['psi'],
                'psi_deg': np.rad2deg(best['psi']),
                'chi_rad': best['chi'],
                'chi_deg': np.rad2deg(best['chi']),
                'quality': best['quality'],
            },
            'top5': [
                {
                    'psi_rad': r['psi'],
                    'psi_deg': np.rad2deg(r['psi']),
                    'chi_rad': r['chi'],
                    'chi_deg': np.rad2deg(r['chi']),
                    'quality': r['quality'],
                }
                for r in top5
            ]
        }
        
        with open(best_file, 'w') as f:
            json.dump(best_data, f, indent=2)
        
        print(f"Лучшие параметры: {best_file}")
    
    return results_file


def plot_results(results: list, psi_scan: np.ndarray, chi_scan: np.ndarray, 
                 output_dir: Path, show: bool = True):
    """Визуализация результатов оптимизации."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    valid_results = [r for r in results if r['quality'] is not None]
    if not valid_results:
        print("Нет валидных результатов для визуализации")
        return
    
    # === Карта качества ===
    quality_map = np.full((len(psi_scan), len(chi_scan)), np.nan)
    for r in valid_results:
        i = np.argmin(np.abs(psi_scan - r['psi']))
        j = np.argmin(np.abs(chi_scan - r['chi']))
        quality_map[i, j] = r['quality']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    
    # Карта качества
    ax = axes[0]
    im = ax.imshow(
        quality_map,
        origin='lower',
        aspect='auto',
        extent=[
            np.rad2deg(chi_scan.min()), np.rad2deg(chi_scan.max()),
            np.rad2deg(psi_scan.min()), np.rad2deg(psi_scan.max())
        ],
        cmap='viridis'
    )
    plt.colorbar(im, ax=ax, label='Quality metric')
    ax.set_xlabel(r'$\chi$ (°)')
    ax.set_ylabel(r'$\psi$ (°)')
    ax.set_title('Карта качества параметров поляризации')
    
    # Лучшая точка
    best = max(valid_results, key=lambda x: x['quality'])
    ax.scatter(np.rad2deg(best['chi']), np.rad2deg(best['psi']), 
               c='red', s=150, marker='*', edgecolors='white', linewidths=1,
               label=f"Best: ψ={np.rad2deg(best['psi']):.1f}°, χ={np.rad2deg(best['chi']):.1f}°")
    ax.legend(loc='upper right')
    
    # === Угол φ(R) для лучших параметров ===
    ax = axes[1]
    R_plot = np.linspace(R_MIN, R_MAX, N_POINTS_SCAN)
    
    # Топ-5 кривых
    top5 = sorted(valid_results, key=lambda x: x['quality'], reverse=True)[:5]
    colors = plt.cm.tab10(np.linspace(0, 0.5, 5))
    
    for idx, (r, color) in enumerate(zip(top5, colors)):
        if r['phi'] is not None:
            phi_arr = np.array(r['phi'])
            # Используем unwrap для непрерывности
            phi_unwrap = np.unwrap(phi_arr)
            # Сдвигаем так, чтобы среднее было близко к 0 (в пределах [-π, π])
            mean_phi = np.mean(phi_unwrap)
            shift = np.round(mean_phi / (2*np.pi)) * 2*np.pi
            phi_plot = phi_unwrap - shift
            
            # Вычисляем угловой путь для подписи
            path = total_angular_path(phi_arr)
            label = f"#{idx+1}: ψ={np.rad2deg(r['psi']):.0f}°, χ={np.rad2deg(r['chi']):.0f}° (Path={path:.2f})"
            lw = 3 if idx == 0 else 1.5
            ax.plot(R_plot, phi_plot, color=color, lw=lw, label=label)
    
    ax.set_xlabel('$R$ (nm)')
    ax.set_ylabel(r'$\varphi = \arctan_2(F_y, F_x)$ (rad)')
    ax.set_title('Угол направления силы от радиуса частицы')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(R_MIN, R_MAX)
    
    # Динамический диапазон по Y, но с подписями в π
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(np.pi/2, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(-np.pi/2, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(np.pi, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(-np.pi, color='gray', linestyle=':', alpha=0.3)
    
    # Подписи на оси Y в терминах π
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    plt.tight_layout()
    
    fig_file = output_dir / f"optimization_quality_map_{timestamp}.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"\nКарта качества: {fig_file}")
    
    if show:
        plt.show()
    plt.close()


def print_summary(results: list):
    """Вывод краткой сводки результатов."""
    valid_results = [r for r in results if r['quality'] is not None]
    
    if not valid_results:
        print("Нет валидных результатов")
        return
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ПАРАМЕТРОВ ПОЛЯРИЗАЦИИ")
    print("="*70)
    
    best = max(valid_results, key=lambda x: x['quality'])
    print(f"\nЛучший набор параметров:")
    print(f"  ψ = {best['psi']:.4f} rad ({np.rad2deg(best['psi']):.1f}°)")
    print(f"  χ = {best['chi']:.4f} rad ({np.rad2deg(best['chi']):.1f}°)")
    print(f"  Quality = {best['quality']:.4f}")
    
    if best['phi'] is not None:
        phi_arr = np.array(best['phi'])
        # Угловой путь - суммарное изменение угла
        path = total_angular_path(phi_arr)
        path_deg = np.rad2deg(path)
        print(f"  Угловой путь: {path:.3f} rad ({path_deg:.1f}°)")
        print(f"  Диапазон углов: [{phi_arr.min():.2f}, {phi_arr.max():.2f}] rad")
        
        # Проверка монотонности
        deltas = np.array([angular_distance(phi_arr[i-1], phi_arr[i]) for i in range(1, len(phi_arr))])
        significant = deltas[np.abs(deltas) > 0.01]
        if len(significant) > 0:
            is_monotonic = np.all(significant >= 0) or np.all(significant <= 0)
        else:
            is_monotonic = True
        print(f"  Монотонность: {'✓ Да' if is_monotonic else '✗ Нет'}")
    
    print("\nИнтерпретация Quality:")
    print("  > 1.0  - Отличная сортировка (угловой путь > 60°)")
    print("  > 0.5  - Хорошая сортировка (угловой путь > 30°)")
    print("  > 0.2  - Приемлемая сортировка")
    print("  < 0.2  - Плохая сортировка")
    
    print("\nТоп-5 наборов параметров:")
    print("-"*70)
    top5 = sorted(valid_results, key=lambda x: x['quality'], reverse=True)[:5]
    
    for i, r in enumerate(top5, 1):
        if r['phi'] is not None:
            phi_arr = np.array(r['phi'])
            path = total_angular_path(phi_arr)
            path_deg = np.rad2deg(path)
            deltas = np.array([angular_distance(phi_arr[j-1], phi_arr[j]) for j in range(1, len(phi_arr))])
            significant = deltas[np.abs(deltas) > 0.01]
            if len(significant) > 0:
                is_mono = "✓" if (np.all(significant >= 0) or np.all(significant <= 0)) else "✗"
            else:
                is_mono = "✓"
        else:
            path_deg = 0
            is_mono = "?"
        print(f"  {i}. ψ={np.rad2deg(r['psi']):5.1f}°, χ={np.rad2deg(r['chi']):6.1f}° | "
              f"Q={r['quality']:.3f} | Path={path_deg:+6.1f}° | Mono:{is_mono}")
    
    print("="*70)


def main():
    """Главная функция."""
    # Запуск оптимизации
    results, psi_scan, chi_scan = run_optimization()
    
    # Сохранение результатов
    save_results(results, psi_scan, chi_scan, OUTPUT_DIR)
    
    # Визуализация
    plot_results(results, psi_scan, chi_scan, OUTPUT_DIR, show=True)
    
    # Вывод сводки
    print_summary(results)


if __name__ == "__main__":
    main()
