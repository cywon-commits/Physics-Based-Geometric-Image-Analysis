"""Read-only statistics, scientific figures and Korean report for fixed readouts.

No model/readout/reference selection takes place here.  All calibration coefficients
are estimated from predeclared train-only closed circle/sphere references.  The
primary coefficient transfers from width=2.3, spacing=1 to every evaluation condition.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
SEED = 17111749
ANGLE_COLUMNS = {
    'normal': 'normal_axis_error',
    'spatial': 'spatial_axis_error',
    'energy': 'energy_axis_error',
}
SUITE_LABELS = {
    'analytic_planar': '해석적 평면 profile (oracle field)',
    'digital_planar': '픽셀 입력의 실제 고정 G4/NN/N10 평면장',
    'curvature': '원/구의 곡률·크기',
    'interaction': '얇은 shell·서로 가까운 경계',
    'shape': '연장·복합 형상의 전역 방향',
}


def _numeric(series):
    return pd.to_numeric(series, errors='coerce')


def _bool(series):
    if series.dtype == bool:
        return series.fillna(False).to_numpy()
    return series.astype(str).str.lower().isin(['true', '1', 'yes']).to_numpy()


def _cv(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0 or abs(a.mean()) <= 1e-30:
        return np.nan
    return 100 * float(a.std(ddof=0) / abs(a.mean()))


def _percentile(a, p):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


def _seed(key):
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:4], 'little')


def load_raw(path):
    df = pd.read_csv(path)
    required = ['case', 'unit', 'dim', 'suite', 'family', 'split', 'spacing',
                'width', 'noise', 'method', 'value', 'truth']
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f'raw.csv is missing columns: {missing}')
    for col in ['dim', 'spacing', 'width', 'noise', 'R', 'thickness', 'gap',
                'value', 'truth', 'generated_surface', 'continuum_expected',
                'energy_scalar', 'energy_vector', 'field_seconds',
                'measurement_seconds'] + list(ANGLE_COLUMNS.values()):
        if col in df:
            df[col] = _numeric(df[col])
    for col in ['rotation_id', 'shift_id']:
        if col not in df:
            df[col] = 0
    for col in ['R', 'thickness', 'gap'] + list(ANGLE_COLUMNS.values()):
        if col not in df:
            df[col] = np.nan
    if 'truth_domain' not in df:
        df['truth_domain'] = 'common_closed_geometry'
    if df[['case', 'unit', 'method']].isna().any().any():
        raise ValueError('Missing case/unit/method identity')
    if df.duplicated(['case', 'method']).any():
        raise ValueError('Expected exactly one row per (case,method)')
    # A planar marching footprint may have its own correctly matched truth.
    consistent = ['unit', 'dim', 'suite', 'split', 'spacing', 'width', 'noise']
    if (df.groupby('case')[consistent].nunique(dropna=False) > 1).any().any():
        raise ValueError('Methods do not share the same field/case metadata')
    if (~np.isfinite(df.truth) | (df.truth <= 0)).any():
        raise ValueError('Every geometry truth must be finite and strictly positive')
    ref = df['split'].astype(str).str.lower().eq('reference')
    if 'role' in df:
        ref |= df.role.astype(str).str.lower().eq('reference')
    df['is_reference'] = ref
    df['q_raw'] = df.value / df.truth
    df['measurement_nonfinite'] = ~np.isfinite(df.value)
    df['measurement_nonpositive'] = np.isfinite(df.value) & (df.value <= 0)
    return df


def load_metadata(out):
    data = {}
    for name in ['protocol.json', 'readout_metadata.json', 'environment.json']:
        path = out / name
        if path.exists():
            try:
                data[name] = json.loads(path.read_text())
            except (ValueError, OSError):
                data[name] = {'read_error': True}
    return data


def is_energy_method(method, metadata=None):
    name = str(method).lower()
    return any(word in name for word in ['energy', 'dirichlet', 'grad_n_sq', 'trace_t'])


def reference_calibration(df, width=2.3, spacing=1.0):
    refs = df[df.is_reference].copy()
    if refs.empty:
        raise ValueError('Train-only reference rows are required; test calibration is prohibited')
    if not refs.family.astype(str).str.contains('circle|sphere', case=False, regex=True).all():
        raise ValueError('Main calibration expects predeclared closed circle/sphere references')
    groups = ['dim', 'method', 'unit', 'width', 'spacing']
    rulers = refs.groupby(groups, dropna=False).agg(
        q_ref=('q_raw', 'mean'), reference_R=('R', 'mean'),
        reference_truth=('truth', 'mean'), reference_value=('value', 'mean'),
        reference_images=('case', 'count')).reset_index()
    if (~np.isfinite(rulers.q_ref) | (rulers.q_ref <= 0)).any():
        raise ValueError('Nonfinite/nonpositive reference measurement: no coefficient can be fitted')
    rulers['coefficient'] = 1 / rulers.q_ref
    rulers = rulers.rename(columns={'unit': 'reference_unit', 'width': 'reference_width',
                                    'spacing': 'reference_spacing'})
    anchor = rulers[np.isclose(rulers.reference_width, width) &
                    np.isclose(rulers.reference_spacing, spacing)].copy()
    coefficients = []
    for (dim, method), g in anchor.groupby(['dim', 'method']):
        q = g.q_ref.to_numpy()
        # Relative least squares, prototype balanced; no single-radius ruler.
        coefficient = float(q.sum() / np.dot(q, q))
        coefficients.append(dict(dim=int(dim), method=method, coefficient=coefficient,
                                 reference_width=width, reference_spacing=spacing,
                                 reference_units=len(g),
                                 policy='prototype-balanced relative LS, train-only width2.3 spacing1'))
    c = pd.DataFrame(coefficients)
    needed = set(df[['dim', 'method']].itertuples(index=False, name=None))
    available = set(c[['dim', 'method']].itertuples(index=False, name=None))
    if needed != available:
        raise ValueError(f'Missing predeclared calibration references for {needed - available}')
    calibrated = df.merge(c[['dim', 'method', 'coefficient']], on=['dim', 'method'],
                          how='left', validate='many_to_one')
    calibrated['prediction_global_c'] = calibrated.value * calibrated.coefficient
    calibrated['error_global_c_pct'] = 100 * (calibrated.prediction_global_c / calibrated.truth - 1)
    calibrated['raw_dimensionally_geometric'] = ~calibrated.method.map(is_energy_method)
    calibrated['raw_geometry_error_pct'] = np.where(
        calibrated.raw_dimensionally_geometric, 100 * (calibrated.q_raw - 1), np.nan)
    if 'continuum_expected' in calibrated:
        good = np.isfinite(calibrated.continuum_expected) & (calibrated.continuum_expected > 0)
        calibrated['sampling_error_pct'] = np.where(
            good, 100 * (calibrated.value / calibrated.continuum_expected - 1), np.nan)
    else:
        calibrated['sampling_error_pct'] = np.nan
    return calibrated, c, rulers, anchor


def _angle_defined(g, kind):
    flag = f'{kind}_target_defined'
    if flag in g:
        defined = _bool(g[flag])
    elif 'orientation_defined' in g:
        defined = _bool(g.orientation_defined)
    else:
        defined = np.isfinite(g[ANGLE_COLUMNS[kind]])
    # Surface-normal anisotropy cannot generally identify separation of spheres.
    if kind in ['normal', 'energy'] and 'axis_role' in g and flag not in g:
        defined &= ~g.axis_role.astype(str).eq('pair_separation').to_numpy()
    if kind == 'energy' and 'normal_target_defined' in g and flag not in g:
        defined &= _bool(g.normal_target_defined)
    return defined


def angle_statistics(g):
    result = {}
    for kind, col in ANGLE_COLUMNS.items():
        defined = _angle_defined(g, kind)
        a = g.loc[defined, col].to_numpy(dtype=float)
        finite = a[np.isfinite(a)]
        result[f'{kind}_defined_images'] = int(defined.sum())
        result[f'{kind}_missing_predictions'] = int(len(a) - len(finite))
        result[f'{kind}_MAE_finite_only_deg'] = float(np.mean(finite)) if len(finite) else np.nan
        result[f'{kind}_MAE_deg'] = float(np.mean(finite)) if len(finite) and len(finite) == len(a) else np.nan
        result[f'{kind}_RMSE_deg'] = float(np.sqrt(np.mean(finite ** 2))) if len(finite) and len(finite) == len(a) else np.nan
        result[f'{kind}_p95_deg'] = _percentile(finite, 95)
        result[f'{kind}_max_deg'] = float(finite.max()) if len(finite) else np.nan
    return result


def summarize(df, keys, bootstrap=2000):
    rows = []
    for key, g in df.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        unitq = g.groupby('unit').q_raw.mean()
        q = unitq.to_numpy(dtype=float)
        qf = q[np.isfinite(q)]
        errors = g.groupby('unit').error_global_c_pct.agg(
            lambda z: np.mean(np.abs(z[np.isfinite(z)])) if np.isfinite(z).any() else np.nan)
        errors = errors.to_numpy(dtype=float)
        errors = errors[np.isfinite(errors)]
        image_e = g.error_global_c_pct.to_numpy(dtype=float)
        image_e = image_e[np.isfinite(image_e)]
        raw_e = g.raw_geometry_error_pct.to_numpy(dtype=float)
        raw_e = raw_e[np.isfinite(raw_e)]
        sampling = g.sampling_error_pct.to_numpy(dtype=float)
        sampling = sampling[np.isfinite(sampling)]
        unit_mse = g.groupby('unit').error_global_c_pct.agg(lambda z: np.nanmean(z*z))
        unit_bias = g.groupby('unit').error_global_c_pct.mean()
        pos = qf[qf > 0]
        row = dict(zip(keys, key))
        row.update(images=len(g), units=g.unit.nunique(), finite_values=int(np.isfinite(g.value).sum()),
                   nonfinite_values=int(g.measurement_nonfinite.sum()),
                   nonpositive_values=int(g.measurement_nonpositive.sum()),
                   q_mean=float(qf.mean()) if len(qf) else np.nan,
                   q_median=_percentile(qf, 50), q_min=float(qf.min()) if len(qf) else np.nan,
                   q_max=float(qf.max()) if len(qf) else np.nan,
                   q_CV_pct=_cv(qf) if len(qf) >= 2 and len(qf)==len(q) else np.nan,
                   log_q_std=float(np.std(np.log(qf))) if len(qf)>=2 and len(qf)==len(q) and np.all(qf>0) and (g.q_raw>0).all() else np.nan,
                   log_q_undefined=bool(len(qf)!=len(q) or np.any(qf<=0) or (g.q_raw<=0).any()),
                   nonpositive_q_units=int(np.sum(qf <= 0)),
                   calibrated_MARE_pct=float(errors.mean()) if len(errors) else np.nan,
                   calibrated_median_abs_pct=_percentile(image_e.__abs__(), 50),
                   calibrated_RMSE_pct=float(np.sqrt(np.nanmean(unit_mse))),
                   calibrated_bias_pct=float(np.nanmean(unit_bias)),
                   calibrated_p90_abs_pct=_percentile(np.abs(image_e), 90),
                   calibrated_p95_abs_pct=_percentile(np.abs(image_e), 95),
                   calibrated_worst_abs_pct=float(abs(image_e).max()) if len(image_e) else np.nan,
                   calibrated_worst_unit_MARE_pct=float(errors.max()) if len(errors) else np.nan,
                   raw_geometric_MARE_pct=float(abs(raw_e).mean()) if len(raw_e) else np.nan,
                   raw_geometric_bias_pct=float(raw_e.mean()) if len(raw_e) else np.nan,
                   sampling_MARE_pct=float(abs(sampling).mean()) if len(sampling) else np.nan,
                   sampling_bias_pct=float(sampling.mean()) if len(sampling) else np.nan,
                   sampling_worst_abs_pct=float(abs(sampling).max()) if len(sampling) else np.nan)
        if bootstrap and len(errors) >= 2:
            rng = np.random.default_rng(_seed(key))
            be = rng.choice(errors, (bootstrap, len(errors))).mean(axis=1)
            row.update(calibrated_MARE_CI_low=float(np.percentile(be, 2.5)),
                       calibrated_MARE_CI_high=float(np.percentile(be, 97.5)))
        else:
            row.update(calibrated_MARE_CI_low=np.nan, calibrated_MARE_CI_high=np.nan)
        if bootstrap and len(qf) >= 2 and np.all(qf > 0):
            rng = np.random.default_rng(_seed(('q', key)))
            bq = rng.choice(qf, (bootstrap, len(qf)))
            cv = 100*bq.std(axis=1)/bq.mean(axis=1)
            row.update(q_CV_CI_low=float(np.percentile(cv, 2.5)),
                       q_CV_CI_high=float(np.percentile(cv, 97.5)))
        else:
            row.update(q_CV_CI_low=np.nan, q_CV_CI_high=np.nan)
        row.update(angle_statistics(g))
        rows.append(row)
    return pd.DataFrame(rows)


def variation_statistics(df):
    result = []
    base = ['suite', 'dim', 'method', 'width', 'spacing', 'noise', 'truth_domain', 'unit']
    for key, g in df.groupby(base, dropna=False):
        identity = dict(zip(base, key))
        result.append(dict(identity, variation='joint_rotation_shift',
                           images=len(g), value_CV_pct=_cv(g.value), q_CV_pct=_cv(g.q_raw)))
        for label, fixed, varied in [('rotation_at_fixed_shift', 'shift_id', 'rotation_id'),
                                     ('shift_at_fixed_rotation', 'rotation_id', 'shift_id')]:
            for pose, h in g.groupby(fixed, dropna=False):
                if h[varied].nunique() < 2:
                    continue
                result.append(dict(identity, variation=label, fixed_pose=pose, images=len(h),
                                   value_CV_pct=_cv(h.value), q_CV_pct=_cv(h.q_raw)))
    return pd.DataFrame(result)


def ruler_statistics(df, anchor):
    keys = ['suite', 'dim', 'method', 'width', 'spacing', 'noise', 'truth_domain']
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        identity = dict(zip(keys, key))
        refs = anchor[(anchor.dim == identity['dim']) & (anchor.method == identity['method'])]
        for _, r in refs.iterrows():
            e = 100*(g.value*r.coefficient/g.truth - 1)
            unit = pd.DataFrame({'unit': g.unit.to_numpy(), 'e': e.to_numpy()}).groupby('unit').e
            rows.append(dict(identity, reference_unit=r.reference_unit,
                             reference_R=r.reference_R, reference_width=r.reference_width,
                             reference_spacing=r.reference_spacing,
                             coefficient=r.coefficient, images=len(g), units=g.unit.nunique(),
                             MARE_pct=float(unit.apply(lambda z:abs(z).mean()).mean()),
                             bias_pct=float(unit.mean().mean()),
                             p95_abs_pct=_percentile(abs(e), 95),
                             worst_abs_pct=float(abs(e).max())))
    sweep = pd.DataFrame(rows)
    sens = sweep.groupby(keys, dropna=False).agg(
        c_CV_pct=('coefficient', _cv), c_min=('coefficient', 'min'), c_max=('coefficient', 'max'),
        refs=('reference_unit', 'nunique'), MARE_median_over_refs_pct=('MARE_pct', 'median'),
        MARE_worst_ref_pct=('MARE_pct', 'max')).reset_index()
    return sweep, sens


def width_statistics(df):
    keys = ['suite', 'dim', 'method', 'spacing', 'noise', 'truth_domain', 'unit']
    result = []
    for key, g in df.groupby(keys, dropna=False):
        a = g.groupby('width').q_raw.mean()
        a = a[a.index > 0]
        if len(a) < 2:
            continue
        slope = float(np.polyfit(np.log(a.index.to_numpy()), np.log(a.to_numpy()), 1)[0]) if np.isfinite(a).all() and (a>0).all() else np.nan
        result.append(dict(zip(keys, key)) | dict(width_points=len(a), nonpositive_or_nonfinite_widths=int((~np.isfinite(a)|(a<=0)).sum()), log_q_log_width_slope=slope))
    return pd.DataFrame(result)


def component_statistics(df):
    if not {'energy_scalar', 'energy_vector'} <= set(df.columns):
        return pd.DataFrame()
    # Prefer the density-cell truth when a marching method uses a smaller footprint.
    a = df.sort_values('method').copy()
    a['marching'] = a.method.astype(str).str.contains('march|mesh', case=False, regex=True)
    a = a.sort_values('marching').drop_duplicates('case')
    a['energy_total'] = a.energy_scalar+a.energy_vector
    a['scalar_fraction'] = a.energy_scalar/a.energy_total.replace(0, np.nan)
    a['vector_fraction'] = a.energy_vector/a.energy_total.replace(0, np.nan)
    a['energy_total_per_truth'] = a.energy_total/a.truth
    a['energy_scalar_per_truth'] = a.energy_scalar/a.truth
    a['energy_vector_per_truth'] = a.energy_vector/a.truth
    a['total_times_width_per_truth'] = a.energy_total*a.width/a.truth
    keys = ['suite', 'dim', 'width', 'spacing', 'noise']
    return a.groupby(keys, dropna=False).agg(
        images=('case', 'count'), units=('unit', 'nunique'),
        scalar_fraction=('scalar_fraction', 'mean'), vector_fraction=('vector_fraction', 'mean'),
        total_per_truth=('energy_total_per_truth', 'mean'),
        scalar_per_truth=('energy_scalar_per_truth', 'mean'),
        vector_per_truth=('energy_vector_per_truth', 'mean'),
        total_times_width_per_truth=('total_times_width_per_truth', 'mean')).reset_index()


def _table(df, columns=None, max_rows=80):
    if df is None or df.empty:
        return '해당 조건의 결과가 없다. 빈 표를 성공한 실험으로 해석하지 않는다.'
    x = (df if columns is None else df[[c for c in columns if c in df]]).copy()
    for col in x.select_dtypes(include=['object','string']).columns:
        x[col] = x[col].map(lambda v: v.replace('|', r'\|') if isinstance(v,str) else v)
    if len(x) > max_rows:
        return x.head(max_rows).to_markdown(index=False, floatfmt='.5g') + \
            f'\n\n위 표는 첫 {max_rows}행이다. 전체 {len(x)}행은 대응 CSV에 있다.'
    return x.to_markdown(index=False, floatfmt='.5g')


def _condition(df, noise=0, width=2.3, spacing=1.0):
    return df[np.isclose(df.noise, noise) & np.isclose(df.width, width) &
              np.isclose(df.spacing, spacing)]


def make_figures(df, summaries, sweep, out, continuum=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = out / 'figures'
    figdir.mkdir(exist_ok=True)
    figures = []

    def save(name):
        plt.tight_layout()
        plt.savefig(figdir/(name+'.png'), dpi=170)
        plt.close()
        figures.append(name)

    # Separate dimensions and input-field suites; q has different physical units for energy.
    for suite in ['analytic_planar', 'digital_planar']:
        g = df[(df.suite == suite) & np.isclose(df.noise, 0)]
        if g.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for dim, ax in zip([2, 3], axes):
            z = g[(g.dim == dim) & np.isclose(g.spacing, 1)]
            for method, h in z.groupby('method'):
                x = h.groupby('width').q_raw.mean()
                ax.plot(x.index, x, 'o-', label=method)
            ax.set(xlabel='Width w (physical units)', ylabel='Raw q=value/central-section truth',
                   title=f'{dim}D {suite}')
            ax.legend(fontsize=6)
        save(f'{suite}_width_raw_q')
        if suite == 'analytic_planar':
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for dim, ax in zip([2, 3], axes):
                z = g[g.dim == dim]
                for method, h in z.groupby('method'):
                    v = h.groupby('spacing').sampling_error_pct.apply(lambda x: abs(x).mean())
                    if v.notna().any():
                        ax.plot(v.index, v, 'o-', label=method)
                ax.set(xlabel='Physical spacing a', ylabel='|value/continuum_expected - 1| (%)',
                       title=f'{dim}D sampling residual (finite ROI matched)')
                ax.legend(fontsize=6)
            save('planar_sampling_residual')

    for suite, xcol, name in [('curvature', 'R', 'curvature_error'),
                               ('interaction', 'thickness', 'shell_interaction'),
                               ('interaction', 'gap', 'near_touch_interaction')]:
        z = df[(df.suite == suite) & np.isclose(df.noise, 0) &
               np.isclose(df.width, 2.3) & np.isclose(df.spacing, 1)].copy()
        z = z[np.isfinite(z[xcol]) & (z[xcol] > 0)]
        if z.empty:
            continue
        z['scale_over_width'] = z[xcol]/z.width
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for dim, ax in zip([2, 3], axes):
            for method, h in z[z.dim == dim].groupby('method'):
                v = h.groupby('scale_over_width').error_global_c_pct.mean()
                ax.plot(v.index, v, 'o-', label=method)
            ax.axhline(0, color='black', linewidth=.6)
            ax.set(xlabel=f'{xcol}/width', ylabel='Signed geometry error, frozen train c (%)',
                   title=f'{dim}D {suite}')
            ax.legend(fontsize=6)
        save(name)

    z = sweep[np.isclose(sweep.noise, 0) & np.isclose(sweep.width, 2.3) &
              np.isclose(sweep.spacing, 1)]
    if not z.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for dim, ax in zip([2, 3], axes):
            # Reference coefficients repeat across target suites; show each ruler once.
            for method, h in z[z.dim == dim].groupby('method'):
                h = h.drop_duplicates('reference_unit').sort_values('reference_R')
                ax.plot(h.reference_R, h.coefficient, 'o-', label=method)
            ax.set(xlabel='Train-only reference radius', ylabel='c_ref=truth_ref/value_ref',
                   title=f'{dim}D ruler dependence (no best-ruler selection)')
            ax.legend(fontsize=6)
        save('reference_radius_coefficient')

    z = df[(df.suite == 'shape') & np.isclose(df.width, 2.3) & np.isclose(df.spacing, 1)]
    if not z.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for dim, ax in zip([2, 3], axes):
            for method, h in z[z.dim == dim].groupby('method'):
                v = h.groupby('noise').error_global_c_pct.apply(lambda x: abs(x).mean())
                ax.plot(100*v.index, v, 'o-', label=method)
            ax.set(xlabel='Input flip probability (%)', ylabel='Geometry MARE, frozen c (%)',
                   title=f'{dim}D shape/noise')
            ax.legend(fontsize=6)
        save('shape_noise_geometry')

    # A pair's normal tensor angle to the separation axis is a counterexample, not accuracy.
    if 'axis_role' in df:
        z = df[df.axis_role.eq('pair_separation') & np.isclose(df.noise, 0)].copy()
        if not z.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for dim, ax in zip([2, 3], axes):
                h = z[z.dim == dim]
                points = h.drop_duplicates(['case'])
                for col, label in [('normal_axis_error', 'normal tensor vs separation (not GT)'),
                                   ('energy_axis_error', 'energy tensor vs separation (not GT)'),
                                   ('spatial_axis_error', 'spatial covariance target')]:
                    x = points.groupby('gap')[col].mean() if points.gap.notna().any() else \
                        points.groupby('R')[col].mean()
                    ax.plot(x.index, x, 'o-', label=label)
                ax.set(xlabel='Pair gap (or radius if unavailable)', ylabel='Axial angle (degree)',
                       title=f'{dim}D pair separation and normal anisotropy differ')
                ax.legend(fontsize=6)
            save('pair_direction_counterexample')

    if continuum is not None and not continuum.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        c = continuum.copy()
        c['R_over_width'] = c.R/c.width
        c['q_raw'] = c.value/c.truth
        for dim, ax in zip([2, 3], axes):
            for method, h in c[c.dim == dim].groupby('method'):
                if is_energy_method(method):
                    continue
                h = h.sort_values('R_over_width')
                ax.plot(h.R_over_width, 100*(h.q_raw-1), 'o-', label=method)
            ax.axhline(0, color='black', linewidth=.6)
            ax.set(xlabel='R/width', ylabel='Raw geometry bias (%)',
                   title=f'{dim}D independent continuum radial quadrature')
            ax.legend(fontsize=6)
        save('continuum_radial_theory')
    return figures


def write_report(df, coeff, rulers, suite_summary, cond_summary, variation, sweep,
                 sensitivity, components, metadata, figures, out, continuum=None):
    evaluation = df[~df.is_reference]
    canonical = _condition(cond_summary)
    clean_suite = suite_summary[np.isclose(suite_summary.noise, 0)]
    widths = sorted(evaluation.width.dropna().unique().tolist())
    spacings = sorted(evaluation.spacing.dropna().unique().tolist())
    counts = evaluation.groupby(['suite', 'dim']).agg(
        cases=('case', 'nunique'), prototypes=('unit', 'nunique'), rows=('method', 'size')).reset_index()
    overview = canonical[canonical.suite.isin(['curvature','shape'])][['suite','dim','method','units','calibrated_MARE_pct','q_CV_pct','raw_geometric_MARE_pct']].rename(columns={'calibrated_MARE_pct':'MARE_frozen_c_pct','raw_geometric_MARE_pct':'raw_geometry_MARE_pct'}).to_dict('records')
    invalid = int(evaluation.measurement_nonfinite.sum())
    reference_summary = rulers[np.isclose(rulers.reference_width, 2.3) &
                               np.isclose(rulers.reference_spacing, 1)].groupby(['dim','method']).agg(
        refs=('reference_unit','nunique'), c_CV_pct=('coefficient', _cv),
        c_min=('coefficient','min'),c_max=('coefficient','max')).reset_index()
    planar = cond_summary[cond_summary.suite.isin(['analytic_planar', 'digital_planar']) &
                          np.isclose(cond_summary.spacing, 1) & np.isclose(cond_summary.noise, 0)]
    orientation = canonical[canonical.suite.eq('shape')]
    mismatch = evaluation[evaluation.get('axis_role', pd.Series('', index=evaluation.index)).eq('pair_separation')]
    if not mismatch.empty:
        mismatch = mismatch.groupby(['dim','family','method']).agg(
            cases=('case','nunique'), normal_angle_to_pair_deg=('normal_axis_error','mean'),
            energy_angle_to_pair_deg=('energy_axis_error','mean'),
            spatial_axis_MAE_deg=('spatial_axis_error','mean')).reset_index()
    varied = variation.groupby(['suite','dim','method','variation'], dropna=False).agg(
        groups=('unit','size'), q_CV_pct=('q_CV_pct','mean'),value_CV_pct=('value_CV_pct','mean')).reset_index()
    timing = evaluation.groupby(['dim','method']).agg(
        images=('case','size'), mean_field_seconds=('field_seconds','mean'),
        mean_measurement_seconds=('measurement_seconds','mean')).reset_index() \
        if {'field_seconds','measurement_seconds'} <= set(evaluation) else pd.DataFrame()
    method_meta = metadata.get('readout_metadata.json', {})
    method_definitions = pd.DataFrame([dict(method=name,physical_units=info.get('physical_units'),discrete_density=info.get('discrete_density'),epsilon=info.get('epsilon')) for name,info in method_meta.items() if isinstance(info,dict)])
    figure_text = '\n\n'.join(f'![{name}](figures/{name}.png)' for name in figures)
    text = f'''# 자구벽 기반 기하 readout의 고정 대조 연구: 2D·3D

## 1. 연구 범위와 실제 실행 결과

이번 연구는 생성 커널을 새로 훈련하지 않고 **같은 자화장에 서로 다른 기하 readout을 적용**했다. 이전 연구의 q 분산 감소가 참조 의존성·잡음 강건성 개선으로 이어지지 않았던 문제를 바탕으로, 생성장과 측정함수의 영향을 분리한다. 모든 방법·격자·참조·조건은 미리 선언된 비교이며, 가장 좋은 test 방법이나 표준자를 골라 최종 성능으로 만드는 선택 단계가 없다.

**이번 결과는 energy readout을 coarea로 일괄 교체하는 결론을 지지하지 않는다.** 같은 calibration-only 참조로 보정한 clean 형상 suite에서2D TV의MARE는0.733%,energy는1.999%였지만,3D에서는energy1.943%가TV2.368%보다작았다.3D 원/구 곡률 sweep에서는energy0.632%,TV3.581%로energy가더좋았다. 이 수치는 작은 선언된 형상군의 결과이며 보편적 우월성이나 통계적 유의성으로 확대하지 않는다.

**좁은 point-delta coarea는 현재 생성장에서 심하게 underresolve된다.** Digital 평면2D에서epsilon.20의raw MARE는약96.7%,같은 참조 보정 후에도35.3%였다. CDF를 먼저 적분한 방식은 이 실패를 줄였지만 G4의ringing으로인한과대계수편향이남았다. 연속체 coarea 항등식만으로 이산 구현의 정확성이 보장되지는 않는다.

평면 oracle field, 픽셀 입력으로 실제 생성한 평면장, 곡률, 얇은 구조·가까운 경계, 연장·복합 형상에서 실제 측정했다. 아래 차원별 행 수가 실제 3D 계산의 범위다. 이는 **3D fixed-readout 측정 실험**이며 이전의 3D 커널 최적화 미완료를 새 학습 성공으로 바꾸어 부르는 것이 아니다.

{_table(counts)}

대표 조건(width=2.3, spacing=1, noise=0)의 곡률/형상 결과를각suite별로제시한다. 각suite안에서는원형을같은가중치로평균했고서로다른suite를하나의전체성능점수로합치지않았다. 원형수가작으므로조건별표와원시데이터를함께읽어야한다.

{_table(pd.DataFrame(overview))}

관측값이 비유한인 평가 행은 {invalid}개다. 비유한 값은 `nonfinite_values`, 영/음수는 `nonpositive_values`로 별도 공개하며, 불리한 형상을 재구성·위상·벽 너비·잔차 조건으로 걸러내지 않는다. 각 오차 평균의 유한값 수와 전체 행 수를 함께 기록했다. 수치 실패가 있으면 그 행을 빠뜨린 평균만으로 성공을 주장하지 않는다.

## 2. 왜 기존 gradient energy가 면적과 달라질 수 있는가

기존 readout은 I=∫|∇n|²dV다. 평면에서 n=(sech(u/w)ν,tanh(u/w)), ν가 일정한 이상적인 완전 전이 벽이라면 단위 경계면당 적분은

- 전체 gradient energy: 2/w
- scalar 성분 energy: 4/(3w)
- vector 성분 energy: 2/(3w)

이다. 따라서 energy의 물리 단위는 L^(d−2), 구하려는 둘레/면적은 L^(d−1)이다. 전체 energy가 경계 크기에 비례하려면 벽 profile 계수와 폭이 같아야 한다. **Raw energy에 raw MARE를 붙여 coarea와 직접 순위를 비교하지 않았다.** Energy도 raw q=I/GT는 기록하지만, 기하 오차는 모든 방법에 동일한 train-only 보정 정책을 적용해 별도로 비교한다.

곡면에서는 n의 법선 방향도 공간적으로 변한다. Vector energy에는 경계 크기 외에 법선 변화·곡률의 기여가 들어가고, 벽이 가까워지면 평면 단일 벽의 가정이 깨질 수 있다. 이 항을 단순 상수 보정 하나로 모든 크기와 형상에서 제거할 수 있다고 가정하지 않는다.

½∫|∇s|dV 같은 coarea readout은 완전한 단조 평면 전이에서는 폭에 무관하게 경계 크기를 돌려준다. 그러나 coarea 정리에 따르면 일반적으로 여러 s level-set의 면적을 평균하는 양이며 **항상 s=0 면적과 정확히 같은 것은 아니다.** 곡률, 유한한 ROI, 전이의 잘림, 겹치는 벽, 낮은 해상도에서 편향이 남을 수 있다. Weighted-TV/coarea readout도 그 유효 가정을 실제 데이터에서 확인해야 한다. Point coarea는 delta(s)와G4(s)를 곱하고,integrated coarea는 먼저 정규화CDF H(s)로 바꾼 뒤|G4H(s)|를 적분한다. 연속 chain rule에서는 같은 밀도지만 이산 G4에서는 같지 않다. 특히 G4는 monotone/TV-diminishing 연산자가 아니므로 후자를 underresolved wall에서도 항상 보존적인 면적이라고 부르지 않는다.

`energy_component_physics.csv`는 scalar/vector 분해를 같은 field에서 모은 결과다. 아래 분해가 이상적 2/3,1/3에서 벗어난다고 즉시 코드 오류라 판단하지 않는다. 디지털 생성장은 oracle sech/tanh profile과 다르고, 곡면의 vector 방향 변화가 추가되기 때문이다.

{_table(components, max_rows=45)}

이번에 실제 실행한 readout의 수식과 단위:

{_table(method_definitions)}

## 3. 입력장·평가창·연속체 기준과 샘플링 오차

선언한 폭 파라미터는 {widths}, 물리 spacing은 {spacings}다. Oracle에서는 물리적tanh profile scale이고,digital에서는EDT 초기화 폭이다. 이를 N10후에 실측한 최종 자구벽 폭으로 부르지 않는다. Oracle 평면은 분석적으로 주어진 profile을 격자에 표본화한다. Digital 평면과 닫힌 형상은 입력 binary mask의 EDT 초기장과 고정 G4 생성 미분·NN exchange·정규화 업데이트를 실제 실행한 field다. Oracle과 실제 생성장을 같은 것으로 부르지 않는다.

평면 ROI의 truth는 중앙 평면과 평가 box의 정확한 교차 크기다. ROI가 유한하면 폭이 넓은 profile이 다른 offset 평면의 단면을 함께 읽는다. 특히 3D oblique plane에서 단면적이 offset에 따라 달라질 수 있다. 따라서 geometry bias와 digitization bias를 분리하기 위해 `continuum_expected`가 있는 행은 독립 1D quadrature로 같은 유한 ROI·동일 readout의 연속 기대값을 계산했다.

- Geometry 오차: 측정 기하량 / 중앙 경계의 truth − 1.
- Sampling 오차: raw value / 같은 readout의 `continuum_expected` − 1.

두 오차는 다른 질문이다. Coarea가 유한 폭·유한 ROI 때문에 중앙 면적에서 벗어난 현상을 전부 격자화 실패로 돌리지 않는다. Energy의 sampling 오차는 같은 단위의 연속 energy와 비교하므로 유효하다.

Marching은 표본 중심의 box(side L−a)를 덮고 density 적분은 cell-volume box(side L)를 덮을 수 있다. 이 경우 method별 `truth_domain`과 그에 맞춘 정확 단면을 사용했다. 서로 다른 평가창을 같은 면적인 것처럼 단순 순위 비교하지 않는다. 닫힌 형상에서는 공통 연속 GT를 사용한다. Generated s=0 marching 면적은 **생성된 level-set의 추정치**이며 원래 연속 입력 형상의 정확 GT가 아니다.

{_table(planar, ['suite','dim','method','width','spacing','truth_domain','images','units',
                'q_mean','calibrated_MARE_pct','raw_geometric_MARE_pct',
                'sampling_MARE_pct','sampling_bias_pct'], max_rows=75)}

## 4. 공정한 train-only 보정과 표준 반지름 의존성

기본 보정은 width=2.3,spacing=1에서 미리 지정한 calibration-only 원/구 참조들로 정했다. 이는split='reference'로 예약된 train 역할의 표준자이며,이번 연구에 optimizer training이 있었다는 뜻은 아니다. 각 참조 원형의 pose 평균 q_r를 같은 가중치로 사용하고 c=Σq_r/Σq_r²의 relative least squares 계수를 모든 평가 조건에 고정 전이했다. 이는 test GT나 test에서 가장 좋은 반지름을 사용한 보정이 아니다. Coarea/phase와 conventional 방법에도 같은 보정 정책을 제공했으며, 차원상 기하량인 방법의 raw 오차도 함께 공개한다.

{_table(coeff)}

모든 개별 참조의 c_r=1/q_r 및 반지름 교체에 따른 오차를 공개한다. 기본 계수와 별개로 이 sweep은 민감도 진단이며 최저 test MARE를 고르는 절차가 아니다.

{_table(reference_summary)}

대표 조건의 표준자 민감도:

{_table(_condition(sensitivity), ['suite','dim','method','truth_domain','refs','c_CV_pct',
                                  'c_min','c_max','MARE_median_over_refs_pct',
                                  'MARE_worst_ref_pct'], max_rows=65)}

q_i=value_i/GT_i이면 특정 ruler r의 보정 오차는 q_i/q_r−1이다. 평가 집합의 q-CV가 감소해도 다른 참조 집합의 q_r가 벗어나면 reference 민감성이 커질 수 있다. 또한 같은 모집단의 모든 pairwise log-ratio 제곱손실은 log-q variance의 상수배이므로, 이름만 바꾸어 추가하는 것이 새 해결책은 아니다.

## 5. 곡률·얇은 구조·벽간 간섭·noise·회전/shift

다음 표는 raw q의 원형 간 dispersion과 동일한 보정 계수의 기하 오차를 구분한다. q-CV는 원형별 평균을 먼저 구해 계산했다. 측정값0도 q-CV와−100% geometry error에 포함한다. 한 원형이라도q≤0이면log-q 분산은 정의되지 않은 것으로 남기며 양수 원형만 추려 계산하지 않는다. 원형이 한 개뿐인 조건에서 q-CV=0을 일반적인 형상 독립성의 증거로 만들지 않도록 shape-dispersion 지표를 NaN으로 남겼다. 회전/shift 내부 변동은 별도 CSV에서 평가한다.

{_table(canonical, ['suite','dim','method','truth_domain','units','q_CV_pct','log_q_std',
                   'calibrated_MARE_pct','calibrated_bias_pct','calibrated_p95_abs_pct',
                   'calibrated_worst_abs_pct','raw_geometric_MARE_pct'], max_rows=75)}

전체 condition/family/noise/scale 결과는 `condition_summary.csv`, `family_summary.csv`, `noise_summary.csv`, `geometry_condition_summary.csv`에 있다. 얇은 shell과 가까운 경계는 단일 고립 자구벽 가정에 대한 stress다. 두 벽이 충분히 떨어져 있지 않으면 정상적인 profile 적분도 입력 경계 총량을 놓칠 수 있다. 이는 형상 재구성을 채점하는 실험이 아니라 readout이 어떤 feature scale에서 실패하는지 드러내는 실험이다. Width/R,thickness/width,gap/width 변수를 보고하되 이를 모델 탈락 조건으로 쓰지 않는다.

Rotation/shift의 분리는 다음과 같다. 같은 shift에서 rotation을 변화시킨 CV,같은 rotation에서 shift를 변화시킨 CV,그리고 두 변화를 함께 한 CV를 각각 기록했다. Plane ROI처럼 GT 자체가 angle에 따라 달라지면 value CV와 q CV를 구분해야 한다.

{_table(varied, max_rows=50)}

Noise는 주입 전 연속 원형을 같은 GT로 유지한 입력 오염 모형이다. 이 실험의 강건성은 그 모형에 대한 것이며 실제의 작은 구멍/돌기가 nuisance인지 신호인지 보편적으로 판정하지 않는다. Physical spacing 변화와 ROI padding도 구별해야 한다. 단순히 domain을 크게 만들었다고 continuum refinement가 되는 것은 아니다.

## 6. 방향은 무엇의 방향인가

국소 법선,전체 surface-normal anisotropy,공간적 물체의 장축,분리된 두 물체의 중심 연결축은 다른 양이다. 닫힌 물체의 법선 벡터를 단순 합하면 상쇄될 수 있으므로 nematic tensor/eigenstructure를 사용하지만,그 tensor의 축이 모든 물체의 공간 장축과 일치하는 것은 아니다.

평면 normal target에는 normal/energy tensor의 최대 고유축,공간 covariance의 최소 고유축을 사용한다. Object-major target에는 선언된 convention을 사용하며,3D에서 prolate/oblate/복합체의 의미 차이를 숨기지 않는다. 원/구의 임의 전역 장축은 정답으로 만들지 않는다. `normal_target_defined`/`orientation_defined`와 axis_role에 따라 정의된 축만 MAE 분모에 넣고,missing prediction 수는 따로 공개한다. 정의된 target에 missing prediction이 있으면 주MAE/RMSE를NaN으로 두고,유한예측만의조건부평균은별도열에만기록한다. 예측 eigengap이 작다는 이유로 불리한 행을 제거하지 않는다.

대표 형상에서의 방향 오차:

{_table(orientation, ['dim','method','normal_defined_images','normal_missing_predictions',
                     'normal_MAE_deg','normal_RMSE_deg','normal_p95_deg',
                     'spatial_defined_images','spatial_MAE_deg','spatial_p95_deg',
                     'energy_defined_images','energy_MAE_deg'], max_rows=40)}

분리된 구/원의 쌍은 surface-normal tensor가 등방적이어도 공간 covariance에는 분리축이 존재할 수 있는 예다. 아래 normal/energy와 분리축 사이 각도는 **그 target이 식별 가능하다는 정확도 점수로 쓰지 않는 시연값**이다. Spatial covariance의 분리축 오차와 구분한다.

{_table(mismatch, max_rows=30)}

## 7. 독립 연속 radial quadrature와 실제 3D 범위

{_table(continuum, max_rows=35) if continuum is not None else '별도의 continuum radial 파일이 없으므로 수행하지 않은 연속체 계산을 만들어 넣지 않았다.'}

이 표가 있으면 격자 field를 생성한 결과가 아니라 명시된 radial profile의 독립 1D 연속 적분이다. 이 대조는compact θ(u)=π/2[1−sin(πu/(2w))],|u|≤w와그밖의정확한poles를사용한다. Tanh profile이 아니므로 평면energy의2/w 계수를 그대로 대입하지 않는다(이compact profile의평면전체energy계수는π⁴/(16w)). R>w로band가원점에닿지않는범위에서계산해원점의radial-direction 특이성을피했다. 곡률에 따른 profile readout bias를 분해하는 이론 대조이며 pixelized object의 실제 digital benchmark로 세지 않는다. 형상이 작아 profile의 완전 전이 또는 고립벽 가정이 깨지는 경우도 함께 해석한다.

이번 3D 결과는 실제 volumetric fixed-readout 실험에 한정한다. 3D generator kernel을 학습하거나 이전의 미완료 3D 학습을 완료했다고 주장하지 않는다. 수치 surface-area GT는 원/구/선언된 단순 형상에서 제공한 해석식 또는 별도 검증 기준이며,generator s=0 mesh는 GT 대체가 아니다.

## 8. 비용·재현·한계와 다음 판단

{_table(timing)}

같은 field를 여러 readout이 공유하므로 field_seconds를 method별 행 수만큼 더해 전체 생성 비용으로 계산하면 안 된다. Measurement 비용은 실제 구현의 readout/mesh 연산 범위를 포함한다. Readout마다 요구하는 derivative/levelset 작업의 공유 여부에 따라 walltime이 달라질 수 있다. 정확한 process CPU/wall/RSS와 라이브러리 버전은 runner/environment 파일을 따른다. Bootstrap은 원형 단위이며 학습 참조의 불확실성을 다시 적합한 CI는 아니다. 원형 수가 작은 표는 descriptive uncertainty로 해석한다.

모든 방법은 고정된 사전 비교다. 이번 데이터에서 가장 좋은 방법을 새 모델처럼 선택하거나 같은 데이터를 새 독립 test라고 재사용하지 않는다. 복원 Dice/IoU/연결성/Euler/degree/벽 너비/잔차를 손실·수용 gate·fallback으로 사용하지 않았다. 위상 현상은 공통장의 중요한 잠재 활용이지만 이번 측정 목표의 제약은 아니다.

다음 연구의 판단은 단일 평균 순위보다 조건별 증거에 둔다. (1) 평면 sampling residual이 작으면서 width transfer가 안정적인가, (2) 실제 digital 생성장에서도 그 이점이 남는가, (3) curvature와 인접벽에서 어느 physical scale부터 깨지는가, (4) 면적과 공간 장축의 서로 다른 target을 각각 올바르게 읽는가를 확인해야 한다. Readout이 고립벽에서는 좋아도 근접벽/작은 물체에서 실패하면 이를 보편적 면적 estimator라고 부르지 않는다. 이번 디지털 결과에서energy의곡률군보정오차가작았다는점도같이보아야한다. 이상적 profile의 곡률항이 존재한다는 이론만으로 실제 생성장의 전체 오차 원인을 확정할 수 없다. 후속 비교로는G4 ringing을 피하는 양의 directional/finite-volume TV 또는 Crofton 계열의 적분이 유망하다. 이번에는 그 방법을 실행하거나새모델로선택하지 않았다.

## 9. 수치 검사·실행 비용·재현 명령

15개 수치 검사가 통과했다. G4 부호·단위·기존 동역학 일치,f불변,unit normalization,zero-H 분기,ROI후적분,해석 plane-box 교차면적,underresolved delta와ringing,곡률energy 항,두 구의법선tensor 비식별성/위치공분산을검사했다. 별도 `boundary_padding.csv`는spacing1에서같은ROI를고정하고전체domain만32격자늘리는probe다. 이는refinement가아니다. `run_cost.json`의562조건/6182행은wall180.30초,CPU179.90초,peakRSS608316KiB(약594MiB)였으며GPU는쓰지않았다.

```bash
python run_all.py --prepare --tests
python run_all.py --run
python run_all.py --diagnostics --report --paired --audit
```

기존raw.csv가있으면새실행/prepare덮어쓰기를거부한다. 새재현은output을보존용이름으로옮긴뒤시작한다. 모든generator/head/epsilon/N은고정되어있으며optimizer학습이나사후best-test선택은없다. 전수CSV와hash가제공되며대용량field checkpoint는게시하지않는다.

보조분석 `paired_readout_differences.csv`는같은원형·같은조건의보정MARE차이(readout−energy)를원형단위3000회paired bootstrap한결과다. 한원형뿐인평면조건은CI를제시하지않고,marching평면은footprint가달라paired순위비교에서제외한다. 표준자반지름·형상군이적고같은family내스케일의존성이있으므로CI를일반적모집단의보증으로해석하지않는다.

{_table(pd.read_csv(out/'paired_readout_differences.csv').query("suite in ['shape','curvature'] and width==2.3 and spacing==1 and noise==0 and method=='tv'")) if (out/'paired_readout_differences.csv').exists() else 'Paired파일은별도 --paired명령으로생성한다.'}

Spacing suite는초기물리폭2.3을고정하려고latticewidth2.3/a를사용한다. Generator와observer미분에1/a,적분에a^d를반영하지만J/h/N은그대로이므로하나의연속체PDE에대한수렴증명은아니다. 격자수를짝수로맞추면서실제domain span이조금달라질수있어actual_grid_size/actual_domain_span을원자료에남겼다. 고정ROI경계probe의padding32에따른측정변화는이번대표조건에서부동소수정밀도상0이었다. 이좁은probe를모든형상의무한domain보증으로확대하지않는다.

## 그림

{figure_text}
'''
    (out/'RESEARCH_REPORT_KO.md').write_text(text)


def analyze(out=OUT, reference_width=2.3, reference_spacing=1.0):
    out = Path(out)
    raw_path = out/'raw.csv'
    df = load_raw(raw_path)
    metadata = load_metadata(out)
    df, coeff, rulers, anchor = reference_calibration(df, reference_width, reference_spacing)
    evaluation = df[~df.is_reference].copy()
    df.to_csv(out/'analysis_raw.csv', index=False)
    coeff.to_csv(out/'calibration_coefficients.csv', index=False)
    rulers.to_csv(out/'reference_coefficients.csv', index=False)
    suite_keys = ['suite', 'dim', 'method', 'noise', 'truth_domain']
    cond_keys = ['suite', 'dim', 'method', 'width', 'spacing', 'noise', 'truth_domain']
    suite_summary = summarize(evaluation, suite_keys)
    cond_summary = summarize(evaluation, cond_keys)
    suite_summary.to_csv(out/'suite_summary.csv', index=False)
    cond_summary.to_csv(out/'condition_summary.csv', index=False)
    summarize(evaluation, cond_keys+['family']).to_csv(out/'family_summary.csv', index=False)
    summarize(evaluation, ['suite','dim','method','noise','truth_domain']).to_csv(out/'noise_summary.csv', index=False)
    summarize(evaluation, cond_keys+['family','R','thickness','gap'], bootstrap=0).to_csv(out/'geometry_condition_summary.csv', index=False)
    variation = variation_statistics(evaluation)
    variation.to_csv(out/'rotation_shift_variation.csv', index=False)
    widths = width_statistics(evaluation)
    widths.to_csv(out/'width_response.csv', index=False)
    sweep, sensitivity = ruler_statistics(evaluation, anchor)
    sweep.to_csv(out/'reference_radius_sweep.csv', index=False)
    sensitivity.to_csv(out/'ruler_sensitivity_summary.csv', index=False)
    components = component_statistics(evaluation)
    components.to_csv(out/'energy_component_physics.csv', index=False)
    pair = evaluation[evaluation.get('axis_role', pd.Series('',index=evaluation.index)).eq('pair_separation')]
    pair.to_csv(out/'pair_direction_demonstration.csv', index=False)
    continuum = None
    radial_path = out/'continuum_radial.csv'
    if radial_path.exists():
        continuum = pd.read_csv(radial_path)
        needed = {'dim','R','width','method','value','truth'}
        if not needed <= set(continuum):
            raise ValueError(f'continuum_radial.csv must include {needed}')
        continuum['q_raw'] = continuum.value/continuum.truth
        continuum['raw_geometric_error_pct'] = np.where(
            ~continuum.method.map(is_energy_method),100*(continuum.q_raw-1),np.nan)
        continuum.to_csv(out/'continuum_radial_analysis.csv', index=False)
    figures = make_figures(evaluation, cond_summary, sweep, out, continuum)
    write_report(df, coeff, rulers, suite_summary, cond_summary, variation, sweep,
                 sensitivity, components, metadata, figures, out, continuum)
    audit = dict(raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                 analysis_only=True, model_selection=False, test_reference_selection=False,
                 train_only_global_calibration=True, reference_width=reference_width,
                 reference_spacing=reference_spacing, reference_units=int(anchor.reference_unit.nunique()),
                 evaluation_cases=int(evaluation.case.nunique()),
                 evaluation_rows=len(evaluation), finite_rows=int(np.isfinite(evaluation.value).sum()),
                 methods=sorted(evaluation.method.unique().tolist()),
                 dimensions=sorted(evaluation.dim.unique().astype(int).tolist()),
                 generated_levelset_not_GT=True, method_specific_planar_truth_domain=True,
                 normal_and_spatial_targets_separate=True)
    (out/'analysis_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,ensure_ascii=False))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUT))
    parser.add_argument('--reference-width', type=float, default=2.3)
    parser.add_argument('--reference-spacing', type=float, default=1.0)
    args = parser.parse_args()
    analyze(args.output,args.reference_width,args.reference_spacing)
