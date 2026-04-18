# -*- coding: utf-8 -*-
"""
imu_processor.py
Carga el IMU (CSV o NPY), detecta ciclos completos, mini-ciclos,
tiempos de espera (wait) y calcula metricas de eficiencia.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.ndimage import uniform_filter1d

# ─── CONSTANTES ───────────────────────────────────────────────────────────────

BUCKET_M3       = 27.0       # m³ del bucket EX-5600
MATERIAL_T_M3   = 2.5        # t/m³ densidad tipica de roca
BUCKET_MAX_T    = BUCKET_M3 * MATERIAL_T_M3   # ~67.5 t maxima por cucharada

CAP_793F        = 221.0      # toneladas CAT 793F
CAP_EH4000      = 218.0      # toneladas EH4000 AC-3

GYRO_RMS_THRESH    = 20.0    # deg/s  — umbral RMS gx para swing (same as analyze_deep)
GYRO_RMS_WINDOW_S  = 1.5     # segundos ventana RMS
SWING_MERGE_GAP_S  = 3.0     # gap maximo para fusionar swing en mismo cluster
CYCLE_PAIR_GAP_S   = 18.0    # gap maximo para parear swing cargado + vacio
WAIT_MIN_SEC       = 5.0     # segundos minimos para clasificar como WAIT
SWING_MIN_PEAK     = 25.0    # deg/s minimo pico para ser swing real (no ruido)
SWING_MIN_DUR_S    = 0.8     # segundos minimos de duracion del swing
ACCEL_DUMP_THRESH  = 14.0    # m/s²   — spike accel = dump de material
MINI_PEAK_RATIO    = 0.45    # pico < 45% del p75 de picos = mini-ciclo

# ─── DATACLASSES ──────────────────────────────────────────────────────────────

@dataclass
class WaitEvent:
    t_start:    float
    t_end:      float
    duration_s: float
    reason:     str   # 'pre_load' | 'inter_cycle' | 'end_session'

@dataclass
class SwingPhase:
    t_start:    float
    t_end:      float
    peak_gyro:  float
    duration_s: float
    is_loaded:  bool   # True = swing cargado, False = swing vacio de retorno

@dataclass
class LoadCycle:
    cycle_id:       int
    t_start:        float       # inicio del swing cargado
    t_end:          float       # fin del swing vacio
    t_dump:         Optional[float]  # timestamp del dump
    swing_loaded:   SwingPhase
    swing_empty:    Optional[SwingPhase]
    fill_factor:    float       # 0.0 - 1.0
    payload_t:      float       # toneladas estimadas
    is_mini_cycle:  bool
    truck_id:       Optional[str] = None  # asignado por video_processor
    dump_frame_sec: Optional[float] = None

@dataclass
class SessionMetrics:
    total_duration_s:    float
    n_full_cycles:       int
    n_mini_cycles:       int
    n_wait_events:       int
    total_wait_s:        float
    total_productive_s:  float
    time_efficiency_pct: float
    avg_fill_factor_pct: float
    avg_payload_t:       float
    total_payload_t:     float
    avg_cycle_time_s:    float
    cycles_per_hour:     float
    productivity_tph:    float   # toneladas por hora
    underfill_count:     int     # ciclos con fill < 80%
    underfill_loss_t:    float   # toneladas perdidas por subfilling

# ─── CARGA DEL IMU ────────────────────────────────────────────────────────────

def load_imu(path: str, sync_duration_s: float = None) -> pd.DataFrame:
    """
    Carga IMU desde CSV o NPY.

    Args:
        path: Path al CSV/NPY del IMU.
        sync_duration_s: Si se provee, REEMPLAZA los timestamps del IMU con
            una grilla uniforme sobre [0, sync_duration_s]. Esto sirve cuando
            los timestamps del IMU tienen gaps irregulares pero son 1:1 con
            los frames de un video (caso JEBI: 9403 IMU samples = 9403 video
            frames, pero los dt del IMU son inconsistentes).

    Retorna DataFrame con columnas estandar:
      timestamp_s, ax, ay, az, gx, gy, gz, qw, qx, qy, qz
    """
    if path.endswith('.npy'):
        raw = np.load(path)
        ts = (raw[:, 0] - raw[0, 0]) / 1e9   # nanosegundos → segundos
        df = pd.DataFrame({
            'timestamp_s': ts,
            'ax': raw[:, 1], 'ay': raw[:, 2], 'az': raw[:, 3],
            'gx': raw[:, 4], 'gy': raw[:, 5], 'gz': raw[:, 6],
            'qw': raw[:, 7], 'qx': raw[:, 8], 'qy': raw[:, 9], 'qz': raw[:, 10],
        })
    else:
        # CSV — intentar detectar columnas automaticamente
        raw = pd.read_csv(path)
        raw.columns = [c.lower().strip() for c in raw.columns]
        df = _normalize_csv_columns(raw)

    # Calcular normas
    df['accel_norm'] = np.sqrt(df.ax**2 + df.ay**2 + df.az**2)
    df['gyro_norm']  = np.sqrt(df.gx**2 + df.gy**2 + df.gz**2)

    # BUGFIX CRÍTICO: si tenemos la duración del video sincronizado,
    # rebalanceamos los timestamps del IMU a una grilla uniforme [0, video_duration]
    # Esto es porque el IMU del dataset JEBI tiene timestamps irregulares
    # (gaps hasta 933ms) pero es 1:1 con frames del video (9403=9403).
    # Usar timestamps inconsistentes → duración inflada (899s vs 627s real).
    n = len(df)
    if sync_duration_s is not None and sync_duration_s > 0:
        original_duration = float(df['timestamp_s'].iloc[-1])
        df['timestamp_s'] = np.linspace(0.0, sync_duration_s, n)
        print(f"  IMU sincronizado al video: {original_duration:.1f}s → {sync_duration_s:.1f}s "
              f"(corrección {sync_duration_s/max(original_duration,1e-9):.3f}x)")

    # Sampleo
    dt = float(np.mean(np.diff(df.timestamp_s.values)))
    df.attrs['fs']           = 1.0 / dt
    df.attrs['duration_s']   = float(df.timestamp_s.iloc[-1])
    df.attrs['n_samples']    = n

    print(f"  IMU cargado: {n} muestras, "
          f"fs={df.attrs['fs']:.1f}Hz, "
          f"duracion={df.attrs['duration_s']:.1f}s")
    return df


def get_video_duration_s(video_path: str) -> float:
    """
    Obtiene la duración real del video en segundos.
    Retorna 0 si no se puede leer.
    """
    if not video_path:
        return 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0 and n_frames > 0:
            return n_frames / fps
    except Exception:
        pass
    return 0.0


def _normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea nombres de columnas CSV a nombres estandar."""
    alias = {
        'timestamp_s': ['timestamp_s', 'timestamp', 'time', 't', 'ts'],
        'ax': ['ax', 'accel_x', 'acceleration_x', 'a_x'],
        'ay': ['ay', 'accel_y', 'acceleration_y', 'a_y'],
        'az': ['az', 'accel_z', 'acceleration_z', 'a_z'],
        'gx': ['gx', 'gyro_x', 'gyroscope_x', 'g_x', 'wx'],
        'gy': ['gy', 'gyro_y', 'gyroscope_y', 'g_y', 'wy'],
        'gz': ['gz', 'gyro_z', 'gyroscope_z', 'g_z', 'wz'],
        'qw': ['qw', 'quat_w', 'quaternion_w', 'w'],
        'qx': ['qx', 'quat_x', 'quaternion_x'],
        'qy': ['qy', 'quat_y', 'quaternion_y'],
        'qz': ['qz', 'quat_z', 'quaternion_z'],
    }
    rename_map = {}
    for target, candidates in alias.items():
        for c in candidates:
            if c in df.columns and target not in rename_map.values():
                rename_map[c] = target
                break

    df = df.rename(columns=rename_map)

    # Si timestamp no es en segundos (podria ser nanosegundos)
    if 'timestamp_s' in df.columns:
        ts = df['timestamp_s'].values
        if ts[0] > 1e12:          # nanosegundos
            ts = (ts - ts[0]) / 1e9
        elif ts[0] > 1e9:         # milisegundos
            ts = (ts - ts[0]) / 1e3
        else:                      # segundos relativos o absolutos
            ts = ts - ts[0]
        df['timestamp_s'] = ts

    # Columnas quaternion opcionales — si no estan, llenar con identidad
    for col, default in [('qw', 1.0), ('qx', 0.0), ('qy', 0.0), ('qz', 0.0)]:
        if col not in df.columns:
            df[col] = default

    return df[['timestamp_s', 'ax', 'ay', 'az', 'gx', 'gy', 'gz',
               'qw', 'qx', 'qy', 'qz']]


# ─── DETECCION DE CICLOS ──────────────────────────────────────────────────────

def detect_cycles(df: pd.DataFrame):
    """
    Detecta ciclos de carga, tiempos de espera y mini-ciclos.
    Usa el mismo enfoque RMS de gx que analyze_deep.py (probado en los datos).
    Retorna (cycles, wait_events, metrics).
    """
    fs  = df.attrs['fs']
    t   = df.timestamp_s.values
    gx  = df.gx.values
    an  = df.accel_norm.values
    az  = df.az.values

    # ── RMS de gx (mismo metodo que analyze_deep.py) ──────────────────────────
    ws = max(1, int(fs * GYRO_RMS_WINDOW_S))
    gx_rms = np.array([
        np.sqrt(np.mean(gx[max(0, i - ws//2): i + ws//2 + 1]**2))
        for i in range(len(gx))
    ])

    # ── Segmentos de swing (gx_rms > umbral) ─────────────────────────────────
    active_mask = gx_rms > GYRO_RMS_THRESH
    raw_swings  = _find_segments(active_mask, t)

    # Filtrar swings muy cortos (ruido) o pico muy bajo
    gn = df.gyro_norm.values
    valid_swings = []
    for s in raw_swings:
        if s['dur'] < SWING_MIN_DUR_S:
            continue
        peak = _peak_in_segment(gn, t, s['t0'], s['t1'])
        if peak < SWING_MIN_PEAK:
            continue
        s['peak'] = peak
        valid_swings.append(s)

    # ── Agrupar swings en clusters cercanos (gap < SWING_MERGE_GAP_S) ─────────
    swing_clusters = _group_segments(valid_swings, max_gap_s=SWING_MERGE_GAP_S)
    # Recalcular peak por cluster
    for c in swing_clusters:
        c['peak'] = _peak_in_segment(gn, t, c['t0'], c['t1'])

    # ── Umbral de mini-ciclo: p75 de todos los picos ───────────────────────────
    all_peaks = [c['peak'] for c in swing_clusters]
    p75_peak  = float(np.percentile(all_peaks, 75)) if all_peaks else 50.0

    # ── Parear swing cargado + swing vacio → ciclo completo ──────────────────
    # Dos clusters consecutivos con gap < CYCLE_PAIR_GAP_S → un ciclo
    # Un cluster solitario → ciclo incompleto o mini
    cycles: List[LoadCycle] = []
    cycle_id = 1
    used = set()

    for i, sw in enumerate(swing_clusters):
        if i in used:
            continue

        # Buscar el siguiente swing como posible "empty return"
        sw_empty_cand = None
        if i + 1 < len(swing_clusters):
            nxt = swing_clusters[i + 1]
            gap = nxt['t0'] - sw['t1']
            if gap <= CYCLE_PAIR_GAP_S:
                sw_empty_cand = nxt
                used.add(i + 1)

        used.add(i)

        # Clasificar mini-ciclo
        is_mini = sw['peak'] < p75_peak * MINI_PEAK_RATIO

        # Crear objetos SwingPhase
        sw_loaded_obj = SwingPhase(
            t_start=sw['t0'], t_end=sw['t1'],
            peak_gyro=sw['peak'], duration_s=sw['dur'],
            is_loaded=True
        )
        sw_empty_obj = None
        if sw_empty_cand:
            sw_empty_obj = SwingPhase(
                t_start=sw_empty_cand['t0'], t_end=sw_empty_cand['t1'],
                peak_gyro=sw_empty_cand['peak'],
                duration_s=sw_empty_cand['dur'],
                is_loaded=False
            )

        # Ventana de tiempo del ciclo completo
        t_cycle_start = sw['t0']
        t_cycle_end   = sw_empty_cand['t1'] if sw_empty_cand else sw['t1']

        # Detectar dump entre los dos swings
        i0 = np.searchsorted(t, t_cycle_start)
        i1 = np.searchsorted(t, t_cycle_end)
        seg_az = az[i0:i1]
        seg_an = an[i0:i1]
        seg_t  = t[i0:i1]
        dump_t = _detect_dump_time(seg_t, seg_az, seg_an) if len(seg_t) > 0 else sw['t1']

        # Fill factor
        sub_swings_info = [sw]
        if sw_empty_cand:
            sub_swings_info.append(sw_empty_cand)
        fill_factor = _estimate_fill_factor(sub_swings_info, p75_peak, is_mini)
        payload_t   = fill_factor * BUCKET_MAX_T

        cycles.append(LoadCycle(
            cycle_id=cycle_id,
            t_start=t_cycle_start,
            t_end=t_cycle_end,
            t_dump=dump_t,
            swing_loaded=sw_loaded_obj,
            swing_empty=sw_empty_obj,
            fill_factor=fill_factor,
            payload_t=payload_t,
            is_mini_cycle=is_mini,
            dump_frame_sec=dump_t,
        ))
        cycle_id += 1

    # ── Wait events: gaps entre ciclos ────────────────────────────────────────
    # Construimos mascara de "en ciclo" y buscamos los gaps
    in_cycle = np.zeros(len(t), dtype=bool)
    for c in cycles:
        i0 = np.searchsorted(t, c.t_start)
        i1 = np.searchsorted(t, c.t_end)
        in_cycle[i0:i1] = True

    idle_segs = _find_segments(~in_cycle, t)
    wait_events: List[WaitEvent] = []
    for s in idle_segs:
        if s['dur'] >= WAIT_MIN_SEC:
            wait_events.append(WaitEvent(
                t_start=s['t0'], t_end=s['t1'],
                duration_s=s['dur'],
                reason='unknown'
            ))

    # Clasificar tipo de wait
    cycle_starts = [c.t_start for c in cycles]
    for w in wait_events:
        next_cs = [cs for cs in cycle_starts if cs >= w.t_end]
        if next_cs and (min(next_cs) - w.t_end) < 8.0:
            w.reason = 'pre_load'
        elif w.t_start > t[-1] - 30:
            w.reason = 'end_session'
        else:
            w.reason = 'inter_cycle'

    # ── Metricas globales ─────────────────────────────────────────────────────
    metrics = _compute_metrics(cycles, wait_events, df.attrs['duration_s'])

    return cycles, wait_events, metrics


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _find_segments(mask: np.ndarray, t: np.ndarray):
    """Encuentra segmentos continuos donde mask=True."""
    segs = []
    in_seg = False
    s_i = 0
    for i in range(len(mask)):
        if mask[i] and not in_seg:
            in_seg, s_i = True, i
        elif not mask[i] and in_seg:
            segs.append({'t0': t[s_i], 't1': t[i-1],
                         'si': s_i, 'ei': i-1,
                         'dur': t[i-1] - t[s_i]})
            in_seg = False
    if in_seg:
        segs.append({'t0': t[s_i], 't1': t[-1],
                     'si': s_i, 'ei': len(t)-1,
                     'dur': t[-1] - t[s_i]})
    return segs


def _group_segments(segs, max_gap_s=8.0):
    """Fusiona segmentos activos separados por menos de max_gap_s."""
    if not segs:
        return []
    merged = [dict(segs[0])]
    for s in segs[1:]:
        if (s['t0'] - merged[-1]['t1']) <= max_gap_s:
            merged[-1]['t1']  = s['t1']
            merged[-1]['dur'] = merged[-1]['t1'] - merged[-1]['t0']
        else:
            merged.append(dict(s))
    return merged


def _peak_in_segment(gn, t, t0, t1):
    idx = np.where((t >= t0) & (t <= t1))[0]
    return float(np.max(gn[idx])) if len(idx) > 0 else 0.0


def _find_sub_swings(seg_t, seg_gn):
    """
    Dentro de un ciclo, separa el swing cargado del swing de retorno
    buscando el valle (minimo) que los separa.
    """
    if len(seg_gn) < 4:
        return []

    # Suavizar para encontrar valles
    smooth = uniform_filter1d(seg_gn, size=max(1, len(seg_gn) // 8))

    # Valle principal separador
    mid = len(smooth) // 2
    region = smooth[max(0, mid-len(smooth)//3) : min(len(smooth), mid+len(smooth)//3)]
    if len(region) == 0:
        valley_idx = mid
    else:
        valley_local = int(np.argmin(region))
        valley_idx   = max(0, mid - len(smooth)//3) + valley_local

    swings = []
    for t_slice, gn_slice in [
        (seg_t[:valley_idx], seg_gn[:valley_idx]),
        (seg_t[valley_idx:], seg_gn[valley_idx:])
    ]:
        if len(t_slice) < 2:
            continue
        swings.append({
            't0':  float(t_slice[0]),
            't1':  float(t_slice[-1]),
            'dur': float(t_slice[-1] - t_slice[0]),
            'peak': float(np.max(gn_slice))
        })
    return swings


def _detect_dump_time(seg_t, seg_az, seg_an):
    """
    Detecta el momento del dump:
    spike de accel_norm > threshold O minimo de az.
    """
    # Spike en accel_norm
    high_accel = np.where(seg_an > ACCEL_DUMP_THRESH)[0]
    if len(high_accel) > 0:
        return float(seg_t[high_accel[0]])
    # Fallback: minimo de az (material cayendo)
    return float(seg_t[int(np.argmin(seg_az))])


def _estimate_fill_factor(sub_swings, median_peak, is_mini):
    """
    Estima fill factor comparando pico del swing cargado
    vs pico del swing vacio.
    
    Principio: loaded swing tiene mas inercia → peak_gyro mas bajo
    comparado con swing vacio (mismo torque, mas masa = menos aceleracion).
    
    Fill factor = 1 - (peak_loaded / peak_empty) normalizado.
    Si solo hay un swing → estimacion por comparacion con mediana.
    """
    if is_mini:
        return 0.30  # mini-ciclos asumimos llenado parcial bajo

    if len(sub_swings) == 0:
        return 0.60  # default conservador

    peak_loaded = sub_swings[0]['peak']

    if len(sub_swings) >= 2:
        peak_empty = sub_swings[1]['peak']
    else:
        # Estimacion: swing vacio seria ~1.3x el swing cargado en peak
        peak_empty = peak_loaded * 1.3

    # Normalizamos: si peak_loaded == peak_empty → balde vacio (fill=0)
    # Si peak_loaded << peak_empty → balde lleno (fill=1)
    ratio = peak_loaded / max(peak_empty, 1.0)   # < 1 si cargado
    fill = 1.0 - np.clip(ratio - 0.5, 0, 1.0) / 0.5

    # Calibracion suave: usamos mediana global como referencia
    fill = np.clip(fill, 0.15, 1.0)
    return float(fill)


def _compute_metrics(cycles, wait_events, duration_s) -> SessionMetrics:
    full_cycles  = [c for c in cycles if not c.is_mini_cycle]
    mini_cycles  = [c for c in cycles if c.is_mini_cycle]
    total_wait_s = sum(w.duration_s for w in wait_events)

    total_productive_s = duration_s - total_wait_s
    time_eff = (total_productive_s / duration_s * 100) if duration_s > 0 else 0

    fill_factors  = [c.fill_factor for c in full_cycles] or [0]
    payloads      = [c.payload_t   for c in full_cycles] or [0]
    cycle_times   = [(c.t_end - c.t_start) for c in full_cycles] or [0]

    avg_fill    = float(np.mean(fill_factors)) * 100
    avg_payload = float(np.mean(payloads))
    total_payload = float(np.sum(payloads))
    avg_cycle   = float(np.mean(cycle_times))
    hours       = duration_s / 3600.0
    cycles_hr   = len(full_cycles) / hours if hours > 0 else 0
    prod_tph    = total_payload / hours if hours > 0 else 0

    underfill   = [c for c in full_cycles if c.fill_factor < 0.80]
    loss_t      = sum((0.80 - c.fill_factor) * BUCKET_MAX_T for c in underfill)

    return SessionMetrics(
        total_duration_s    = duration_s,
        n_full_cycles       = len(full_cycles),
        n_mini_cycles       = len(mini_cycles),
        n_wait_events       = len(wait_events),
        total_wait_s        = total_wait_s,
        total_productive_s  = total_productive_s,
        time_efficiency_pct = round(time_eff, 1),
        avg_fill_factor_pct = round(avg_fill, 1),
        avg_payload_t       = round(avg_payload, 1),
        total_payload_t     = round(total_payload, 1),
        avg_cycle_time_s    = round(avg_cycle, 1),
        cycles_per_hour     = round(cycles_hr, 1),
        productivity_tph    = round(prod_tph, 1),
        underfill_count     = len(underfill),
        underfill_loss_t    = round(loss_t, 1),
    )


def cycles_to_dataframe(cycles: List[LoadCycle], wait_events: List[WaitEvent]):
    """Convierte ciclos y waits a DataFrames para exportar CSV."""
    cycle_rows = []
    for c in cycles:
        cycle_rows.append({
            'cycle_id':      c.cycle_id,
            'type':          'mini' if c.is_mini_cycle else 'full',
            't_start_s':     round(c.t_start, 2),
            't_end_s':       round(c.t_end, 2),
            't_dump_s':      round(c.t_dump, 2) if c.t_dump else None,
            'duration_s':    round(c.t_end - c.t_start, 1),
            'fill_factor_pct': round(c.fill_factor * 100, 1),
            'payload_t':     round(c.payload_t, 1),
            'peak_gyro_loaded': round(c.swing_loaded.peak_gyro, 1) if c.swing_loaded else None,
            'peak_gyro_empty':  round(c.swing_empty.peak_gyro, 1)  if c.swing_empty  else None,
            'truck_id':      c.truck_id or 'unknown',
        })

    wait_rows = []
    for w in wait_events:
        wait_rows.append({
            't_start_s':  round(w.t_start, 2),
            't_end_s':    round(w.t_end, 2),
            'duration_s': round(w.duration_s, 1),
            'reason':     w.reason,
        })

    return pd.DataFrame(cycle_rows), pd.DataFrame(wait_rows)
