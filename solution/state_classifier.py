# -*- coding: utf-8 -*-
"""
state_classifier.py — JEBI Hackathon 2026

Clasificador de 3 estados para eventos detectados en el ciclo de la pala EX-5600.

Estados:
  INACTIVIDAD_JUSTIFICADA:    pala detenida por causa externa (camion no listo, etc.)
  INACTIVIDAD_INJUSTIFICADA:  pala detenida sin razon aparente
  ACTIVIDAD_CONTRAPRODUCENTE: pala en movimiento pero fuera del ciclo productivo

Adaptado del proyecto jevi/ para integrarlo al pipeline actual.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict


# ─── DEFINICIÓN DE ESTADOS ────────────────────────────────────────────────────

STATES: Dict[str, Dict] = {
    'INACTIVIDAD_JUSTIFICADA': {
        'color':  '#d29922',
        'icon':   '⏸',
        'label':  'Inactividad Justificada',
        'short':  'JUSTIFICADA',
    },
    'INACTIVIDAD_INJUSTIFICADA': {
        'color':  '#f85149',
        'icon':   '⚠',
        'label':  'Inactividad Injustificada',
        'short':  'INJUSTIFICADA',
    },
    'ACTIVIDAD_CONTRAPRODUCENTE': {
        'color':  '#bc8cff',
        'icon':   '↯',
        'label':  'Actividad Contraproducente',
        'short':  'CONTRAPROD.',
    },
}

# Umbral calibrado: midpoint entre pre-gyro_mean de justificada (~9.4) vs
# injustificada (~12.0) observados en los 2 eventos reales (2:40 y 4:53)
_GYRO_PRE_THR = 10.5     # deg/s


# ─── CLASIFICACIÓN DE IDLE ────────────────────────────────────────────────────

def classify_idle(seg: Dict, df_imu: pd.DataFrame) -> str:
    """
    Clasifica un segmento idle como JUSTIFICADA o INJUSTIFICADA.

    Filosofía:
      - Si la pala venía GIRANDO FUERTE justo antes de detenerse, probablemente
        estaba esperando un camión que no llegó → INJUSTIFICADA.
      - Si venía con movimiento moderado, es una pausa natural → JUSTIFICADA.

    Args:
        seg: dict con 'tiempo_inicio_s', 'tiempo_fin_s', 'duracion_s'
        df_imu: DataFrame con columnas 'time_s', 'gyro_mag'

    Returns:
        'INACTIVIDAD_JUSTIFICADA' | 'INACTIVIDAD_INJUSTIFICADA'
    """
    t0 = seg.get('tiempo_inicio_s', seg.get('t_start', 0))

    # Mirar los 15s previos al idle
    if 'time_s' not in df_imu.columns:
        # Compatibilidad con reporter actual que usa 'timestamp_s'
        t_col = 'timestamp_s'
    else:
        t_col = 'time_s'

    if 'gyro_mag' not in df_imu.columns:
        return 'INACTIVIDAD_INJUSTIFICADA'

    pre = df_imu[(df_imu[t_col] >= t0 - 15) & (df_imu[t_col] < t0)]
    if len(pre) < 5:
        return 'INACTIVIDAD_INJUSTIFICADA'

    return ('INACTIVIDAD_INJUSTIFICADA'
            if pre['gyro_mag'].mean() > _GYRO_PRE_THR
            else 'INACTIVIDAD_JUSTIFICADA')


# ─── ACTIVIDAD CONTRAPRODUCENTE ───────────────────────────────────────────────

def _periodicity(gyro: np.ndarray, sr: float,
                  min_lag: float = 10.0, max_lag: float = 40.0) -> float:
    """
    Pico de autocorrelación en el rango de ciclo esperado (10-40s).
    Retorna valor 0-1: 1 = patrón periódico fuerte (ciclos productivos).
    """
    n = len(gyro)
    if n < 30:
        return 0.0
    s = gyro - gyro.mean()
    if s.std() < 1e-8:
        return 0.0
    s = s / s.std()
    ac = np.correlate(s, s, mode='full')[n - 1:]
    ac = ac / ac[0]
    lo = int(min_lag * sr)
    hi = min(int(max_lag * sr), n // 2)
    if lo >= hi:
        return 0.0
    return float(np.clip(ac[lo:hi].max(), 0.0, 1.0))


def detect_contraproductive(df_imu: pd.DataFrame,
                              idle_segs: List[Dict],
                              min_duration_s: float = 180.0,
                              periodicity_thr: float = 0.15,
                              active_gyro_thr: float = 10.0) -> List[Dict]:
    """
    Detecta periodos activos (no idle) donde la pala se mueve pero SIN seguir
    el ciclo productivo (excavar → girar → descargar).

    Criterio: segmento activo ≥ min_duration_s donde la autocorrelación del
    gyro_mag NO muestra patrón periódico → pala moviéndose fuera del ciclo.

    Args:
        df_imu: DataFrame con 'time_s', 'timestamp_s', 'gyro_mag'
        idle_segs: lista de segmentos idle con 't_start'/'tiempo_inicio_s'
        min_duration_s: duración mínima de segmento activo a evaluar
        periodicity_thr: debajo de este valor → no periódico → contraprod.
        active_gyro_thr: mínimo movimiento para considerar segmento "activo"

    Returns:
        list[dict] con eventos ACTIVIDAD_CONTRAPRODUCENTE
    """
    if 'time_s' not in df_imu.columns or 'gyro_mag' not in df_imu.columns:
        return []

    if len(df_imu) == 0:
        return []

    t_max = float(df_imu['time_s'].iloc[-1])

    # Construir intervalos "no-idle" entre idle_segs
    idle_ranges = sorted(
        (s.get('tiempo_inicio_s', s.get('t_start', 0)),
         s.get('tiempo_fin_s',    s.get('t_end', 0)))
        for s in idle_segs
    )
    boundaries = [0.0]
    for pair in idle_ranges:
        boundaries.extend(pair)
    boundaries.append(t_max)

    # Los intervalos ACTIVOS están entre pares consecutivos saltando idle
    active_intervals: List = []
    # boundaries = [0, i1_start, i1_end, i2_start, i2_end, ..., t_max]
    # active = (0, i1_start), (i1_end, i2_start), ..., (ik_end, t_max)
    i = 0
    while i < len(boundaries) - 1:
        a, b = boundaries[i], boundaries[i + 1]
        if b - a >= min_duration_s:
            active_intervals.append((a, b))
        i += 2  # saltear el segmento idle que sigue

    events: List[Dict] = []
    for t0, t1 in active_intervals:
        ctx = df_imu[(df_imu['time_s'] >= t0) & (df_imu['time_s'] <= t1)]
        if len(ctx) < 30:
            continue

        dur = float(ctx['time_s'].iloc[-1] - ctx['time_s'].iloc[0]) + 1e-8
        sr = len(ctx) / dur

        # Sólo marcar si la máquina se mueve (no es otro idle no detectado)
        if ctx['gyro_mag'].mean() < active_gyro_thr:
            continue

        p = _periodicity(ctx['gyro_mag'].values, sr)
        if p < periodicity_thr:
            events.append({
                'estado':          'ACTIVIDAD_CONTRAPRODUCENTE',
                'tiempo_inicio_s': round(t0, 3),
                'tiempo_fin_s':    round(t1, 3),
                'duracion_s':      round(t1 - t0, 3),
                'periodicity':     round(p, 3),
            })
    return events
