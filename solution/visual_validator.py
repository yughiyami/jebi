# -*- coding: utf-8 -*-
"""
visual_validator.py — JEBI Hackathon 2026

Validador visual dual-source.

ANTES: usaba YOLO para detectar camiones y validar si había actividad.
AHORA: usa OPTICAL FLOW (motion_detector.py) para detectar movimiento de cámara.

¿Por qué el cambio?
  - La cámara va montada sobre la pala → camera motion = pala motion
  - Optical flow es classical CV, cero entrenamiento, super rápido
  - Más directo y confiable que detectar objetos

Este módulo ahora es un THIN WRAPPER sobre motion_detector, orientado a
producir el shape de datos que el resto del pipeline espera.
"""

from typing import List, Dict
from motion_detector import (
    compute_motion_timeline,
    cross_validate_waits,
    cross_validate_cycles,
    compute_verified_idle,
    compute_productivity_loss_alerts,
    motion_timeline_to_dict,
    MotionTimeline,
)


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def run_visual_validation(video_path: str,
                           duration_s: float,
                           wait_events: List,
                           cycles: List,
                           interval_s: float = 1.0) -> Dict:
    """
    Pipeline completo de validación visual con optical flow.

    Args:
        video_path:   path al video (left o right)
        duration_s:   duración de la sesión (segundos)
        wait_events:  list[WaitEvent] del IMU
        cycles:       list[LoadCycle] del IMU
        interval_s:   cada cuánto muestrear (1.0s default)

    Returns:
        Dict serializable con:
          - motion_timeline:       serie temporal de motion
          - verified_idle:         métricas de idle cruzado IMU+visual
          - wait_confidence:       confianza por cada wait_event
          - cycle_confidence:      confianza por cada cycle
          - productivity_alerts:   alertas operador por pérdida productiva
          - overall_confidence:    HIGH | MEDIUM | LOW
          - confidence_score:      0-1
          - disagreements:         casos IMU↔visual que disienten
    """
    print(f'\n  [validator] Iniciando validación por optical flow')

    # 1. Calcular motion timeline sobre video completo
    motion_tl = compute_motion_timeline(video_path, duration_s, interval_s)
    if not motion_tl.samples:
        return _empty_result(interval_s)

    # 2. Cruzar con IMU
    wait_conf = cross_validate_waits(wait_events, motion_tl)
    cycle_conf = cross_validate_cycles(cycles, motion_tl)
    verified = compute_verified_idle(wait_events, motion_tl, duration_s)

    # 3. Alertas de pérdida productiva (idle periods detectados por visual)
    prod_alerts = compute_productivity_loss_alerts(motion_tl.idle_periods)

    # 4. Detectar discrepancias concretas
    disagreements = []
    for wc in wait_conf:
        if not wc['validated']:
            disagreements.append({
                'type':          'IMU_IDLE_VISUAL_MOVING',
                't_start':       wc['t_start'],
                't_end':         wc['t_end'],
                'duration_s':    wc['duration_s'],
                'visual_score':  wc['motion_score'],
                'note':          'IMU marca ocio pero la cámara detecta movimiento',
            })
    for cc in cycle_conf:
        if cc['confidence'] == 'LOW':
            disagreements.append({
                'type':          'IMU_ACTIVE_VISUAL_STATIC',
                'cycle_id':      cc['cycle_id'],
                't_start':       cc['t_start'],
                't_end':         cc['t_end'],
                'visual_score':  cc['motion_score'],
                'note':          'IMU marca ciclo pero la cámara no detecta movimiento',
            })

    # Logs informativos
    print(f'  [validator] Motion avg: {motion_tl.avg_motion_magnitude:.2f} (max {motion_tl.max_motion_magnitude:.2f})')
    print(f'  [validator] Tiempo estático (cámara quieta): {motion_tl.total_static_s:.0f}s')
    print(f'  [validator] Períodos idle visuales (>10s): {len(motion_tl.idle_periods)}')
    print(f'  [validator] Idle verificado (IMU+visual): {verified["verified_idle_s"]:.0f}s')
    print(f'  [validator] Acuerdo entre fuentes: {verified["agreement_pct"]:.1f}%')
    print(f'  [validator] Confianza global: {verified["confidence"]}')
    print(f'  [validator] Alertas productividad: {len(prod_alerts)}')
    if disagreements:
        print(f'  [validator] ⚠ {len(disagreements)} discrepancias')

    return {
        'motion_timeline':      motion_timeline_to_dict(motion_tl),
        'verified_idle':        verified,
        'wait_confidence':      wait_conf,
        'cycle_confidence':     cycle_conf,
        'productivity_alerts':  prod_alerts,
        'overall_confidence':   verified['confidence'],
        'confidence_score':     verified['confidence_score'],
        'disagreements':        disagreements,
    }


def _empty_result(interval_s: float) -> Dict:
    return {
        'motion_timeline':     {'n_samples': 0, 'idle_periods': [], 'series': []},
        'verified_idle':       {
            'verified_idle_s':    0.0, 'verified_active_s': 0.0,
            'imu_only_idle_s':    0.0, 'visual_only_idle_s': 0.0,
            'verified_iti':       0.0, 'verified_iti_pct': 0.0,
            'agreement_pct':      0.0, 'confidence': 'LOW',
            'confidence_score':   0.0, 'n_idle_periods_visual': 0,
        },
        'wait_confidence':     [],
        'cycle_confidence':    [],
        'productivity_alerts': [],
        'overall_confidence':  'LOW',
        'confidence_score':    0.0,
        'disagreements':       [],
    }


# ─── INTEGRACIÓN CON IDLE INDEX (retrocompat) ─────────────────────────────────

def enhance_idle_index(idle_index: Dict, validation: Dict) -> Dict:
    """
    Enriquece el idle_index original con la validación visual dual-source.

    Agrega campos:
      - verified_idle_s:       ocio confirmado por IMU+visual
      - verified_iti_pct:      ITI validado (más bajo y confiable)
      - imu_only_idle_s:       IMU marca ocio pero cámara se mueve (sospechoso)
      - visual_only_idle_s:    cámara quieta pero IMU no marca wait
      - agreement_pct:         % de acuerdo entre fuentes
      - confidence:            HIGH | MEDIUM | LOW
      - productivity_alerts:   alertas para operador
      - disagreements:         lista para auditoría
    """
    if not idle_index or not validation:
        return idle_index or {}

    verified = validation.get('verified_idle', {})

    enhanced = dict(idle_index)
    enhanced.update({
        'verified_idle_s':      verified.get('verified_idle_s', 0.0),
        'verified_active_s':    verified.get('verified_active_s', 0.0),
        'imu_only_idle_s':      verified.get('imu_only_idle_s', 0.0),
        'visual_only_idle_s':   verified.get('visual_only_idle_s', 0.0),
        'verified_iti':         verified.get('verified_iti', 0.0),
        'verified_iti_pct':     verified.get('verified_iti_pct', 0.0),
        'agreement_pct':        verified.get('agreement_pct', 0.0),
        'confidence':           verified.get('confidence', 'LOW'),
        'confidence_score':     verified.get('confidence_score', 0.0),
        'n_idle_periods_visual': verified.get('n_idle_periods_visual', 0),
        'productivity_alerts':  validation.get('productivity_alerts', []),
        'wait_confidence':      validation.get('wait_confidence', []),
        'disagreements':        validation.get('disagreements', []),
        'disagreements_n':      len(validation.get('disagreements', [])),
    })
    return enhanced
