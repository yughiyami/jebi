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
    fuse_imu_motion,
    extract_idle_segments_from_fusion,
)
from state_classifier import (
    classify_idle,
    detect_contraproductive,
    STATES as OPERATOR_STATES,
)


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def run_visual_validation(video_path: str,
                           duration_s: float,
                           wait_events: List,
                           cycles: List,
                           interval_s: float = 1.0,
                           df_imu=None) -> Dict:
    """
    Pipeline completo de validación visual con optical flow.

    Si se provee `df_imu`, también corre la FUSIÓN estricta al estilo jevi:
    detecta idle cuando IMU y cámara concuerdan, clasifica en 3 estados
    (justificada / injustificada / contraproducente).

    Args:
        video_path:   path al video (left o right)
        duration_s:   duración de la sesión (segundos)
        wait_events:  list[WaitEvent] del IMU
        cycles:       list[LoadCycle] del IMU
        interval_s:   cada cuánto muestrear (1.0s default)
        df_imu:       DataFrame del IMU (opcional, habilita fusion)

    Returns:
        Dict serializable con:
          - motion_timeline:       serie temporal de motion
          - verified_idle:         métricas de idle cruzado IMU+visual
          - fusion_events:         eventos ops (pausas clasificadas)  ← NUEVO
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

    # 4. CLASIFICACIÓN de idle periods detectados por optical flow ─────────────
    # Usa los idle_periods del motion_tl (ya validados contra el video real,
    # detectaron los 2:40 y 4:54 que el operador confirmó). Para cada uno,
    # clasificamos usando el state_classifier (justificada vs injustificada).
    # Además detectamos actividad contraproductive en los segmentos activos.
    fusion_events: List[Dict] = []
    total_inactivo_s = 0.0
    if df_imu is not None:
        try:
            print('  [validator] Clasificando idle periods visuales + contraprod...')
            # Preparar df_imu con features para el classifier
            df_fused = fuse_imu_motion(df_imu, motion_tl)

            # Los idle periods reales = los del motion_tl (visualmente confirmados)
            idle_segs = [
                {
                    'tiempo_inicio_s': p.t_start,
                    'tiempo_fin_s':    p.t_end,
                    'duracion_s':      p.duration_s,
                }
                for p in motion_tl.idle_periods
            ]

            # Clasificar cada idle segment en JUSTIFICADA / INJUSTIFICADA
            classified_idles = [
                {**s, 'estado': classify_idle(s, df_fused)}
                for s in idle_segs
            ]
            # Detectar períodos CONTRAPRODUCENTES (activo sin periodicidad)
            cp_events = detect_contraproductive(df_fused, idle_segs)

            # Merge y ordenar por timestamp
            fusion_events = sorted(
                classified_idles + cp_events,
                key=lambda e: e['tiempo_inicio_s'],
            )

            total_inactivo_s = sum(e['duracion_s'] for e in classified_idles)

            # Stats por categoría
            n_just = sum(1 for e in classified_idles
                         if e['estado'] == 'INACTIVIDAD_JUSTIFICADA')
            n_injust = sum(1 for e in classified_idles
                           if e['estado'] == 'INACTIVIDAD_INJUSTIFICADA')
            n_cp = len(cp_events)
            print(f'  [validator] Clasificación: {len(classified_idles)} idles '
                  f'(⚠{n_injust} injust + ⏸{n_just} just) + ↯{n_cp} contraprod.')
            print(f'  [validator] Tiempo inactivo total (verificado): {total_inactivo_s:.1f}s')
        except Exception as e:
            print(f'  [validator] Clasificación falló: {e}')
            import traceback; traceback.print_exc()
            fusion_events = []

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
        'fusion_events':        fusion_events,
        'total_inactivo_s':     round(total_inactivo_s, 1),
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
        'fusion_events':       [],
        'total_inactivo_s':    0.0,
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

    CAMBIO MAYOR: ahora la FUENTE DE VERDAD son los idle_periods del optical flow,
    no los wait_events del IMU. El IMU tiene falsos positivos que inflaban el
    ITI (~76%) cuando el ocio real es mucho menor (~3%).

    El `iti_pct` principal se RECALCULA usando solo los idle_periods visuales.
    El valor original del IMU se guarda como `iti_pct_imu_raw` para referencia.

    Agrega campos:
      - iti_pct:            ITI REAL (idle_periods visuales / duration) ← MÉTRICA PRINCIPAL
      - iti_pct_imu_raw:    ITI que el IMU reportaba (con falsos positivos)
      - real_idle_periods:  lista de los períodos idle confirmados (los 2 reales)
      - real_idle_s:        suma de duraciones de períodos reales
      - false_positives_s:  waits del IMU que el video NO confirma (tiempo)
      - false_positives_n:  cantidad de waits que son falsos positivos
      - agreement_pct:      % de acuerdo entre fuentes
      - confidence:         HIGH | MEDIUM | LOW
      - productivity_alerts: alertas para operador
    """
    if not idle_index or not validation:
        return idle_index or {}

    verified = validation.get('verified_idle', {})
    motion_tl = validation.get('motion_timeline', {})

    # ── Los períodos idle REALES son los del optical flow (visual confirmado) ─
    real_periods = motion_tl.get('idle_periods', [])
    real_idle_s = sum(p.get('duration_s', 0) for p in real_periods)

    duration_s = idle_index.get('total_duration_s', 0)
    iti_real = real_idle_s / duration_s if duration_s > 0 else 0.0

    # ── Recalcular banda para el ITI REAL (mucho más bajo que el IMU crudo) ──
    # Bandas del IDLE_BANDS de metrics.py
    if iti_real < 0.15:
        band, band_color, rec = 'OPTIMO', 'green', 'Operacion muy eficiente. Los tiempos de ocio son minimos.'
    elif iti_real < 0.30:
        band, band_color, rec = 'NORMAL', 'blue', 'Ocio dentro de rango esperado.'
    elif iti_real < 0.50:
        band, band_color, rec = 'CRITICO', 'yellow', 'Mucho tiempo sin producir. Revisar coordinacion.'
    else:
        band, band_color, rec = 'COMPROMETIDO', 'red', 'Intervencion urgente.'

    # ── Calcular cuántos waits del IMU son FALSOS POSITIVOS ───────────────────
    # Un wait del IMU es "real" si hay OVERLAP con algún idle_period visual
    wait_confs = validation.get('wait_confidence', [])
    false_pos_n = 0
    false_pos_s = 0.0
    real_waits = []
    for wc in wait_confs:
        # Verificar si este wait se solapa con algún idle_period visual
        overlaps = False
        for p in real_periods:
            # Overlap si hay intersección
            if wc['t_start'] <= p.get('t_end', 0) and wc['t_end'] >= p.get('t_start', 0):
                overlaps = True
                break
        if overlaps:
            real_waits.append(wc)
        else:
            false_pos_n += 1
            false_pos_s += wc.get('duration_s', 0)

    # ── Guardar el ITI del IMU crudo con otro nombre para referencia ──────────
    iti_pct_imu_raw = idle_index.get('iti_pct', 0.0)
    total_idle_imu_raw = idle_index.get('total_idle_s', 0.0)

    enhanced = dict(idle_index)
    enhanced.update({
        # ── Métricas PRINCIPALES ahora son las visuales (fuente de verdad) ───
        'iti':              round(iti_real, 3),
        'iti_pct':          round(iti_real * 100, 1),
        'total_idle_s':     round(real_idle_s, 1),
        'band':             band,
        'band_color':       band_color,
        'recommendation':   rec,
        'productive_s':     round(max(0.0, duration_s - real_idle_s), 1),
        'ratio_prod_idle':  round((duration_s - real_idle_s) / real_idle_s, 2) if real_idle_s > 0 else -1,
        'n_events':         len(real_periods),
        # ── Datos del IMU original (para comparación y transparencia) ────────
        'iti_pct_imu_raw':   round(iti_pct_imu_raw, 1),
        'total_idle_imu_raw': round(total_idle_imu_raw, 1),
        # ── Falsos positivos detectados ──────────────────────────────────────
        'false_positives_n':  false_pos_n,
        'false_positives_s':  round(false_pos_s, 1),
        'real_idle_periods':  real_periods,
        'real_waits_n':       len(real_waits),
        # ── Métricas dual-source adicionales ─────────────────────────────────
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
        # Reemplazar top_waits con los períodos REALES (2 en nuestro caso)
        'top_waits':            [
            {
                't_start':   p['t_start'],
                't_end':     p['t_end'],
                'duration':  p['duration_s'],
                'reason':    'verified_idle',
                'label':     f'Ocio real #{i+1} (confirmado por video)',
                'pct_total': round(p['duration_s'] / duration_s * 100, 2) if duration_s > 0 else 0,
            }
            for i, p in enumerate(sorted(real_periods, key=lambda x: -x.get('duration_s', 0)))
        ],
    })
    return enhanced
