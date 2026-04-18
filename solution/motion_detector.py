# -*- coding: utf-8 -*-
"""
motion_detector.py — JEBI Hackathon 2026

Detector de movimiento de cámara por OPTICAL FLOW.

INSIGHT CLAVE:
  La cámara está montada sobre la pala. Por lo tanto:
    → Cámara se mueve = Pala se mueve
    → Cámara quieta   = Pala quieta (IDLE real)

Esto da un SENSOR VISUAL independiente del IMU que podemos cruzar
para validar idle/productivo con muy alta confianza.

Filosofía:
  - CV clásico (Farneback), NO machine learning
  - Fast, robusto, sin entrenamiento
  - Downscale agresivo para velocidad (~10-20ms por frame)
  - Frame-a-frame sobre TODO el video

Entrega (por timestamp):
  motion_magnitude:  float 0-inf (magnitud promedio del flujo)
  motion_score:      float 0-1   (normalizado)
  is_static:         bool        (motion debajo del threshold)
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

# Resolución interna para calcular flujo (downscale agresivo → rapidez)
FLOW_WIDTH  = 320
FLOW_HEIGHT = 180

# Muestreo temporal
DEFAULT_SAMPLE_INTERVAL_S = 1.0   # 1 sample/seg da buen balance

# Thresholds de motion
# CRÍTICO: son ADAPTIVOS al video porque cada escenario tiene su baseline.
# El threshold real se calcula como percentil sobre el propio video.
# Esto garantiza que funcione en cualquier dataset (brief: test dataset diferente).
STATIC_PERCENTILE       = 25     # motion < percentil 25 del video → considerado static
LOW_PERCENTILE          = 50     # motion < percentil 50 → low activity
# Fallback si el percentil da valores degenerados
MIN_STATIC_THRESHOLD    = 0.15   # mínimo absoluto
MAX_STATIC_THRESHOLD    = 3.0    # máximo absoluto

# Parámetros Farneback (tuned for mining shovel scenarios)
FARNEBACK_PARAMS = dict(
    pyr_scale = 0.5,
    levels    = 3,
    winsize   = 15,
    iterations = 2,      # 3 es más preciso pero 2 basta y es 30% más rápido
    poly_n    = 5,
    poly_sigma = 1.2,
    flags     = 0,
)

# Idle detection
MIN_IDLE_STREAK_S = 10.0  # segundos contiguos de static → evento idle


# ─── DATACLASSES ──────────────────────────────────────────────────────────────

@dataclass
class MotionSample:
    """Un punto en el timeline de motion."""
    timestamp_s:       float
    motion_magnitude:  float   # magnitud raw del optical flow
    motion_score:      float   # normalizado 0-1
    is_static:         bool    # below STATIC_MOTION_THRESHOLD


@dataclass
class IdlePeriod:
    """Un periodo continuo donde la cámara estuvo quieta."""
    t_start:      float
    t_end:        float
    duration_s:   float
    avg_motion:   float      # motion promedio en el periodo


@dataclass
class MotionTimeline:
    """Timeline completo de motion."""
    samples:              List[MotionSample]
    sample_interval_s:    float
    idle_periods:         List[IdlePeriod]
    total_static_s:       float     # tiempo total con motion < threshold
    total_motion_s:       float     # tiempo total con motion >= threshold
    avg_motion_magnitude: float     # promedio global
    max_motion_magnitude: float     # pico


# ─── EXTRACCIÓN OPTICAL FLOW ──────────────────────────────────────────────────

def _prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Convierte BGR a gray y downscale para velocidad."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    if gray.shape[1] != FLOW_WIDTH or gray.shape[0] != FLOW_HEIGHT:
        gray = cv2.resize(gray, (FLOW_WIDTH, FLOW_HEIGHT),
                          interpolation=cv2.INTER_AREA)
    return gray


def _motion_between(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Calcula la magnitud promedio del optical flow entre dos frames.
    Retorna float (píxeles de desplazamiento promedio).
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))


def compute_motion_timeline(video_path: str,
                             duration_s: float,
                             interval_s: float = DEFAULT_SAMPLE_INTERVAL_S) -> MotionTimeline:
    """
    Procesa el video completo calculando optical flow cada `interval_s` segundos.

    Args:
        video_path: path al video (.mp4)
        duration_s: duración total de la sesión
        interval_s: cada cuánto samplear (1s es buen balance)

    Returns:
        MotionTimeline con samples y periodos idle detectados.
    """
    if not os.path.exists(video_path):
        print(f'  [motion] Video no encontrado: {video_path}')
        return _empty_timeline(interval_s)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    n_samples = int(duration_s / interval_s) + 1
    print(f'  [motion] Procesando optical flow: {n_samples} samples cada {interval_s}s')

    samples: List[MotionSample] = []
    prev_gray: Optional[np.ndarray] = None
    magnitudes: List[float] = []

    for i in range(n_samples):
        ts = i * interval_s
        if ts > duration_s:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = _prepare_frame(frame)

        if prev_gray is None:
            # Primer frame: no hay referencia
            magnitude = 0.0
        else:
            try:
                magnitude = _motion_between(prev_gray, gray)
            except Exception as e:
                print(f'  [motion] Error en t={ts:.1f}s: {e}')
                magnitude = 0.0

        magnitudes.append(magnitude)
        samples.append(MotionSample(
            timestamp_s      = round(ts, 2),
            motion_magnitude = round(magnitude, 3),
            motion_score     = 0.0,   # se rellena abajo tras normalización
            is_static        = False,  # placeholder, se calcula con threshold adaptivo
        ))

        prev_gray = gray

        if (i + 1) % 60 == 0 or (i + 1) == n_samples:
            print(f'    [motion] {i+1}/{n_samples} frames — motion avg={np.mean(magnitudes):.2f}')

    cap.release()

    if not samples:
        return _empty_timeline(interval_s)

    # ── THRESHOLD ADAPTIVO ────────────────────────────────────────────────────
    # Calculamos static_threshold desde la DISTRIBUCIÓN del video actual.
    # Así funciona en cualquier escenario (pala vibrando, cámara temblorosa,
    # etc.) — el "static" se define RELATIVO al video, no con un valor fijo.
    mags = np.array([s.motion_magnitude for s in samples])
    # Ignoramos el primer sample (es 0 porque no tiene referencia)
    mags_nonzero = mags[1:] if len(mags) > 1 else mags
    # Percentil 25 = los samples más quietos del video
    p25 = float(np.percentile(mags_nonzero, STATIC_PERCENTILE))
    # Añadimos un pequeño margen (30%) para capturar "casi-quieto"
    static_threshold = p25 * 1.3
    # Clampeamos a rango razonable
    static_threshold = max(MIN_STATIC_THRESHOLD, min(MAX_STATIC_THRESHOLD, static_threshold))
    print(f'  [motion] Threshold adaptivo: {static_threshold:.3f} (p25={p25:.3f}, avg={np.mean(mags_nonzero):.3f})')

    # Clasificar samples con el threshold adaptivo
    for s in samples:
        s.is_static = s.motion_magnitude < static_threshold

    # Normalización a 0-1 (min-max con clipping al percentil 95)
    mag_max = float(np.percentile(mags, 95)) if len(mags) > 0 else 1.0
    mag_max = max(mag_max, 0.1)  # evitar división por 0
    for s in samples:
        s.motion_score = round(min(1.0, s.motion_magnitude / mag_max), 3)

    # Detectar períodos idle contiguos
    idle_periods = _detect_idle_periods(samples, interval_s, MIN_IDLE_STREAK_S)

    total_static_s = sum(interval_s for s in samples if s.is_static)
    total_motion_s = sum(interval_s for s in samples if not s.is_static)

    return MotionTimeline(
        samples              = samples,
        sample_interval_s    = interval_s,
        idle_periods         = idle_periods,
        total_static_s       = round(total_static_s, 1),
        total_motion_s       = round(total_motion_s, 1),
        avg_motion_magnitude = round(float(np.mean(mags)), 3),
        max_motion_magnitude = round(float(np.max(mags)), 3),
    )


def _detect_idle_periods(samples: List[MotionSample],
                          interval_s: float,
                          min_streak_s: float) -> List[IdlePeriod]:
    """
    Encuentra runs contiguos de samples static y los agrupa como períodos idle.
    Solo los períodos >= min_streak_s califican.
    """
    periods: List[IdlePeriod] = []
    if not samples:
        return periods

    run_start: Optional[int] = None
    run_mags: List[float] = []

    for i, s in enumerate(samples):
        if s.is_static:
            if run_start is None:
                run_start = i
                run_mags = [s.motion_magnitude]
            else:
                run_mags.append(s.motion_magnitude)
        else:
            if run_start is not None:
                t_start = samples[run_start].timestamp_s
                t_end   = samples[i - 1].timestamp_s + interval_s
                dur     = t_end - t_start
                if dur >= min_streak_s:
                    periods.append(IdlePeriod(
                        t_start    = round(t_start, 1),
                        t_end      = round(t_end, 1),
                        duration_s = round(dur, 1),
                        avg_motion = round(float(np.mean(run_mags)), 3),
                    ))
                run_start = None
                run_mags = []

    # Caso run abierto al final
    if run_start is not None:
        t_start = samples[run_start].timestamp_s
        t_end   = samples[-1].timestamp_s + interval_s
        dur     = t_end - t_start
        if dur >= min_streak_s:
            periods.append(IdlePeriod(
                t_start    = round(t_start, 1),
                t_end      = round(t_end, 1),
                duration_s = round(dur, 1),
                avg_motion = round(float(np.mean(run_mags)), 3),
            ))

    return periods


def _empty_timeline(interval_s: float) -> MotionTimeline:
    return MotionTimeline(
        samples=[], sample_interval_s=interval_s,
        idle_periods=[], total_static_s=0.0, total_motion_s=0.0,
        avg_motion_magnitude=0.0, max_motion_magnitude=0.0,
    )


# ─── CRUCE CON IMU ────────────────────────────────────────────────────────────

def motion_at_timestamp(motion_tl: MotionTimeline,
                         t: float,
                         window_s: float = 2.0) -> Tuple[float, bool]:
    """
    Devuelve (motion_score promedio, is_static) alrededor de t.
    """
    if not motion_tl.samples:
        return 0.0, False

    window = [s for s in motion_tl.samples
              if abs(s.timestamp_s - t) <= window_s]
    if not window:
        nearest = min(motion_tl.samples, key=lambda s: abs(s.timestamp_s - t))
        return nearest.motion_score, nearest.is_static

    avg_score = float(np.mean([s.motion_score for s in window]))
    # Static si la mayoría del window es static
    majority_static = sum(1 for s in window if s.is_static) > len(window) / 2
    return round(avg_score, 3), majority_static


def cross_validate_waits(wait_events: List,
                          motion_tl: MotionTimeline) -> List[Dict]:
    """
    Para cada wait_event del IMU, cruza con motion timeline.

    Reglas:
      IMU wait + motion static → HIGH confidence idle (confirmed)
      IMU wait + motion moving → LOW confidence (cámara movida pero IMU marca wait)

    Returns: list[{t_start, t_end, duration, reason, visual_static, confidence, score}]
    """
    out = []
    for w in wait_events:
        t_mid = (w.t_start + w.t_end) / 2
        motion_score, is_static = motion_at_timestamp(motion_tl, t_mid, window_s=3.0)

        if is_static:
            conf = 'HIGH'
            cval = 0.95
            validated = True
        elif motion_score < 0.3:
            conf = 'MEDIUM'
            cval = 0.7
            validated = True
        else:
            conf = 'LOW'
            cval = 0.3
            validated = False

        out.append({
            't_start':       round(w.t_start, 1),
            't_end':         round(w.t_end, 1),
            'duration_s':    round(w.duration_s, 1),
            'reason':        w.reason,
            'motion_score':  motion_score,
            'visual_static': is_static,
            'confidence':    conf,
            'conf_score':    cval,
            'validated':     validated,
        })

    return out


def cross_validate_cycles(cycles: List,
                           motion_tl: MotionTimeline) -> List[Dict]:
    """
    Para cada ciclo del IMU, confirma actividad visual.
    """
    out = []
    for c in cycles:
        if c.is_mini_cycle:
            continue
        t_mid = (c.t_start + c.t_end) / 2
        motion_score, is_static = motion_at_timestamp(motion_tl, t_mid, window_s=3.0)

        if not is_static and motion_score > 0.3:
            conf = 'HIGH'
        elif motion_score > 0.1:
            conf = 'MEDIUM'
        else:
            conf = 'LOW'

        out.append({
            'cycle_id':      c.cycle_id,
            't_start':       round(c.t_start, 1),
            't_end':         round(c.t_end, 1),
            'motion_score':  motion_score,
            'visual_static': is_static,
            'confidence':    conf,
        })

    return out


# ─── MÉTRICA: VERIFIED IDLE ───────────────────────────────────────────────────

def compute_verified_idle(wait_events: List,
                           motion_tl: MotionTimeline,
                           duration_s: float) -> Dict:
    """
    Calcula el IDLE VERIFICADO (ambas fuentes confirman ocio).

    Es la métrica más robusta para productividad real:
      verified_idle_s = waits donde IMU dice ocio Y cámara confirma quietud

    También detecta casos de DESACUERDO para auditoría.

    Returns:
      {
        verified_idle_s:        tiempo confirmado como idle por ambas fuentes
        verified_active_s:      tiempo activo confirmado
        imu_only_idle_s:        IMU dice idle pero cámara moviéndose (disputado)
        visual_only_idle_s:     cámara estática pero IMU sin wait (posible falso positivo IMU)
        verified_iti:           idle verificado / duración total
        agreement_pct:          % de tiempo donde ambas fuentes concuerdan
        confidence:             HIGH | MEDIUM | LOW global
      }
    """
    if not motion_tl.samples:
        return {
            'verified_idle_s':    0.0,
            'verified_active_s':  0.0,
            'imu_only_idle_s':    sum(w.duration_s for w in wait_events),
            'visual_only_idle_s': 0.0,
            'verified_iti':       0.0,
            'verified_iti_pct':   0.0,
            'agreement_pct':      0.0,
            'confidence':         'LOW',
            'confidence_score':   0.0,
            'n_idle_periods_visual': 0,
        }

    # Helper: ¿el timestamp t está dentro de algún wait_event del IMU?
    def is_in_wait(t):
        for w in wait_events:
            if w.t_start <= t <= w.t_end:
                return True
        return False

    interval = motion_tl.sample_interval_s
    verified_idle = 0.0
    verified_active = 0.0
    imu_only_idle = 0.0
    visual_only_idle = 0.0
    agree = 0

    for s in motion_tl.samples:
        in_wait = is_in_wait(s.timestamp_s)
        if in_wait and s.is_static:
            verified_idle += interval
            agree += 1
        elif not in_wait and not s.is_static:
            verified_active += interval
            agree += 1
        elif in_wait and not s.is_static:
            imu_only_idle += interval
        elif not in_wait and s.is_static:
            visual_only_idle += interval

    n_samples = len(motion_tl.samples)
    agreement_pct = (agree / n_samples * 100) if n_samples > 0 else 0.0

    verified_iti = verified_idle / duration_s if duration_s > 0 else 0.0

    # Confidence global
    if agreement_pct >= 85:
        conf = 'HIGH'
    elif agreement_pct >= 65:
        conf = 'MEDIUM'
    else:
        conf = 'LOW'

    return {
        'verified_idle_s':     round(verified_idle, 1),
        'verified_active_s':   round(verified_active, 1),
        'imu_only_idle_s':     round(imu_only_idle, 1),
        'visual_only_idle_s':  round(visual_only_idle, 1),
        'verified_iti':        round(verified_iti, 3),
        'verified_iti_pct':    round(verified_iti * 100, 1),
        'agreement_pct':       round(agreement_pct, 1),
        'confidence':          conf,
        'confidence_score':    round(agreement_pct / 100, 3),
        'n_idle_periods_visual': len(motion_tl.idle_periods),
    }


# ─── ALERTAS DE PÉRDIDA PRODUCTIVA ────────────────────────────────────────────

@dataclass
class ProductivityLossAlert:
    """Alerta específica: no hay movimiento → pérdida productiva."""
    t_start:       float
    t_end:         float
    duration_s:    float
    severity:      str        # 'critical' | 'warning' | 'info'
    estimated_loss_t: float   # toneladas estimadas no movidas


def compute_productivity_loss_alerts(idle_periods: List[IdlePeriod],
                                       productivity_tph: float = 400.0) -> List[Dict]:
    """
    Genera alertas por cada período idle visual significativo.
    El operador ve: "Esto te costó ~X toneladas no movidas"
    """
    alerts = []
    for p in idle_periods:
        if p.duration_s < 15:
            continue   # ignorar muy cortos
        if p.duration_s >= 60:
            sev = 'critical'
        elif p.duration_s >= 30:
            sev = 'warning'
        else:
            sev = 'info'

        loss_t = (p.duration_s / 3600) * productivity_tph

        alerts.append({
            't_start':          p.t_start,
            't_end':            p.t_end,
            'duration_s':       p.duration_s,
            'severity':         sev,
            'estimated_loss_t': round(loss_t, 2),
            'type':             'PRODUCTIVITY_LOSS',
            'message':          f'Sin movimiento de cámara {p.duration_s:.0f}s — pérdida ~{loss_t:.1f}t estimadas',
        })
    return alerts


# ─── SERIALIZACIÓN ────────────────────────────────────────────────────────────

def motion_timeline_to_dict(mt: MotionTimeline) -> Dict:
    """Serializa para outputs JSON (sin todos los samples, para tamaño)."""
    return {
        'n_samples':            len(mt.samples),
        'sample_interval_s':    mt.sample_interval_s,
        'total_static_s':       mt.total_static_s,
        'total_motion_s':       mt.total_motion_s,
        'avg_motion_magnitude': mt.avg_motion_magnitude,
        'max_motion_magnitude': mt.max_motion_magnitude,
        'idle_periods':         [asdict(p) for p in mt.idle_periods],
        # Serie temporal (liviana)
        'series': [
            {'t': s.timestamp_s, 'score': s.motion_score, 'static': s.is_static}
            for s in mt.samples
        ],
    }
