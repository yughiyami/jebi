# -*- coding: utf-8 -*-
"""
metrics.py
Calcula todas las eficiencias operacionales del ciclo de carga minera:
  - Recoleccion      → fill factor del bucket
  - Volumen          → m³ por ciclo
  - Maniobra         → tiempo productivo vs total
  - Carga camion     → toneladas acumuladas vs capacidad
  - Descarga         → calidad del dump
  - Colocacion       → velocidad de posicionamiento
  - Transporte       → espera entre camiones / despacho
  - Ciclo total      → OEE minero simplificado
  
  + Sistema de alertas en tiempo real indexado por timestamp
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Specs del equipo
BUCKET_M3       = 27.0
MATERIAL_T_M3   = 2.5
BUCKET_MAX_T    = BUCKET_M3 * MATERIAL_T_M3      # 67.5 t
CAP_793F        = 218.0   # CAT 793F  (nota: brief dice 218t)
CAP_EH4000      = 221.0   # EH4000 AC-3

# Umbrales de alerta
UNDERFILL_THRESH      = 0.80   # fill < 80% → alerta
OVERFILL_THRESH       = 0.98   # fill > 98% → esfuerzo excesivo
HARD_IMPACT_THRESH    = 22.0   # m/s² accel durante dig → impacto duro
ROUGH_DUMP_THRESH     = 18.0   # m/s² durante dump → volcado brusco
LONG_WAIT_THRESH      = 25.0   # segundos → camion no esta listo
MINI_CYCLE_DUR_THRESH = 4.0    # segundos → ciclo muy corto
BAD_POSITION_THRESH   = 3.0    # segundos para alcanzar pico gyro → lento
DISPATCH_GAP_THRESH   = 45.0   # segundos entre camiones → despacho lento


# ─── ALERTAS ─────────────────────────────────────────────────────────────────

ALERT_SEVERITY = {
    'UNDERFILL':        'warning',
    'OVERFILL':         'warning',
    'HARD_IMPACT':      'critical',
    'ROUGH_DUMP':       'warning',
    'LONG_WAIT':        'info',
    'MINI_CYCLE':       'info',
    'BAD_POSITIONING':  'warning',
    'ROUGH_DIG':        'warning',
    'DISPATCH_DELAY':   'info',
    'TRUCK_FULL':       'success',
    'TRUCK_UNDERFULL':  'warning',
    'POSSIBLE_SPILL':   'warning',
}

ALERT_MESSAGES = {
    'UNDERFILL':       'Fill < 80%: considerar re-mordida antes del swing',
    'OVERFILL':        'Posible sobrecarga del bucket (>98%)',
    'HARD_IMPACT':     'Impacto duro en dig: aceleracion anomala detectada',
    'ROUGH_DUMP':      'Volcado brusco: puede causar derrame de material',
    'LONG_WAIT':       'Espera larga: camion no esta en posicion',
    'MINI_CYCLE':      'Mini-ciclo detectado: movimiento de correccion',
    'BAD_POSITIONING': 'Posicionamiento lento: demora en alcanzar angulo optimo',
    'ROUGH_DIG':       'Dig irregular: vibracion alta durante excavacion',
    'DISPATCH_DELAY':  'Retraso en despacho: brecha larga entre camiones',
    'TRUCK_FULL':      'Camion completado: capacidad alcanzada',
    'TRUCK_UNDERFULL': 'Camion sale con capacidad incompleta',
    'POSSIBLE_SPILL':  'Posible derrame: aceleracion alta durante swing cargado',
}


@dataclass
class Alert:
    timestamp_s:  float
    cycle_id:     int
    alert_type:   str
    severity:     str    # 'critical' | 'warning' | 'info' | 'success'
    message:      str
    value:        float  # valor que disparo la alerta
    unit:         str


@dataclass
class EfficiencyProfile:
    """Perfil de eficiencia completo para un ciclo."""
    cycle_id:              int
    t_start:               float
    t_end:                 float

    # 1. Recoleccion
    collection_eff_pct:    float   # fill factor %
    volume_m3:             float   # m³ recogidos

    # 2. Maniobra
    maneuver_eff_pct:      float   # tiempo swing / tiempo total ciclo
    swing_time_s:          float
    idle_time_s:           float

    # 3. Posicionamiento
    positioning_eff_pct:   float   # rapidez de alcanzar peak gyro
    time_to_peak_s:        float

    # 4. Descarga
    discharge_eff_pct:     float   # calidad del dump
    dump_smoothness:       float   # 0-1 (1=muy suave)

    # 5. Ciclo total
    cycle_eff_pct:         float   # OEE-like combinado

    # 6. Alertas del ciclo
    alerts:                List[Alert] = field(default_factory=list)


@dataclass
class TruckLoadEvent:
    """Registro de carga por camion."""
    truck_id:         str
    model:            str
    capacity_t:       float
    t_first_pass_s:   float
    t_last_pass_s:    float
    n_passes:         int
    total_payload_t:  float
    fill_pct:         float    # vs capacidad del camion
    dispatched:       bool     # True si alcanzo capacidad y salio


@dataclass
class TransportMetrics:
    """Metricas de despacho y transporte."""
    n_trucks_served:      int
    n_trucks_full:        int
    n_trucks_partial:     int
    avg_passes_per_truck: float
    avg_truck_fill_pct:   float
    total_dispatch_t:     float
    avg_dispatch_gap_s:   float
    max_dispatch_gap_s:   float
    dispatch_efficiency:  float  # trucks_full / trucks_served * 100


# ─── COMPUTACION DE EFICIENCIAS ──────────────────────────────────────────────

def compute_efficiency_profiles(
    cycles,          # List[LoadCycle]
    wait_events,     # List[WaitEvent]
    df_imu: pd.DataFrame,
) -> Tuple[List[EfficiencyProfile], List[Alert]]:
    """
    Calcula todos los perfiles de eficiencia y genera alertas.
    """
    t   = df_imu.timestamp_s.values
    gx  = df_imu.gx.values
    gn  = df_imu.gyro_norm.values
    an  = df_imu.accel_norm.values
    az  = df_imu.az.values
    fs  = df_imu.attrs['fs']

    # Estadisticas globales para normalizacion
    full_cycles = [c for c in cycles if not c.is_mini_cycle]
    if full_cycles:
        median_cycle_dur = np.median([c.t_end - c.t_start for c in full_cycles])
        median_peak_gyro = np.median([c.swing_loaded.peak_gyro
                                       for c in full_cycles if c.swing_loaded])
    else:
        median_cycle_dur = 30.0
        median_peak_gyro = 60.0

    profiles = []
    all_alerts: List[Alert] = []

    for c in cycles:
        i0 = max(0, np.searchsorted(t, c.t_start) - 1)
        i1 = min(len(t) - 1, np.searchsorted(t, c.t_end) + 1)
        seg_t  = t[i0:i1]
        seg_gn = gn[i0:i1]
        seg_an = an[i0:i1]
        seg_az = az[i0:i1]
        seg_gx = gx[i0:i1]

        cycle_dur = c.t_end - c.t_start
        alerts: List[Alert] = []

        # ── 1. Recoleccion ────────────────────────────────────────────────────
        coll_eff = c.fill_factor * 100
        volume   = c.fill_factor * BUCKET_M3

        # ── 2. Maniobra ───────────────────────────────────────────────────────
        swing_time = 0.0
        if c.swing_loaded:
            swing_time += c.swing_loaded.duration_s
        if c.swing_empty:
            swing_time += c.swing_empty.duration_s
        idle_time   = max(0.0, cycle_dur - swing_time)
        maneuver_eff = min(100.0, (swing_time / max(cycle_dur, 1.0)) * 100 * 1.6)

        # ── 3. Posicionamiento ────────────────────────────────────────────────
        # Tiempo desde inicio del ciclo hasta que gyro alcanza 50% del pico
        time_to_peak = _time_to_peak_fraction(seg_t, seg_gn, fraction=0.5)
        # Normalizar: < 1.5s = perfecto, > 4s = lento
        pos_eff = max(10.0, min(100.0, (1.0 - (time_to_peak - 1.0) / 4.0) * 100))

        # ── 4. Descarga ───────────────────────────────────────────────────────
        # Calidad del dump: spike de accel suave vs brusco
        dump_smooth, dump_eff = _compute_dump_quality(
            seg_t, seg_az, seg_an, c.t_dump
        )

        # ── 5. OEE simplificado ───────────────────────────────────────────────
        cycle_eff = (coll_eff * 0.4 + maneuver_eff * 0.3 +
                     pos_eff * 0.15 + dump_eff * 0.15)

        # ── ALERTAS POR CICLO ─────────────────────────────────────────────────

        def add_alert(atype, val, unit):
            a = Alert(
                timestamp_s = c.t_dump or c.t_start,
                cycle_id    = c.cycle_id,
                alert_type  = atype,
                severity    = ALERT_SEVERITY[atype],
                message     = ALERT_MESSAGES[atype],
                value       = round(val, 2),
                unit        = unit,
            )
            alerts.append(a)
            all_alerts.append(a)

        if c.fill_factor < UNDERFILL_THRESH and not c.is_mini_cycle:
            add_alert('UNDERFILL', c.fill_factor * 100, '%')

        if c.fill_factor > OVERFILL_THRESH:
            add_alert('OVERFILL', c.fill_factor * 100, '%')

        if c.is_mini_cycle:
            add_alert('MINI_CYCLE', cycle_dur, 's')

        # Impacto duro durante dig (inicio del ciclo)
        dig_region = seg_an[:max(1, len(seg_an) // 3)]
        if len(dig_region) > 0 and np.max(dig_region) > HARD_IMPACT_THRESH:
            add_alert('HARD_IMPACT', float(np.max(dig_region)), 'm/s2')

        # Volcado brusco
        if c.t_dump:
            dump_idx = np.searchsorted(seg_t, c.t_dump)
            dump_window = seg_an[max(0, dump_idx-5):dump_idx+10]
            if len(dump_window) > 0 and np.max(dump_window) > ROUGH_DUMP_THRESH:
                add_alert('ROUGH_DUMP', float(np.max(dump_window)), 'm/s2')

        # Posicionamiento lento
        if time_to_peak > BAD_POSITION_THRESH * 1.5:
            add_alert('BAD_POSITIONING', time_to_peak, 's')

        # Posible derrame: alta accel durante swing cargado
        if c.swing_loaded:
            s_i0 = np.searchsorted(t, c.swing_loaded.t_start)
            s_i1 = np.searchsorted(t, c.swing_loaded.t_end)
            swing_an = an[s_i0:s_i1]
            if len(swing_an) > 0 and np.max(swing_an) > 18.0:
                add_alert('POSSIBLE_SPILL', float(np.max(swing_an)), 'm/s2')

        profiles.append(EfficiencyProfile(
            cycle_id             = c.cycle_id,
            t_start              = c.t_start,
            t_end                = c.t_end,
            collection_eff_pct   = round(coll_eff, 1),
            volume_m3            = round(volume, 1),
            maneuver_eff_pct     = round(maneuver_eff, 1),
            swing_time_s         = round(swing_time, 1),
            idle_time_s          = round(idle_time, 1),
            positioning_eff_pct  = round(pos_eff, 1),
            time_to_peak_s       = round(time_to_peak, 2),
            discharge_eff_pct    = round(dump_eff, 1),
            dump_smoothness      = round(dump_smooth, 2),
            cycle_eff_pct        = round(cycle_eff, 1),
            alerts               = alerts,
        ))

    # Alertas de wait events
    for w in wait_events:
        if w.duration_s > LONG_WAIT_THRESH:
            all_alerts.append(Alert(
                timestamp_s = w.t_start,
                cycle_id    = 0,
                alert_type  = 'LONG_WAIT',
                severity    = 'info',
                message     = ALERT_MESSAGES['LONG_WAIT'],
                value       = round(w.duration_s, 1),
                unit        = 's',
            ))

    # Ordenar alertas por timestamp
    all_alerts.sort(key=lambda a: a.timestamp_s)
    return profiles, all_alerts


def compute_truck_loads(cycles, video_events: List[Dict]) -> List[TruckLoadEvent]:
    """
    Agrupa ciclos por truck_id y calcula carga acumulada por camion.
    Determina si el camion salio lleno o no (despacho).
    """
    # Mapear ciclo → truck_id desde video_events
    cycle_to_truck = {}
    for ev in video_events:
        if ev.get('event_type') == 'dump':
            cid = ev.get('cycle_id')
            if cid:
                cycle_to_truck[cid] = {
                    'truck_id': ev.get('truck_id', 'unknown'),
                    'model':    ev.get('truck_model', 'unknown'),
                    'capacity': ev.get('capacity_t', 219.0),
                }

    # Agrupar ciclos por truck_id
    truck_cycles: Dict[str, List] = {}
    for c in cycles:
        if c.is_mini_cycle:
            continue
        info = cycle_to_truck.get(c.cycle_id, {
            'truck_id': c.truck_id or 'unknown',
            'model':    'unknown',
            'capacity': 219.0,
        })
        tid = info['truck_id']
        if tid not in truck_cycles:
            truck_cycles[tid] = {'cycles': [], 'info': info}
        truck_cycles[tid]['cycles'].append(c)

    events = []
    for tid, data in truck_cycles.items():
        cs       = sorted(data['cycles'], key=lambda x: x.t_start)
        info     = data['info']
        total_t  = sum(c.payload_t for c in cs)
        cap      = info['capacity']
        fill_pct = min(100.0, total_t / cap * 100)

        events.append(TruckLoadEvent(
            truck_id        = tid,
            model           = info['model'],
            capacity_t      = cap,
            t_first_pass_s  = cs[0].t_start,
            t_last_pass_s   = cs[-1].t_end,
            n_passes        = len(cs),
            total_payload_t = round(total_t, 1),
            fill_pct        = round(fill_pct, 1),
            dispatched      = fill_pct >= 85.0,
        ))

    return events


def compute_transport_metrics(
    truck_events: List[TruckLoadEvent],
    wait_events,
    duration_s: float,
) -> TransportMetrics:
    """Metricas de despacho y transporte."""
    if not truck_events:
        return TransportMetrics(0,0,0,0.0,0.0,0.0,0.0,0.0,0.0)

    n_full    = sum(1 for te in truck_events if te.dispatched)
    n_partial = len(truck_events) - n_full
    avg_passes = np.mean([te.n_passes for te in truck_events])
    avg_fill   = np.mean([te.fill_pct for te in truck_events])
    total_t    = sum(te.total_payload_t for te in truck_events)

    # Gaps entre camiones (tiempo entre ultimo dump del camion N y primero del N+1)
    sorted_trucks = sorted(truck_events, key=lambda x: x.t_first_pass_s)
    gaps = []
    for i in range(1, len(sorted_trucks)):
        gap = sorted_trucks[i].t_first_pass_s - sorted_trucks[i-1].t_last_pass_s
        if gap > 0:
            gaps.append(gap)

    avg_gap = float(np.mean(gaps)) if gaps else 0.0
    max_gap = float(np.max(gaps)) if gaps else 0.0
    disp_eff = (n_full / len(truck_events) * 100) if truck_events else 0.0

    return TransportMetrics(
        n_trucks_served      = len(truck_events),
        n_trucks_full        = n_full,
        n_trucks_partial     = n_partial,
        avg_passes_per_truck = round(float(avg_passes), 1),
        avg_truck_fill_pct   = round(float(avg_fill), 1),
        total_dispatch_t     = round(total_t, 1),
        avg_dispatch_gap_s   = round(avg_gap, 1),
        max_dispatch_gap_s   = round(max_gap, 1),
        dispatch_efficiency  = round(disp_eff, 1),
    )


def build_realtime_timeline(
    df_imu: pd.DataFrame,
    cycles,
    wait_events,
    profiles: List[EfficiencyProfile],
    alerts: List[Alert],
    step_s: float = 0.5,
) -> List[Dict]:
    """
    Construye un array de snapshots cada `step_s` segundos.
    Usado por el JavaScript del dashboard para simular tiempo real.
    """
    t   = df_imu.timestamp_s.values
    ax  = df_imu.ax.values
    ay  = df_imu.ay.values
    az  = df_imu.az.values
    gx  = df_imu.gx.values
    gy  = df_imu.gy.values
    gz  = df_imu.gz.values
    an  = df_imu.accel_norm.values
    gn  = df_imu.gyro_norm.values
    dur = float(df_imu.attrs['duration_s'])

    # Pre-map: para cada time-step, cual es el ciclo activo y sus metricas
    prof_map = {p.cycle_id: p for p in profiles}

    # Alertas indexadas por segundo
    alert_by_sec: Dict[int, List[Alert]] = {}
    for a in alerts:
        key = int(a.timestamp_s)
        alert_by_sec.setdefault(key, []).append(a)

    # Ciclo activo por tiempo
    def active_cycle_at(ts):
        for c in cycles:
            if c.t_start <= ts <= c.t_end:
                return c
        return None

    timeline = []
    snap_t = 0.0
    while snap_t <= dur:
        imu_i = min(np.searchsorted(t, snap_t), len(t) - 1)

        # Ciclo activo
        ac = active_cycle_at(snap_t)
        ac_id   = ac.cycle_id if ac else None
        ac_prof = prof_map.get(ac_id) if ac_id else None
        phase   = _phase_at(snap_t, ac)

        # Alertas en este segundo
        sec_alerts = []
        for delta in [0, 1]:
            k = int(snap_t) + delta
            for a in alert_by_sec.get(k, []):
                sec_alerts.append({
                    'type':     a.alert_type,
                    'severity': a.severity,
                    'msg':      a.message,
                    'value':    a.value,
                    'unit':     a.unit,
                })

        snap = {
            't':            round(snap_t, 1),
            'ax':           round(float(ax[imu_i]), 2),
            'ay':           round(float(ay[imu_i]), 2),
            'az':           round(float(az[imu_i]), 2),
            'gx':           round(float(gx[imu_i]), 2),
            'gy':           round(float(gy[imu_i]), 2),
            'gz':           round(float(gz[imu_i]), 2),
            'accel_norm':   round(float(an[imu_i]), 2),
            'gyro_norm':    round(float(gn[imu_i]), 2),
            'phase':        phase,
            'cycle_id':     ac_id,
            'coll_eff':     ac_prof.collection_eff_pct if ac_prof else None,
            'maneuver_eff': ac_prof.maneuver_eff_pct  if ac_prof else None,
            'pos_eff':      ac_prof.positioning_eff_pct if ac_prof else None,
            'disc_eff':     ac_prof.discharge_eff_pct if ac_prof else None,
            'cycle_eff':    ac_prof.cycle_eff_pct    if ac_prof else None,
            'fill_pct':     round(ac.fill_factor * 100, 1) if ac else None,
            'payload_t':    round(ac.payload_t, 1) if ac else None,
            'volume_m3':    round(ac.fill_factor * BUCKET_M3, 1) if ac else None,
            'alerts':       sec_alerts,
        }
        timeline.append(snap)
        snap_t = round(snap_t + step_s, 1)

    return timeline


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _time_to_peak_fraction(seg_t, seg_gn, fraction=0.5) -> float:
    """Tiempo desde inicio hasta que se alcanza `fraction` del pico maximo."""
    if len(seg_gn) < 2:
        return 2.0
    peak = np.max(seg_gn)
    thresh = peak * fraction
    idx = np.where(seg_gn >= thresh)[0]
    if len(idx) == 0:
        return float(seg_t[-1] - seg_t[0])
    return max(0.1, float(seg_t[idx[0]] - seg_t[0]))


def _compute_dump_quality(seg_t, seg_az, seg_an, dump_t) -> Tuple[float, float]:
    """
    Mide la suavidad del dump.
    Un dump suave tiene un spike gradual de accel.
    Retorna (smoothness 0-1, efficiency_pct).
    """
    if dump_t is None or len(seg_az) < 4:
        return 0.5, 70.0

    di = np.searchsorted(seg_t, dump_t)
    window = seg_an[max(0, di-3):di+8]
    if len(window) < 2:
        return 0.5, 70.0

    # Calidad = 1 - (rango del spike / max_spike)
    spike_max = float(np.max(window))
    spike_std = float(np.std(np.diff(window)))
    smoothness = max(0.0, 1.0 - (spike_std / max(spike_max, 1.0)) * 3.0)
    smoothness = min(1.0, smoothness)
    eff = 40.0 + smoothness * 60.0
    return smoothness, eff


def _phase_at(ts: float, cycle) -> str:
    """Determina la fase operacional en un timestamp dado."""
    if cycle is None:
        return 'WAIT'
    if cycle.swing_loaded and cycle.swing_loaded.t_start <= ts <= cycle.swing_loaded.t_end:
        return 'SWING_LOADED'
    if cycle.swing_empty and cycle.swing_empty.t_start <= ts <= cycle.swing_empty.t_end:
        return 'SWING_EMPTY'
    if cycle.t_dump and abs(ts - cycle.t_dump) < 2.0:
        return 'DUMP'
    if ts < (cycle.t_dump or cycle.t_end):
        return 'DIG'
    return 'REPOSITION'


# ─── WEAR SCORE + SPILL + MATERIAL HARDNESS (BPMN extendido) ─────────────────

def compute_wear_score(alerts: List[Alert], df_imu: pd.DataFrame, cycles) -> Dict:
    """
    Calcula un score de desgaste acumulado del equipo.

    Combina:
      - Impactos duros durante DIG (peso 3)
      - Dumps bruscos (peso 2)
      - Mini-ciclos (peso 0.5)
      - Vibracion media global (peso 1)

    Ademas estima la dureza del material (material_hardness 0-1)
    desde la aceleracion RMS durante DIG.

    Retorna dict con:
      total_score:       float  (0-100 aprox)
      hard_impacts:      int
      rough_dumps:       int
      mini_cycles:       int
      vibration_rms:     float
      material_hardness: float (0-1)
    """
    hard_impacts = sum(1 for a in alerts if a.alert_type == 'HARD_IMPACT')
    rough_dumps  = sum(1 for a in alerts if a.alert_type == 'ROUGH_DUMP')
    mini_cyc     = sum(1 for a in alerts if a.alert_type == 'MINI_CYCLE')

    # Vibracion RMS global (indicador constante de estres)
    an = df_imu.accel_norm.values if hasattr(df_imu, 'accel_norm') else np.array([])
    vib_rms = float(np.sqrt(np.mean(np.square(an - np.mean(an))))) if len(an) else 0.0

    # Score compuesto (aproxima 0-100)
    raw = (hard_impacts * 3.0 +
           rough_dumps * 2.0 +
           mini_cyc * 0.5 +
           vib_rms * 0.8)
    # Normalizar a ~0-100 asumiendo 30 eventos severos ≈ 100
    total_score = min(100.0, raw * 2.0)

    # Material hardness: promedio del accel_norm durante DIG (inicio del ciclo)
    hardness_scores = []
    if len(an) > 0 and cycles:
        t = df_imu.timestamp_s.values
        for c in cycles:
            if c.is_mini_cycle:
                continue
            i0 = np.searchsorted(t, c.t_start)
            i1 = min(np.searchsorted(t, c.t_start + 3.0), len(an))
            if i1 > i0:
                dig_region = an[i0:i1]
                # Normalizar: 10 m/s² = suave, 20+ m/s² = duro
                hardness_scores.append(np.mean(dig_region))

    if hardness_scores:
        avg_dig_accel = float(np.mean(hardness_scores))
        material_hardness = min(1.0, max(0.0, (avg_dig_accel - 8.0) / 14.0))
    else:
        material_hardness = 0.5

    return {
        'total_score':       round(total_score, 1),
        'hard_impacts':      hard_impacts,
        'rough_dumps':       rough_dumps,
        'mini_cycles':       mini_cyc,
        'vibration_rms':     round(vib_rms, 2),
        'material_hardness': round(material_hardness, 2),
    }


# ─── IDLE TIME INDEX (ITI) — metrica maestra de tiempos de ocio ───────────────

IDLE_CATEGORY_LABELS = {
    'pre_load':     'Pre-carga (esperando primer camion)',
    'inter_cycle':  'Entre ciclos (cambio de camion / reposicion)',
    'end_session':  'Fin de sesion (sin actividad posterior)',
    'critical':     'Interrupcion critica (>60s)',
    'unknown':      'Otros',
}

IDLE_BANDS = [
    # (lo, hi, label, color, recommendation)
    (0.00, 0.15, 'OPTIMO',       'green',
     'Operacion muy eficiente. Los tiempos de ocio son minimos.'),
    (0.15, 0.30, 'NORMAL',       'blue',
     'Ocio dentro de rango esperado. Oportunidad de optimizar despacho.'),
    (0.30, 0.50, 'CRITICO',      'yellow',
     'Mucho tiempo sin producir. Revisar coordinacion pala-camion.'),
    (0.50, 1.01, 'COMPROMETIDO', 'red',
     'Mas de la mitad del tiempo sin actividad. Intervencion urgente.'),
]


def _classify_idle(idle_pct_0_1: float) -> Dict[str, str]:
    for lo, hi, lbl, col, rec in IDLE_BANDS:
        if lo <= idle_pct_0_1 < hi:
            return {'band': lbl, 'color': col, 'recommendation': rec}
    return {'band': 'N/A', 'color': 'muted', 'recommendation': 'Sin datos'}


def compute_idle_index(wait_events: List,
                        cycles: List,
                        duration_s: float) -> Dict:
    """
    Calcula el Indice de Tiempos de Ocio (ITI).

    Definicion:
        ITI = T_ocio_total / T_observado

    Donde T_ocio = suma de tiempos en wait events (inter_cycle + pre_load +
    end_session + waits >5s entre ciclos).

    Categoriza cada wait segun razon operacional.

    Retorna dict con:
        iti:                float  (0-1)
        iti_pct:            float  (0-100)
        total_idle_s:       float
        total_duration_s:   float
        n_events:           int
        by_category:        dict {categoria: {total_s, n_events, pct_of_idle}}
        top_waits:          list[dict] top 5 waits mas largos
        critical_waits_n:   int  (cantidad de waits > 60s)
        longest_wait_s:     float
        productive_s:       float
        band:               str OPTIMO/NORMAL/CRITICO/COMPROMETIDO
        band_color:         str
        recommendation:     str
        ratio_prod_idle:    float  (productive / idle)
    """
    # Suma total de ocio
    total_idle_s = sum(w.duration_s for w in wait_events)

    if duration_s <= 0:
        return {
            'iti': 0.0, 'iti_pct': 0.0,
            'total_idle_s': 0.0, 'total_duration_s': 0.0,
            'n_events': 0, 'by_category': {},
            'top_waits': [], 'critical_waits_n': 0, 'longest_wait_s': 0.0,
            'productive_s': 0.0, 'band': 'N/A', 'band_color': 'muted',
            'recommendation': 'Sin duracion observada', 'ratio_prod_idle': 0.0,
        }

    iti = total_idle_s / duration_s
    iti = float(np.clip(iti, 0.0, 1.0))

    # ── Desglose por categoria ─────────────────────────────────────────────────
    by_cat: Dict[str, Dict] = {}
    for cat in IDLE_CATEGORY_LABELS.keys():
        by_cat[cat] = {'total_s': 0.0, 'n_events': 0, 'pct_of_idle': 0.0}

    for w in wait_events:
        cat = w.reason if w.reason in IDLE_CATEGORY_LABELS else 'unknown'
        # Reclasificar waits muy largos como criticos (superponer)
        if w.duration_s > 60:
            by_cat['critical']['total_s'] += w.duration_s
            by_cat['critical']['n_events'] += 1
        # Siempre registrar en la categoria original tambien
        by_cat[cat]['total_s'] += w.duration_s
        by_cat[cat]['n_events'] += 1

    # Recalcular porcentajes
    for cat in by_cat:
        by_cat[cat]['total_s'] = round(by_cat[cat]['total_s'], 1)
        if total_idle_s > 0:
            by_cat[cat]['pct_of_idle'] = round(
                by_cat[cat]['total_s'] / total_idle_s * 100, 1)
        by_cat[cat]['label'] = IDLE_CATEGORY_LABELS[cat]

    # ── Top waits mas largos ───────────────────────────────────────────────────
    sorted_waits = sorted(wait_events, key=lambda w: w.duration_s, reverse=True)
    top_waits = []
    for w in sorted_waits[:5]:
        top_waits.append({
            't_start':   round(w.t_start, 1),
            't_end':     round(w.t_end, 1),
            'duration':  round(w.duration_s, 1),
            'reason':    w.reason,
            'label':     IDLE_CATEGORY_LABELS.get(w.reason, 'Otros'),
            'pct_total': round(w.duration_s / duration_s * 100, 2),
        })

    critical_waits_n = sum(1 for w in wait_events if w.duration_s > 60)
    longest = max((w.duration_s for w in wait_events), default=0.0)

    productive_s = max(0.0, duration_s - total_idle_s)
    ratio = (productive_s / total_idle_s) if total_idle_s > 0 else float('inf')

    band = _classify_idle(iti)

    return {
        'iti':              round(iti, 3),
        'iti_pct':          round(iti * 100, 1),
        'total_idle_s':     round(total_idle_s, 1),
        'total_duration_s': round(duration_s, 1),
        'n_events':         len(wait_events),
        'by_category':      by_cat,
        'top_waits':        top_waits,
        'critical_waits_n': critical_waits_n,
        'longest_wait_s':   round(longest, 1),
        'productive_s':     round(productive_s, 1),
        'band':             band['band'],
        'band_color':       band['color'],
        'recommendation':   band['recommendation'],
        'ratio_prod_idle':  round(ratio, 2) if ratio != float('inf') else -1,
    }


def extract_spill_events(alerts: List[Alert]) -> List[Dict]:
    """
    Extrae todos los eventos que implican desperdicio de material:
      - POSSIBLE_SPILL
      - OVERFILL
      - ROUGH_DUMP
    Cada evento se convierte en un dict serializable.
    """
    spill_types = {'POSSIBLE_SPILL', 'OVERFILL', 'ROUGH_DUMP'}
    out = []
    for a in alerts:
        if a.alert_type in spill_types:
            out.append({
                't':        round(a.timestamp_s, 2),
                'cycle_id': a.cycle_id,
                'type':     a.alert_type,
                'value':    a.value,
                'unit':     a.unit,
                'severity': a.severity,
                'message':  a.message,
            })
    return out
