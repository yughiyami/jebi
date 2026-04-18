# -*- coding: utf-8 -*-
"""
ipo_metrics.py  —  JEBI Hackathon 2026

Indice de Productividad Operativa (IPO) — formula maestra del proceso minero.

Formula maestra:

                V_nom · η_R · η_M · C_T
    IPO =  ─────────────────────────────────── · (1 - α_W) · (1 - α_DE)
                        T_ciclo

Desglose por componente:

  1. Output de volumen efectivo:   V_ef = V_nom · η_R
     η_R = V_real / V_nom  →  Fill Factor de recoleccion

  2. Tiempo de ciclo total:
     T_ciclo = T_pos_C + T_carga + T_viaje_c + T_desc + T_pos_D + T_viaje_v

  3. Ciclos por turno:   C_T = (T_turno · DA) / T_ciclo
     DA = Disponibilidad de equipo en el turno (0-1)
     DA esta ligada al nodo "Desgaste de Equipos"

  4. Factores de perdida (los "leaks" del diagrama):
     α_W  = V_desperdiciado / V_cargado      (Desperdicios)
     α_DE = T_inactivo_mant  / T_turno       (Desgaste de Equipos)

Interpretacion:
  IPO > 0.85     →  Operacion optima
  0.65 - 0.85    →  Operacion normal, mejoras puntuales
  0.45 - 0.65    →  Ineficiencias criticas, revisar η_M o α_DE
  IPO < 0.45     →  Operacion comprometida

Estas son las CLAVES MAESTRAS usadas en el dashboard.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ─── CONSTANTES DEL EQUIPO ────────────────────────────────────────────────────

V_NOM_M3   = 27.0          # Volumen nominal del balde EX-5600 (m³)
T_TURNO_S  = 8 * 3600.0    # Duracion estandar del turno minero (8h = 28800s)


# ─── INTERPRETACION IPO ───────────────────────────────────────────────────────

# Modelo OEE (Overall Equipment Effectiveness) adaptado:
#   OEE = η_T · η_F · η_M_speed · (1 - α_W)
#
# Donde:
#   η_T       = (T_total - T_ocio) / T_total              → Availability
#   η_F       = fill_factor / TARGET_FILL_FACTOR          → Quality
#   η_M_speed = TARGET_CYCLE_S / T_ciclo                  → Performance
#   α_W       = V_desperdiciado / V_cargado               → Loss factor
#
# Cada componente está en 0-1 y compara contra un BENCHMARK REAL, no contra un
# ideal teórico. Esto evita que η_M quede artificialmente aplastado cuando se
# divide swing_time por un t_ciclo inflado con waits inter-ciclo.

TARGET_FILL_FACTOR = 0.85    # 85% fill = óptimo para EX-5600 bucket
TARGET_CYCLE_S     = 28.0    # benchmark de ciclo completo en segundos

# Bandas ajustadas a escala OEE (los productos de 3 factores <=1 son más bajos)
IPO_BANDS = [
    (0.60, 1.01,  'OPTIMA',        'green',  'OEE optimo. Mantener estandares.'),
    (0.40, 0.60,  'NORMAL',        'blue',   'OEE normal. Revisar el factor mas bajo entre η_T, η_F, η_M.'),
    (0.20, 0.40,  'CRITICA',       'yellow', 'OEE critico. Identificar cuello de botella dominante.'),
    (0.00, 0.20,  'COMPROMETIDA',  'red',    'OEE comprometido. Intervencion inmediata requerida.'),
]


def classify_ipo(ipo: float) -> Dict:
    """Devuelve banda, color y recomendacion para un valor IPO."""
    for lo, hi, label, color, rec in IPO_BANDS:
        if lo <= ipo < hi:
            return {'band': label, 'color': color, 'recommendation': rec}
    return {'band': 'N/A', 'color': 'muted', 'recommendation': 'Datos insuficientes'}


# ─── DATACLASS RESULTADO ──────────────────────────────────────────────────────

@dataclass
class IPOResult:
    """Resultado OEE (mantiene nombre IPO por compat).

    Formula:  OEE = η_T · η_F · η_M_speed · (1 - α_W)
    """

    # Valor principal (OEE)
    ipo:              float

    # ── Componentes OEE nuevos ─────────────────────────────────────────────
    eta_T:            float       # η_T: availability  (productive / total)
    eta_F:            float       # η_F: fill vs target  (fill / 0.85)
    eta_M_speed:      float       # η_M: speed vs benchmark  (28s / t_ciclo)

    # ── Componentes legacy (para referencia / trazabilidad) ────────────────
    v_nom:            float       # Volumen nominal balde (m³)
    eta_R:            float       # Fill factor raw (0-1)
    eta_M:            float       # η_M legacy (swing/t_ciclo)
    c_T:              float       # Ciclos por turno estimados

    # Tiempo de ciclo
    t_ciclo:          float
    t_pos_C:          float
    t_carga:          float
    t_viaje_c:        float
    t_desc:           float
    t_pos_D:          float
    t_viaje_v:        float

    # Disponibilidad
    t_turno:          float
    DA:               float

    # Factores de perdida
    alpha_W:          float
    alpha_DE:         float
    v_desperdiciado:  float
    v_cargado:        float
    t_mant:           float

    # Derivados
    v_ef:             float
    band:             str
    band_color:       str
    recommendation:   str

    # Validación visual (dual-source)
    ipo_validated:          float = 0.0
    visual_agreement_pct:   float = 0.0
    confidence:             str   = 'N/A'
    eta_M_validated:        float = 0.0

    # Benchmarks usados
    target_fill:      float = TARGET_FILL_FACTOR
    target_cycle_s:   float = TARGET_CYCLE_S

    # Debug
    notes:            List[str] = field(default_factory=list)


# ─── CALCULO PRINCIPAL ────────────────────────────────────────────────────────

def compute_ipo(cycles, wait_events, alerts, transport, metrics,
                t_turno_s: float = T_TURNO_S,
                visual_validation: Optional[Dict] = None) -> IPOResult:
    """
    Calcula el IPO completo y todos sus componentes desde los datos del pipeline.

    Args:
        cycles:            lista de LoadCycle
        wait_events:       lista de WaitEvent
        alerts:            lista de Alert
        transport:         TransportMetrics
        metrics:           SessionMetrics
        t_turno_s:         Duracion del turno (default 8h)
        visual_validation: Dict con datos de motion_detector (optional).
                           Si se provee, el IPO se ENRIQUECE con:
                             - eta_M_validated (corregido por motion visual)
                             - ipo_validated (IPO ajustado por acuerdo IMU↔cámara)
                             - confidence score dual-source

    Returns:
        IPOResult con ipo y todos los desgloses (incluye validación visual si aplica).
    """
    notes: List[str] = []
    full_cycles = [c for c in cycles if not c.is_mini_cycle]

    if not full_cycles:
        notes.append('Sin ciclos completos; IPO no calculable.')
        return IPOResult(
            ipo=0.0, eta_T=0, eta_F=0, eta_M_speed=0,
            v_nom=V_NOM_M3, eta_R=0, eta_M=0, c_T=0,
            t_ciclo=0, t_pos_C=0, t_carga=0, t_viaje_c=0,
            t_desc=0, t_pos_D=0, t_viaje_v=0,
            t_turno=t_turno_s, DA=0,
            alpha_W=0, alpha_DE=0,
            v_desperdiciado=0, v_cargado=0, t_mant=0,
            v_ef=0, band='N/A', band_color='muted',
            recommendation='Sin ciclos detectados', notes=notes,
        )

    # ── η_R  Fill factor promedio ─────────────────────────────────────────────
    fills = [c.fill_factor for c in full_cycles]
    eta_R = float(np.mean(fills))

    # ── Tiempo de ciclo REAL (incluyendo waits entre ciclos) ──────────────────
    # Metodo robusto: duracion total de la ventana / numero de ciclos completos
    # Esto captura el tiempo efectivo por ciclo incluyendo posicionamiento, wait, etc.
    if metrics.total_duration_s > 0 and len(full_cycles) > 0:
        t_ciclo = metrics.total_duration_s / len(full_cycles)
    else:
        t_ciclo = float(np.mean([(c.t_end - c.t_start) for c in full_cycles]))

    # ── Swings individuales ───────────────────────────────────────────────────
    swing_loaded_durs = [c.swing_loaded.duration_s for c in full_cycles if c.swing_loaded]
    swing_empty_durs  = [c.swing_empty.duration_s  for c in full_cycles if c.swing_empty]

    t_viaje_c = float(np.mean(swing_loaded_durs)) if swing_loaded_durs else 0.0
    t_viaje_v = float(np.mean(swing_empty_durs))  if swing_empty_durs  else 0.0

    # T_carga: tiempo efectivo de excavacion (fase DIG)
    # Aproximamos como 25% del tiempo ACTIVO (sin wait) del ciclo
    t_activo  = float(np.mean([(c.t_end - c.t_start) for c in full_cycles]))
    t_carga   = max(2.0, t_activo * 0.25)
    t_desc    = 2.0

    # Lo que queda es posicionamiento + waits
    t_swing   = t_viaje_c + t_viaje_v
    t_remain  = max(0.0, t_ciclo - t_swing - t_carga - t_desc)
    t_pos_C   = max(0.5, t_remain * 0.55)
    t_pos_D   = max(0.5, t_remain * 0.45)

    # ── η_M  Eficiencia de maniobra ──────────────────────────────────────────
    # Fraccion del ciclo completo que es swing productivo (0-1)
    # Sin factor de calibracion artificial
    eta_M = t_swing / max(t_ciclo, 1.0)
    eta_M = float(np.clip(eta_M, 0.05, 1.0))

    # ── DA  Disponibilidad de equipo ─────────────────────────────────────────
    # DA = fraccion del tiempo en estado operable (no en mantenimiento).
    # En una ventana corta asumimos disponibilidad alta salvo que haya waits
    # largos anomalos (> 120s) que sugieren falla/mantenimiento.
    abnormal_waits_s = sum(w.duration_s for w in wait_events
                            if w.duration_s > 120)
    t_mant_s = abnormal_waits_s
    if metrics.total_duration_s > 0:
        DA = max(0.0, 1.0 - (t_mant_s / metrics.total_duration_s))
    else:
        DA = 1.0
    DA = float(np.clip(DA, 0.0, 1.0))

    # ── C_T  Ciclos por turno ────────────────────────────────────────────────
    # Proyeccion: cuantos ciclos se harian en un turno estandar a este ritmo
    if t_ciclo > 0:
        c_T = (t_turno_s * DA) / t_ciclo
    else:
        c_T = 0.0

    # ── α_W  Desperdicios (SOLO derrames / overfill reales) ──────────────────
    # NO incluimos underfill aca: eso ya esta capturado en η_R.
    # α_W = V_derramado / V_cargado
    v_cargado = sum(c.fill_factor * V_NOM_M3 for c in full_cycles)

    # Eventos de spill/overfill/rough_dump: cada uno ~3% del balde en perdida
    spill_count = sum(1 for a in alerts
                       if a.alert_type in ('POSSIBLE_SPILL', 'ROUGH_DUMP', 'OVERFILL'))
    v_desperdiciado = spill_count * 0.03 * V_NOM_M3

    alpha_W = v_desperdiciado / max(v_cargado, 1.0)
    alpha_W = float(np.clip(alpha_W, 0.0, 1.0))

    # ── α_DE  Desgaste de equipos ────────────────────────────────────────────
    alpha_DE = t_mant_s / max(metrics.total_duration_s, 1.0)
    alpha_DE = float(np.clip(alpha_DE, 0.0, 1.0))

    # ── Volumen efectivo ──────────────────────────────────────────────────────
    v_ef = V_NOM_M3 * eta_R

    # ── OEE FORMULA (Overall Equipment Effectiveness) ─────────────────────────
    # Reemplaza el IPO legacy que aplastaba η_M por dividir swing/t_ciclo inflado.
    #
    # OEE = η_T · η_F · η_M_speed · (1 - α_W)
    #
    # Cada factor compara contra un BENCHMARK real:
    #   η_T  = tiempo productivo / tiempo total               (availability)
    #   η_F  = fill_factor_real / TARGET_FILL_FACTOR (0.85)   (quality)
    #   η_M  = TARGET_CYCLE_S (28s) / t_ciclo_real            (performance)
    # ──────────────────────────────────────────────────────────────────────────

    # η_T (availability): cuánto del tiempo fue productivo.
    # FIX: si tenemos validación visual (fusión IMU+Cámara), usamos el ocio
    # REAL detectado por fusión estricta (9s min duration + ground-truth thr).
    # El total_wait_s del IMU crudo está inflado por falsos positivos.
    total_wait_s_raw = sum(w.duration_s for w in wait_events)
    if visual_validation and visual_validation.get('total_inactivo_s', 0) > 0:
        # Fuente de verdad: fusión dual-source
        total_idle_s = visual_validation['total_inactivo_s']
        notes.append(f'eta_T usa ocio fusionado (IMU+Camara): {total_idle_s:.1f}s '
                      f'(IMU crudo: {total_wait_s_raw:.1f}s)')
    else:
        # Fallback: IMU crudo (inflado pero mejor que nada)
        total_idle_s = total_wait_s_raw

    if metrics.total_duration_s > 0:
        eta_T = (metrics.total_duration_s - total_idle_s) / metrics.total_duration_s
    else:
        eta_T = 0.0
    eta_T = float(np.clip(eta_T, 0.0, 1.0))

    # η_F (quality): fill vs target óptimo
    eta_F = float(np.clip(eta_R / TARGET_FILL_FACTOR, 0.0, 1.0))

    # η_M_speed (performance): velocidad de ciclo vs benchmark
    eta_M_speed = float(np.clip(TARGET_CYCLE_S / max(t_ciclo, 1.0), 0.0, 1.0))

    # OEE final
    ipo = eta_T * eta_F * eta_M_speed * (1 - alpha_W)
    ipo = float(np.clip(ipo, 0.0, 1.0))

    if not np.isfinite(ipo):
        ipo = 0.0
        notes.append('OEE calculado como 0 por valores no finitos.')

    notes.append(f'OEE components: eta_T={eta_T:.3f}, eta_F={eta_F:.3f}, '
                  f'eta_M={eta_M_speed:.3f}, alpha_W={alpha_W:.3f}')

    band = classify_ipo(ipo)

    # ── VALIDACIÓN VISUAL (dual-source) ──────────────────────────────────────
    # Si tenemos motion data, ajustamos el OEE con validación visual.
    # Penalizamos η_M_speed si el motion visual no confirma actividad durante
    # los ciclos (los waits que IMU no detectó pero cámara sí → ciclos "ficticios").
    eta_M_val = eta_M_speed
    ipo_val = ipo
    agreement_pct = 0.0
    confidence = 'N/A'

    if visual_validation:
        verified = visual_validation.get('verified_idle', {})
        agreement_pct = float(verified.get('agreement_pct', 0.0))
        confidence    = verified.get('confidence', 'N/A')

        cycle_confs = visual_validation.get('cycle_confidence', [])
        if cycle_confs:
            high_conf_cycles = sum(1 for cc in cycle_confs if cc.get('confidence') == 'HIGH')
            total_cycles_val = len(cycle_confs)
            if total_cycles_val > 0:
                motion_confirmation_ratio = high_conf_cycles / total_cycles_val
                # Ajustar η_M_speed según confirmación visual
                eta_M_val = eta_M_speed * (0.5 + 0.5 * motion_confirmation_ratio)
                eta_M_val = float(np.clip(eta_M_val, 0.0, 1.0))

        # OEE validado con η_M ajustado
        ipo_val = eta_T * eta_F * eta_M_val * (1 - alpha_W)
        ipo_val = float(np.clip(ipo_val, 0.0, 1.0))

        notes.append(f'Validacion visual aplicada (acuerdo {agreement_pct:.1f}%).')

    return IPOResult(
        ipo=round(ipo, 3),
        eta_T=round(eta_T, 3),
        eta_F=round(eta_F, 3),
        eta_M_speed=round(eta_M_speed, 3),
        v_nom=V_NOM_M3,
        eta_R=round(eta_R, 3),
        eta_M=round(eta_M, 3),
        c_T=round(c_T, 1),
        t_ciclo=round(t_ciclo, 1),
        t_pos_C=round(t_pos_C, 1),
        t_carga=round(t_carga, 1),
        t_viaje_c=round(t_viaje_c, 1),
        t_desc=round(t_desc, 1),
        t_pos_D=round(t_pos_D, 1),
        t_viaje_v=round(t_viaje_v, 1),
        t_turno=round(t_turno_s, 0),
        DA=round(DA, 3),
        alpha_W=round(alpha_W, 3),
        alpha_DE=round(alpha_DE, 3),
        v_desperdiciado=round(v_desperdiciado, 1),
        v_cargado=round(v_cargado, 1),
        t_mant=round(t_mant_s, 1),
        v_ef=round(v_ef, 1),
        band=band['band'],
        band_color=band['color'],
        recommendation=band['recommendation'],
        ipo_validated=round(ipo_val, 3),
        eta_M_validated=round(eta_M_val, 3),
        visual_agreement_pct=round(agreement_pct, 1),
        confidence=confidence,
        target_fill=TARGET_FILL_FACTOR,
        target_cycle_s=TARGET_CYCLE_S,
        notes=notes,
    )


# ─── DESCRIPCION TEXTUAL PARA EL DASHBOARD ────────────────────────────────────

IPO_DEFINITIONS = {
    'IPO':      ('OEE — Overall Equipment Effectiveness',
                 'Metrica maestra en escala 0-1 compuesta por 3 factores independientes: '
                 'Availability (η_T) × Quality (η_F) × Performance (η_M) × (1 - Losses).'),
    'eta_T':    ('η_T — Availability (disponibilidad productiva)',
                 'Fraccion del tiempo total que fue productivo (no ocio). '
                 'η_T = (T_total - T_ocio) / T_total.'),
    'eta_F':    ('η_F — Quality (calidad de carga)',
                 f'Fill factor real vs target optimo ({TARGET_FILL_FACTOR*100:.0f}%). '
                 f'η_F = fill_factor / {TARGET_FILL_FACTOR:.2f}.'),
    'eta_M_speed': ('η_M — Performance (velocidad vs benchmark)',
                 f'Velocidad de ciclo real vs benchmark de {TARGET_CYCLE_S:.0f}s. '
                 f'η_M = {TARGET_CYCLE_S:.0f} / T_ciclo_real.'),
    'V_nom':    ('Volumen nominal del balde',
                 f'Capacidad teorica del balde EX-5600 = {V_NOM_M3} m³.'),
    'eta_R':    ('η_R — Fill Factor raw (referencia)',
                 'Fraccion del balde realmente lleno por cucharada. η_R = V_real / V_nom. '
                 'Usado para calcular η_F.'),
    'eta_M':    ('η_M legacy (referencia)',
                 'Fracción de swing / t_ciclo. Sustituido por η_M_speed en OEE.'),
    'C_T':      ('Ciclos por turno',
                 'Estimacion de ciclos ejecutables en un turno de 8h dada la disponibilidad. '
                 'C_T = (T_turno · DA) / T_ciclo.'),
    'T_ciclo':  ('Tiempo de ciclo total',
                 'T_ciclo = T_pos_C + T_carga + T_viaje_c + T_desc + T_pos_D + T_viaje_v'),
    'T_pos_C':  ('T_pos_C — Posicionamiento de carga',
                 'Tiempo para alinear la pala antes de cargar. Stage "Eficiencia de Maniobra (Carga)".'),
    'T_carga':  ('T_carga — Tiempo de carga / DIG',
                 'Tiempo efectivo de excavacion. Stage "Volumen de Carga".'),
    'T_viaje_c': ('T_viaje_c — Swing cargado',
                 'Tiempo de giro con balde lleno hacia el camion. Stage "Transporte".'),
    'T_desc':   ('T_desc — Tiempo de descarga',
                 'Tiempo en soltar el material sobre la tolva. Stage "Eficiencia de Colocacion".'),
    'T_pos_D':  ('T_pos_D — Posicionamiento en descarga',
                 'Reposicion del balde en la tolva. Stage "Eficiencia de Maniobra (Descarga)".'),
    'T_viaje_v': ('T_viaje_v — Swing vacio',
                 'Retorno del balde sin carga. Stage "Transporte".'),
    'DA':       ('Disponibilidad de equipo',
                 'Fraccion del turno que el equipo esta operando (0-1). '
                 'Ligada al nodo "Desgaste de Equipos".'),
    'alpha_W':  ('α_W — Factor de desperdicios',
                 'α_W = V_desperdiciado / V_cargado. Incluye underfill, derrames y overfills. '
                 'Es el primer "leak" del diagrama.'),
    'alpha_DE': ('α_DE — Factor de desgaste de equipos',
                 'α_DE = T_inactivo_mantenimiento / T_turno. Tiempo perdido en mantenimiento correctivo. '
                 'Segundo "leak" del diagrama.'),
    'V_ef':     ('V_ef — Volumen efectivo',
                 'Volumen real por cucharada = V_nom · η_R.'),
    'V_W':      ('V_W — Volumen desperdiciado',
                 'Volumen (m³) perdido por underfill, derrames y overfills en la ventana.'),
    'T_mant':   ('T_mant — Tiempo en mantenimiento',
                 'Tiempo inactivo por mantenimiento correctivo en el turno.'),
    'T_turno':  ('T_turno — Duracion del turno',
                 'Duracion estandar del turno minero (8 horas = 28800 s).'),
}
