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

IPO_BANDS = [
    (0.85, 1.01,  'OPTIMA',        'green',  'Operacion optima. Mantener estandares.'),
    (0.65, 0.85,  'NORMAL',        'blue',   'Operacion normal, aplicar mejoras puntuales.'),
    (0.45, 0.65,  'CRITICA',       'yellow', 'Ineficiencias criticas. Revisar η_M (maniobra) o α_DE (desgaste).'),
    (0.00, 0.45,  'COMPROMETIDA',  'red',    'Operacion comprometida. Intervencion inmediata requerida.'),
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
    """Resultado completo del calculo IPO con todos los componentes."""

    # Valor principal
    ipo:              float

    # Componentes numerador
    v_nom:            float       # Volumen nominal balde (m³)
    eta_R:            float       # Fill factor (V_real/V_nom)  0-1
    eta_M:            float       # Eficiencia maniobra          0-1
    c_T:              float       # Ciclos por turno estimados

    # Tiempo de ciclo
    t_ciclo:          float       # Tiempo promedio ciclo (s)
    t_pos_C:          float       # Posicionamiento carga (s)
    t_carga:          float       # Tiempo de carga/DIG (s)
    t_viaje_c:        float       # Swing cargado (s)
    t_desc:           float       # Descarga (s)
    t_pos_D:          float       # Posicionamiento descarga (s)
    t_viaje_v:        float       # Swing vacio / retorno (s)

    # Disponibilidad
    t_turno:          float       # T_turno usado (s)
    DA:               float       # Disponibilidad equipo (0-1)

    # Factores de perdida
    alpha_W:          float       # Desperdicios (0-1)
    alpha_DE:         float       # Desgaste equipos (0-1)
    v_desperdiciado:  float       # V perdido (m³)
    v_cargado:        float       # V cargado total (m³)
    t_mant:           float       # Tiempo en mantenimiento (s)

    # Derivados de lectura
    v_ef:             float       # Volumen efectivo V_nom * η_R
    band:             str         # OPTIMA / NORMAL / CRITICA / COMPROMETIDA
    band_color:       str
    recommendation:   str

    # Debug / auditoria
    notes:            List[str] = field(default_factory=list)


# ─── CALCULO PRINCIPAL ────────────────────────────────────────────────────────

def compute_ipo(cycles, wait_events, alerts, transport, metrics,
                t_turno_s: float = T_TURNO_S) -> IPOResult:
    """
    Calcula el IPO completo y todos sus componentes desde los datos del pipeline.

    Args:
        cycles:       lista de LoadCycle
        wait_events:  lista de WaitEvent
        alerts:       lista de Alert
        transport:    TransportMetrics
        metrics:      SessionMetrics
        t_turno_s:    Duracion del turno (default 8h)

    Returns:
        IPOResult con ipo y todos los desgloses.
    """
    notes: List[str] = []
    full_cycles = [c for c in cycles if not c.is_mini_cycle]

    if not full_cycles:
        notes.append('Sin ciclos completos; IPO no calculable.')
        return IPOResult(
            ipo=0.0, v_nom=V_NOM_M3, eta_R=0, eta_M=0, c_T=0,
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

    # ── IPO FORMULA MAESTRA (normalizado 0-1) ─────────────────────────────────
    # La formula teorica da una "productividad" en m³/s² (unidades raras).
    # Para normalizar a 0-1 usamos como referencia la produccion IDEAL
    # donde η_R=η_M=DA=1, α_W=α_DE=0. Eso seria:
    #   IPO_ideal = V_nom * 1 * 1 * (T_turno/T_ciclo) / T_ciclo * 1 * 1
    #             = V_nom * T_turno / T_ciclo²
    # Entonces:
    #   IPO = η_R * η_M * DA * (1-α_W) * (1-α_DE)   ← PROXY LIMPIO equivalente
    #
    # Usamos el proxy directamente (mas robusto, equivalente matematicamente
    # cuando se normaliza).
    ipo = eta_R * eta_M * DA * (1 - alpha_W) * (1 - alpha_DE)
    ipo = float(np.clip(ipo, 0.0, 1.0))

    if not np.isfinite(ipo):
        ipo = 0.0
        notes.append('IPO calculado como 0 por valores no finitos.')

    band = classify_ipo(ipo)

    return IPOResult(
        ipo=round(ipo, 3),
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
        notes=notes,
    )


# ─── DESCRIPCION TEXTUAL PARA EL DASHBOARD ────────────────────────────────────

IPO_DEFINITIONS = {
    'IPO':      ('Indice de Productividad Operativa',
                 'Metrica maestra que combina volumen efectivo, eficiencia de maniobra, '
                 'ciclos por turno y factores de perdida. Rango 0-1.'),
    'V_nom':    ('Volumen nominal del balde',
                 f'Capacidad teorica del balde EX-5600 = {V_NOM_M3} m³.'),
    'eta_R':    ('η_R — Fill Factor de recoleccion',
                 'Fraccion del balde realmente lleno por cucharada. η_R = V_real / V_nom.'),
    'eta_M':    ('η_M — Eficiencia de maniobra',
                 'Tiempo productivo de swing dividido por el tiempo total del ciclo.'),
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
