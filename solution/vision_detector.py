# -*- coding: utf-8 -*-
"""
vision_detector.py  —  JEBI Hackathon 2026

Detector visual avanzado:
  1. YOLO26 (Ultralytics 2026) para detectar camiones/vehiculos
     Fallback a YOLOv8/v11 si YOLO26 no esta disponible.
  2. OCR especializado para numero de identificacion del camion
  3. OCR con preprocesamiento especifico para displays 7-segmentos
     del medidor de peso frontal.
  4. Calculo de posicion ideal vs real del camion (zona de colocacion).

Decisiones de diseño:
  - Modelo YOLO se carga lazy (primera vez)
  - Si YOLO falla/no esta, fallback a deteccion visual simple
  - OCR es EasyOCR con preprocesing dedicado por tipo de texto
  - Todo es tolerante a fallos: ningun metodo crashea, devuelve 'unknown'
"""

import os
import cv2
import numpy as np
import base64
import re
from typing import Dict, List, Optional, Tuple

# ─── IMPORTS OPCIONALES ───────────────────────────────────────────────────────

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print('  [vision] Ultralytics no disponible, usando fallback visual')

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ─── CONFIGURACION ────────────────────────────────────────────────────────────

# Ruta del modelo YOLO26 (descarga automatica en primera ejecucion)
YOLO_MODEL_CANDIDATES = [
    'yolo26s.pt',   # YOLO26 small (2026, mas preciso)
    'yolo26n.pt',   # YOLO26 nano (mas rapido, fallback)
    'yolo11s.pt',   # YOLO11 small (2024, fallback estable)
    'yolo11n.pt',   # YOLO11 nano (fallback mas ligero)
    'yolov8s.pt',   # YOLOv8 small (fallback)
    'yolov8n.pt',   # YOLOv8 nano (fallback mas robusto)
]

# Clases COCO relevantes para camiones mineros
# COCO class IDs: 7 = truck, 2 = car (no aplica), 5 = bus
VEHICLE_CLASS_IDS = {7}   # truck

# Umbrales
YOLO_CONFIDENCE_THRESH = 0.35
OCR_CONFIDENCE_THRESH = 0.45

# Patterns de texto
TRUCK_ID_PATTERN      = re.compile(r'\b\d{2,5}\b')
WEIGHT_DISPLAY_PATTERN = re.compile(r'\b\d{1,4}(?:[.,]\d{1,2})?\b')

# Capacidades por modelo
TRUCK_CAPS = {
    'CAT_793F':   221.0,
    'EH4000_AC3': 218.0,
    'unknown':    219.0,
}


# ─── STATE (lazy) ─────────────────────────────────────────────────────────────

_YOLO_MODEL = None
_OCR_READER = None


# ─── CARGA LAZY DEL MODELO YOLO ───────────────────────────────────────────────

def _get_yolo():
    """Carga lazy del modelo YOLO. Intenta YOLO26 -> YOLO11 -> YOLOv8."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None or not YOLO_AVAILABLE:
        return _YOLO_MODEL

    for candidate in YOLO_MODEL_CANDIDATES:
        try:
            print(f'  [vision] Cargando {candidate}...')
            _YOLO_MODEL = YOLO(candidate)
            print(f'  [vision] Modelo cargado: {candidate}')
            return _YOLO_MODEL
        except Exception as e:
            print(f'  [vision] {candidate} no disponible: {e}')
            continue

    print('  [vision] ERROR: ningun modelo YOLO pudo cargarse')
    return None


def _get_ocr():
    global _OCR_READER
    if _OCR_READER is not None or not OCR_AVAILABLE:
        return _OCR_READER
    try:
        _OCR_READER = easyocr.Reader(['en'], verbose=False, gpu=False)
    except Exception as e:
        print(f'  [vision] EasyOCR no pudo inicializar: {e}')
    return _OCR_READER


# ─── DETECCION DE CAMION CON YOLO ─────────────────────────────────────────────

def detect_truck_yolo(frame: np.ndarray) -> Optional[Dict]:
    """
    Corre YOLO sobre el frame y devuelve el bounding box del camion
    con mayor confianza.

    Returns:
        {
          'bbox': (x1, y1, x2, y2),
          'confidence': float,
          'class_name': str,
          'crop': np.ndarray (ROI del camion)
        } | None
    """
    if frame is None:
        return None
    model = _get_yolo()
    if model is None:
        return None

    try:
        # YOLO inference — tolerante a distintas versiones
        results = model(frame, conf=YOLO_CONFIDENCE_THRESH, verbose=False)
        if not results:
            return None
        res = results[0]
        if not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
            return None

        # Filtrar por clases relevantes (truck)
        candidates = []
        for box in res.boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls[0])
            conf   = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf[0])
            if cls_id in VEHICLE_CLASS_IDS or cls_id == 7:  # truck
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.asarray(box.xyxy[0])
                x1, y1, x2, y2 = map(int, xyxy[:4])
                candidates.append({
                    'bbox':       (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id':   cls_id,
                    'class_name': res.names.get(cls_id, 'truck') if hasattr(res, 'names') else 'truck',
                })

        if not candidates:
            return None

        # El de mayor confianza
        best = max(candidates, key=lambda c: c['confidence'])
        x1, y1, x2, y2 = best['bbox']
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        best['crop'] = frame[y1:y2, x1:x2].copy()
        return best
    except Exception as e:
        print(f'  [vision] YOLO error: {e}')
        return None


# ─── OCR DE ID DE CAMION ──────────────────────────────────────────────────────

def ocr_truck_id(frame_or_roi: np.ndarray) -> Tuple[str, float]:
    """
    OCR del numero de identificacion del camion.
    Busca patrones de 2-5 digitos con alta confianza.

    Devuelve (id_detectado, confianza 0-1).
    """
    if frame_or_roi is None:
        return 'unknown', 0.0
    reader = _get_ocr()
    if reader is None:
        return 'unknown', 0.0

    try:
        # Preprocesamiento: upscale si es muy chiquito
        h, w = frame_or_roi.shape[:2]
        target = frame_or_roi
        if max(h, w) < 600:
            scale = 800 / max(h, w)
            target = cv2.resize(frame_or_roi, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)

        # Mejorar contraste
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if len(target.shape) == 3 else target
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        results = reader.readtext(enhanced, detail=1,
                                   allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
        candidates = []
        for (bbox, text, conf) in results:
            cleaned = text.strip().upper().replace(' ', '')
            nums = TRUCK_ID_PATTERN.findall(cleaned)
            if nums and conf >= OCR_CONFIDENCE_THRESH:
                # Prefer numeros mas largos (mas especificos)
                candidates.append((conf * len(nums[0]) / 4.0, nums[0], conf))

        if candidates:
            candidates.sort(reverse=True)
            _, best_num, best_conf = candidates[0]
            return best_num, float(best_conf)
    except Exception as e:
        print(f'  [vision] OCR ID error: {e}')
    return 'unknown', 0.0


# ─── OCR DE DISPLAY DE PESO (7-SEGMENTOS) ─────────────────────────────────────

def ocr_weight_display(frame_or_roi: np.ndarray) -> Tuple[str, float, str]:
    """
    OCR especializado para displays de 7-segmentos del medidor de peso
    en el frontal del camion.

    Preprocesamiento:
      1. Conversion a gris
      2. Gaussian blur para reducir ruido
      3. Threshold adaptativo (displays tipicamente son brillantes sobre oscuro)
      4. Dilatacion para conectar segmentos sueltos
      5. OCR con allowlist de digitos

    Devuelve (valor_str, confianza 0-1, unidad).
    """
    if frame_or_roi is None:
        return '—', 0.0, 't'
    reader = _get_ocr()
    if reader is None:
        return '—', 0.0, 't'

    try:
        h, w = frame_or_roi.shape[:2]
        # Zona frontal superior: donde suelen estar los displays
        # Si ya es un ROI (por ejemplo del bbox del camion), tomamos el tercio superior
        upper = frame_or_roi[:int(h * 0.45), :]

        if upper.size == 0:
            return '—', 0.0, 't'

        # Preprocesamiento para 7-segmentos
        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY) if len(upper.shape) == 3 else upper
        # Upscale para facilitar lectura
        scale = 3.0 if max(gray.shape) < 400 else 2.0
        gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Suavizar ruido
        blur = cv2.GaussianBlur(gray_up, (5, 5), 0)

        # Displays rojos/verdes/amarillos sobre fondo oscuro: threshold invertido
        _, thr_dark = cv2.threshold(blur, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Tambien intentar threshold normal por si el display es oscuro sobre claro
        _, thr_light = cv2.threshold(blur, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        candidates_by_thresh = []
        for variant_name, variant in [('dark', thr_dark), ('light', thr_light)]:
            # Dilatacion para conectar segmentos
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated = cv2.dilate(variant, kernel, iterations=1)

            try:
                results = reader.readtext(dilated, detail=1,
                                           allowlist='0123456789.,-')
                for (_, text, conf) in results:
                    cleaned = text.strip().replace(' ', '').replace(',', '.')
                    nums = WEIGHT_DISPLAY_PATTERN.findall(cleaned)
                    if nums and conf >= 0.3:
                        # El display de peso tipicamente muestra 2-3 digitos enteros
                        val_str = nums[0]
                        try:
                            val_float = float(val_str)
                            # Sanidad: peso realista 0-400t
                            if 0 < val_float <= 400:
                                candidates_by_thresh.append(
                                    (conf, val_str, val_float, variant_name)
                                )
                        except ValueError:
                            continue
            except Exception:
                continue

        if candidates_by_thresh:
            candidates_by_thresh.sort(reverse=True)
            _, best_str, best_val, _ = candidates_by_thresh[0]
            # Unidad: >100 sugiere toneladas, <100 sugiere porcentaje
            unit = 't' if best_val >= 20 else '%'
            return best_str, float(candidates_by_thresh[0][0]), unit
    except Exception as e:
        print(f'  [vision] OCR weight error: {e}')
    return '—', 0.0, 't'


# ─── INFERENCIA DE MODELO ─────────────────────────────────────────────────────

def infer_truck_model(bbox_hw_ratio: float, confidence: float) -> str:
    """
    Heuristica de modelo por aspect ratio del bbox.
    CAT 793F: ~1.6-1.8 (mas alargado)
    EH4000 AC-3: ~1.3-1.5 (mas cuadrado)
    """
    if confidence < 0.3:
        return 'unknown'
    if 1.55 <= bbox_hw_ratio <= 1.90:
        return 'CAT_793F'
    elif 1.25 <= bbox_hw_ratio <= 1.54:
        return 'EH4000_AC3'
    return 'unknown'


# ─── PIPELINE COMPLETO POR FRAME ──────────────────────────────────────────────

def analyze_frame(frame: np.ndarray,
                  timestamp_s: float,
                  cycle_id: int,
                  event_type: str = 'dump') -> Dict:
    """
    Pipeline completo sobre un frame:
      1. YOLO detecta el camion
      2. OCR del ID sobre ROI
      3. OCR del medidor de peso sobre ROI superior
      4. Estima posicion del camion en el frame (bbox center)
      5. Calcula offset vs zona ideal (centro del frame derecho)
    """
    result = {
        'timestamp_s':       round(timestamp_s, 2),
        'cycle_id':          cycle_id,
        'event_type':        event_type,
        'truck_detected':    False,
        'truck_id_ocr':      'unknown',
        'confidence_id':     0.0,
        'weight_reading':    '—',
        'confidence_weight': 0.0,
        'weight_unit':       't',
        'truck_model':       'unknown',
        'capacity_t':        TRUCK_CAPS['unknown'],
        'bbox':              None,
        'position_score':    0.0,      # 0-100, que tan bien posicionado
        'method':            'none',
        'frame_b64':         '',
    }

    if frame is None:
        return result

    # ── 1. YOLO ───────────────────────────────────────────────────────────────
    det = detect_truck_yolo(frame)
    annotated = frame.copy()

    if det is not None:
        result['truck_detected'] = True
        result['bbox']           = det['bbox']
        result['method']         = 'yolo26'

        x1, y1, x2, y2 = det['bbox']
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        ratio  = bbox_w / max(bbox_h, 1)
        result['truck_model'] = infer_truck_model(ratio, det['confidence'])
        result['capacity_t']  = TRUCK_CAPS.get(result['truck_model'], TRUCK_CAPS['unknown'])

        # Dibujar bbox en el frame
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (74, 158, 255), 3)
        cv2.putText(annotated, f"{result['truck_model']} ({det['confidence']:.2f})",
                    (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (74, 158, 255), 2)

        # ── 2. OCR sobre crop del camion ──────────────────────────────────────
        crop = det['crop']
        tid, conf_id = ocr_truck_id(crop)
        result['truck_id_ocr']  = tid
        result['confidence_id'] = conf_id

        # ── 3. OCR del medidor de peso (zona frontal superior del camion) ────
        # Tomamos la mitad derecha superior del crop, donde suele estar el display
        ch, cw = crop.shape[:2]
        front_roi = crop[:int(ch * 0.55), int(cw * 0.45):]
        weight_str, conf_w, unit = ocr_weight_display(front_roi)
        result['weight_reading']    = weight_str
        result['confidence_weight'] = conf_w
        result['weight_unit']       = unit

        # Anotar OCR results
        if tid != 'unknown':
            cv2.putText(annotated, f"ID: {tid}",
                        (x1, min(frame.shape[0] - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        if weight_str != '—':
            cv2.putText(annotated, f"W: {weight_str} {unit}",
                        (x1, min(frame.shape[0] - 10, y2 + 48)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (63, 185, 80), 2)

        # ── 4. Score de posicionamiento ───────────────────────────────────────
        # Zona ideal: bbox centrado horizontalmente en el 40-60% del frame
        fh, fw = frame.shape[:2]
        bbox_cx = (x1 + x2) / 2
        ideal_cx = fw * 0.5
        offset_pct = abs(bbox_cx - ideal_cx) / fw * 100  # 0 = perfecto, 50 = borde
        pos_score = max(0.0, 100.0 - offset_pct * 2.5)
        # Penalizar si bbox es muy chiquito (camion muy lejos) o muy grande (muy cerca)
        bbox_area_pct = (bbox_w * bbox_h) / (fw * fh) * 100
        if bbox_area_pct < 8:    pos_score *= 0.6      # muy lejos
        elif bbox_area_pct > 60: pos_score *= 0.5      # muy cerca
        result['position_score'] = round(pos_score, 1)

        # Dibujar zona ideal
        cv2.rectangle(annotated,
                      (int(fw * 0.35), int(fh * 0.4)),
                      (int(fw * 0.65), int(fh * 0.95)),
                      (63, 185, 80), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Pos: {pos_score:.0f}%",
                    (10, fh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (63, 185, 80) if pos_score >= 70 else (248, 81, 73), 2)

    # ── 5. Codificar frame anotado a base64 para el dashboard ────────────────
    try:
        h, w = annotated.shape[:2]
        if w > 480:
            scale = 480 / w
            annotated = cv2.resize(annotated, None, fx=scale, fy=scale)
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result['frame_b64'] = base64.b64encode(buf).decode('utf-8')
    except Exception:
        pass

    return result


def batch_analyze_frames(frames_by_ts: Dict[float, np.ndarray],
                          cycle_map: Dict[float, int],
                          event_map: Optional[Dict[float, str]] = None) -> List[Dict]:
    """
    Corre analyze_frame sobre multiples frames.

    Args:
        frames_by_ts: {timestamp: frame_bgr}
        cycle_map:    {timestamp: cycle_id}
        event_map:    {timestamp: event_type} (dump|start|other), opcional
    """
    out = []
    for ts, frame in frames_by_ts.items():
        cid = cycle_map.get(ts, 0)
        ev  = (event_map.get(ts, 'dump') if event_map else 'dump')
        result = analyze_frame(frame, ts, cid, ev)
        out.append(result)
    return out


# ─── DUST INDEX (para BPMN node "Aire") ───────────────────────────────────────

def compute_dust_index(frame: np.ndarray) -> float:
    """
    Indice de polvo en el frame (0-100).
    Heuristica: frames con mucho polvo tienen baja varianza y histograma sesgado
    hacia gris medio.
    """
    if frame is None:
        return 0.0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        # Baja varianza = cielo/polvo uniforme
        var = float(np.var(gray))
        # Media cerca de 128 = gris (polvo)
        mean = float(np.mean(gray))
        # Score: alta media + baja varianza = mucho polvo
        gray_dominance = 1.0 - (abs(mean - 128) / 128)
        var_inv = max(0.0, 1.0 - var / 3000.0)
        dust = (gray_dominance * 0.5 + var_inv * 0.5) * 100
        return float(max(0.0, min(100.0, dust)))
    except Exception:
        return 0.0
