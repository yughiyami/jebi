# -*- coding: utf-8 -*-
"""
vision_detector.py  —  JEBI Hackathon 2026

Detector visual avanzado v3:
  1. YOLO26 (Ultralytics 2026) para detectar camiones/vehiculos.
     Fallback a YOLOv11 o YOLOv8 si YOLO26 no esta disponible.
  2. Deteccion por COLOR:
     - ID del camion: texto AMARILLO en la parte superior del tanque
     - Balanza (peso): display LED ROJO en la parte inferior
  3. OCR dirigido por ROI de color.
  4. Calculo de posicion del camion en el frame.

Flujo:
  frame  →  YOLO detecta bbox camion  →  dentro del bbox:
    - Zona superior + filtro amarillo HSV → OCR del ID
    - Zona inferior + filtro rojo HSV    → OCR del peso (7-seg)
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

YOLO_MODEL_CANDIDATES = [
    'yolo26s.pt',    # YOLO26 small (2026)
    'yolo26n.pt',
    'yolo11s.pt',
    'yolo11n.pt',
    'yolov8s.pt',
    'yolov8n.pt',
]
VEHICLE_CLASS_IDS    = {7}   # COCO: truck
YOLO_CONFIDENCE      = 0.30
OCR_CONFIDENCE_MIN   = 0.30

# HSV para AMARILLO brillante (ID camion - pintura o vinilo reflectivo)
HSV_YELLOW_LO = np.array([18, 100, 110])
HSV_YELLOW_HI = np.array([38, 255, 255])

# HSV para ROJO (display LED balanza). Rojo cruza 0/180 → 2 rangos
HSV_RED_LO_1  = np.array([0, 130, 110])
HSV_RED_HI_1  = np.array([10, 255, 255])
HSV_RED_LO_2  = np.array([165, 130, 110])
HSV_RED_HI_2  = np.array([180, 255, 255])

# Patterns
TRUCK_ID_PATTERN       = re.compile(r'\b\d{2,4}[A-Z]?\b')
WEIGHT_DISPLAY_PATTERN = re.compile(r'\b\d{1,4}(?:[.,]\d{1,2})?\b')

# Capacidades por modelo
TRUCK_CAPS = {
    'CAT_793F':   218.0,   # 793F 218t
    'EH4000_AC3': 221.0,   # EH4000 221t
    'unknown':    219.0,
}

# Minimo area del contorno amarillo/rojo para ser candidato
MIN_YELLOW_AREA_PX = 150
MIN_RED_AREA_PX    = 80

# ─── STATE (lazy) ─────────────────────────────────────────────────────────────

_YOLO_MODEL = None
_OCR_READER = None


def _get_yolo():
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
            print(f'  [vision] {candidate} fallo: {e}')
            continue
    print('  [vision] ERROR: ningun YOLO pudo cargarse')
    return None


def _get_ocr():
    global _OCR_READER
    if _OCR_READER is not None or not OCR_AVAILABLE:
        return _OCR_READER
    try:
        _OCR_READER = easyocr.Reader(['en'], verbose=False, gpu=False)
    except Exception as e:
        print(f'  [vision] EasyOCR init fallo: {e}')
    return _OCR_READER


# ─── DETECCION YOLO ───────────────────────────────────────────────────────────

def detect_truck_yolo(frame: np.ndarray) -> Optional[Dict]:
    """
    Detecta el camion mas grande con YOLO.
    Returns: { bbox, confidence, class_name, crop } | None
    """
    if frame is None:
        return None
    model = _get_yolo()
    if model is None:
        return None

    try:
        results = model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        if not results:
            return None
        res = results[0]
        if not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
            return None

        candidates = []
        for box in res.boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls[0])
            conf   = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf[0])
            if cls_id in VEHICLE_CLASS_IDS or cls_id == 7:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.asarray(box.xyxy[0])
                x1, y1, x2, y2 = map(int, xyxy[:4])
                area = (x2 - x1) * (y2 - y1)
                candidates.append({
                    'bbox':       (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id':   cls_id,
                    'class_name': res.names.get(cls_id, 'truck') if hasattr(res, 'names') else 'truck',
                    'area':       area,
                })

        if not candidates:
            return None

        # Preferir el de mayor area (camion mas cercano)
        best = max(candidates, key=lambda c: c['area'])
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


# ─── ROI POR COLOR ────────────────────────────────────────────────────────────

def find_yellow_roi(bgr: np.ndarray, min_area: int = MIN_YELLOW_AREA_PX) -> Optional[Tuple[int, int, int, int]]:
    """
    Encuentra la region AMARILLA mas grande en el BGR (para ID del camion).
    Retorna (x, y, w, h) del bounding box del contorno mas grande, o None.
    """
    if bgr is None or bgr.size == 0:
        return None
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
    # Morfologia: cerrar huecos pequenos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not big:
        return None

    # El mas grande — asumimos que es el numero del ID pintado grande
    best = max(big, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    # Padding pequeno
    pad = 6
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(bgr.shape[1] - x, w + pad*2)
    h = min(bgr.shape[0] - y, h + pad*2)
    return (x, y, w, h)


def find_red_roi(bgr: np.ndarray, min_area: int = MIN_RED_AREA_PX) -> Optional[Tuple[int, int, int, int]]:
    """
    Encuentra la region ROJA mas grande (display LED balanza del peso).
    """
    if bgr is None or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, HSV_RED_LO_1, HSV_RED_HI_1)
    m2 = cv2.inRange(hsv, HSV_RED_LO_2, HSV_RED_HI_2)
    mask = m1 | m2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not big:
        return None

    # Preferir el contorno con aspect ratio horizontal (displays son anchos)
    best = None
    best_score = -1
    for c in big:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = w / max(h, 1)
        # Score: mas area + aspect ratio ~2-5 (horizontal)
        score = area * (1.0 if 1.5 <= aspect <= 6.0 else 0.5)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None
    x, y, w, h = best
    pad = 4
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(bgr.shape[1] - x, w + pad*2)
    h = min(bgr.shape[0] - y, h + pad*2)
    return (x, y, w, h)


# ─── OCR DIRIGIDO ─────────────────────────────────────────────────────────────

def ocr_on_yellow(bgr_crop: np.ndarray) -> Tuple[str, float]:
    """
    OCR sobre la region amarilla encontrada en el frame del camion.
    Devuelve (id_str, confianza).
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return 'unknown', 0.0
    reader = _get_ocr()
    if reader is None:
        return 'unknown', 0.0

    roi_bbox = find_yellow_roi(bgr_crop)
    if roi_bbox is None:
        return 'unknown', 0.0
    x, y, w, h = roi_bbox

    # Extraer ROI con padding generoso (contexto ayuda al OCR)
    pad_x = max(10, w // 2)
    pad_y = max(10, h // 2)
    ex = max(0, x - pad_x); ey = max(0, y - pad_y)
    ew = min(bgr_crop.shape[1] - ex, w + 2 * pad_x)
    eh = min(bgr_crop.shape[0] - ey, h + 2 * pad_y)
    roi = bgr_crop[ey:ey + eh, ex:ex + ew]
    if roi.size == 0:
        return 'unknown', 0.0

    # Upscale GRANDE (EasyOCR performa mucho mejor con texto >30px altura)
    scale = max(2.0, 300.0 / max(roi.shape[:2]))
    roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    candidates = []
    try:
        # 1) OCR directo sobre BGR upscaled
        try:
            results_bgr = reader.readtext(roi_up, detail=1,
                                           allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            for (_, text, conf) in results_bgr:
                cleaned = text.strip().upper().replace(' ', '').replace('O', '0')
                matches = TRUCK_ID_PATTERN.findall(cleaned)
                if matches and conf >= OCR_CONFIDENCE_MIN:
                    for m in matches:
                        if 2 <= len(m) <= 4:
                            candidates.append((conf, m))
        except Exception:
            pass

        # 2) OCR sobre gray con CLAHE (contraste realzado)
        try:
            gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            results_gray = reader.readtext(enhanced, detail=1, allowlist='0123456789')
            for (_, text, conf) in results_gray:
                cleaned = text.strip().replace(' ', '').replace('O', '0')
                matches = TRUCK_ID_PATTERN.findall(cleaned)
                if matches and conf >= OCR_CONFIDENCE_MIN:
                    for m in matches:
                        if 2 <= len(m) <= 4:
                            candidates.append((conf, m))
        except Exception:
            pass

        # 3) OCR sobre mask amarillo (ultimo recurso)
        try:
            hsv = cv2.cvtColor(roi_up, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
            results_mask = reader.readtext(mask, detail=1, allowlist='0123456789')
            for (_, text, conf) in results_mask:
                cleaned = text.strip().replace(' ', '').replace('O', '0')
                matches = TRUCK_ID_PATTERN.findall(cleaned)
                if matches and conf >= 0.20:
                    for m in matches:
                        if 2 <= len(m) <= 4:
                            candidates.append((conf * 0.8, m))  # penaliza confianza
        except Exception:
            pass

        if candidates:
            best_conf, best_id = max(candidates, key=lambda x: x[0])
            return best_id, float(min(1.0, best_conf))
    except Exception as e:
        print(f'  [vision] OCR yellow error: {e}')

    return 'unknown', 0.0


def ocr_on_red(bgr_crop: np.ndarray) -> Tuple[str, float, str]:
    """
    OCR sobre la region roja encontrada (display LED balanza).
    Devuelve (peso_str, confianza, unidad).
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return '—', 0.0, 't'
    reader = _get_ocr()
    if reader is None:
        return '—', 0.0, 't'

    roi_bbox = find_red_roi(bgr_crop)
    if roi_bbox is None:
        return '—', 0.0, 't'
    x, y, w, h = roi_bbox

    # Padding para mas contexto
    pad_x = max(8, w // 2)
    pad_y = max(8, h // 2)
    ex = max(0, x - pad_x); ey = max(0, y - pad_y)
    ew = min(bgr_crop.shape[1] - ex, w + 2 * pad_x)
    eh = min(bgr_crop.shape[0] - ey, h + 2 * pad_y)
    roi = bgr_crop[ey:ey + eh, ex:ex + ew]
    if roi.size == 0:
        return '—', 0.0, 't'

    # Upscale FUERTE (displays LED son pequenos)
    scale = max(3.0, 400.0 / max(roi.shape[:2]))
    roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    candidates = []
    try:
        # 1) OCR sobre BGR directo upscaled
        try:
            results_bgr = reader.readtext(roi_up, detail=1, allowlist='0123456789.,-')
            for (_, text, conf) in results_bgr:
                cleaned = text.strip().replace(' ', '').replace(',', '.')
                matches = WEIGHT_DISPLAY_PATTERN.findall(cleaned)
                if matches and conf >= 0.25:
                    for m in matches:
                        try:
                            val = float(m)
                            if 0 < val <= 500:
                                candidates.append((conf, m, val))
                        except ValueError:
                            continue
        except Exception:
            pass

        # 2) OCR sobre gray con threshold Otsu
        try:
            gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results_thr = reader.readtext(thr, detail=1, allowlist='0123456789.,-')
            for (_, text, conf) in results_thr:
                cleaned = text.strip().replace(' ', '').replace(',', '.')
                matches = WEIGHT_DISPLAY_PATTERN.findall(cleaned)
                if matches and conf >= 0.25:
                    for m in matches:
                        try:
                            val = float(m)
                            if 0 < val <= 500:
                                candidates.append((conf, m, val))
                        except ValueError:
                            continue
        except Exception:
            pass

        # 3) OCR sobre mask rojo (aislando display LED)
        try:
            hsv = cv2.cvtColor(roi_up, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, HSV_RED_LO_1, HSV_RED_HI_1)
            m2 = cv2.inRange(hsv, HSV_RED_LO_2, HSV_RED_HI_2)
            red_mask = m1 | m2
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            red_mask = cv2.dilate(red_mask, kernel, iterations=2)
            results_mask = reader.readtext(red_mask, detail=1, allowlist='0123456789.,-')
            for (_, text, conf) in results_mask:
                cleaned = text.strip().replace(' ', '').replace(',', '.')
                matches = WEIGHT_DISPLAY_PATTERN.findall(cleaned)
                if matches and conf >= 0.20:
                    for m in matches:
                        try:
                            val = float(m)
                            if 0 < val <= 500:
                                candidates.append((conf * 0.85, m, val))
                        except ValueError:
                            continue
        except Exception:
            pass

        if candidates:
            best = max(candidates, key=lambda x: x[0])
            conf, val_str, val_float = best
            unit = 't' if val_float >= 20 else '%'
            return val_str, float(min(1.0, conf)), unit
    except Exception as e:
        print(f'  [vision] OCR red error: {e}')

    return '—', 0.0, 't'


# ─── INFERENCIA MODELO CAMION ─────────────────────────────────────────────────

def infer_truck_model(bbox_w_h_ratio: float, confidence: float) -> str:
    if confidence < 0.3:
        return 'unknown'
    if 1.55 <= bbox_w_h_ratio <= 1.90:
        return 'CAT_793F'
    elif 1.25 <= bbox_w_h_ratio <= 1.54:
        return 'EH4000_AC3'
    return 'unknown'


# ─── FLUJO COMPLETO POR FRAME ─────────────────────────────────────────────────

def analyze_frame(frame: np.ndarray,
                  timestamp_s: float,
                  cycle_id: int,
                  event_type: str = 'dump') -> Dict:
    """
    Analisis completo:
      1. YOLO detecta camion
      2. Dentro del crop:
         - ROI amarillo (superior) → OCR ID
         - ROI rojo (inferior)     → OCR peso
      3. Calcula position score
      4. Devuelve frame anotado en base64
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
        'yellow_roi':        None,
        'red_roi':           None,
        'position_score':    0.0,
        'method':            'none',
        'frame_b64':         '',
    }

    if frame is None:
        return result

    # ── 1. YOLO ───────────────────────────────────────────────────────────────
    det = detect_truck_yolo(frame)
    annotated = frame.copy()

    if det is None:
        # Fallback: intentar color directamente en todo el frame
        # (util si YOLO falla con camion cercano)
        yellow_bbox = find_yellow_roi(frame, min_area=300)
        red_bbox    = find_red_roi(frame, min_area=100)
        if yellow_bbox is None and red_bbox is None:
            _annotate_no_detection(annotated)
            result['frame_b64'] = _encode_frame(annotated)
            return result
        else:
            # Armar bbox sintetico desde union de rois de color
            x_min = frame.shape[1]; y_min = frame.shape[0]; x_max = 0; y_max = 0
            for bb in (yellow_bbox, red_bbox):
                if bb:
                    x, y, w, h = bb
                    x_min = min(x_min, x); y_min = min(y_min, y)
                    x_max = max(x_max, x + w); y_max = max(y_max, y + h)
            # Expandir
            h_fr, w_fr = frame.shape[:2]
            x_min = max(0, x_min - 50); y_min = max(0, y_min - 50)
            x_max = min(w_fr, x_max + 50); y_max = min(h_fr, y_max + 50)
            det = {
                'bbox':       (x_min, y_min, x_max, y_max),
                'confidence': 0.4,
                'class_name': 'truck-color',
                'crop':       frame[y_min:y_max, x_min:x_max].copy(),
            }
            result['method'] = 'color_only'

    result['truck_detected'] = True
    result['bbox']           = det['bbox']
    if result['method'] == 'none':
        result['method'] = 'yolo26+color+ocr'

    x1, y1, x2, y2 = det['bbox']
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    ratio  = bbox_w / max(bbox_h, 1)
    result['truck_model'] = infer_truck_model(ratio, det['confidence'])
    result['capacity_t']  = TRUCK_CAPS.get(result['truck_model'], TRUCK_CAPS['unknown'])

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (74, 158, 255), 3)
    cv2.putText(annotated, f"{result['truck_model']} ({det['confidence']:.2f})",
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (74, 158, 255), 2)

    # ── 2. OCR sobre crop del camion ──────────────────────────────────────────
    crop = det['crop']

    # ID amarillo — buscar en la MITAD SUPERIOR del tanque (donde esta pintado)
    ch, cw = crop.shape[:2]
    top_half = crop[:int(ch * 0.75), :]  # hasta 75% para capturar ID en tanque

    tid, conf_id = ocr_on_yellow(top_half)
    result['truck_id_ocr']  = tid
    result['confidence_id'] = conf_id

    # Localizar yellow_roi para dibujarlo en el annotated
    y_bbox = find_yellow_roi(top_half)
    if y_bbox is not None:
        yx, yy, yw, yh = y_bbox
        ax1, ay1 = x1 + yx, y1 + yy
        ax2, ay2 = ax1 + yw, ay1 + yh
        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 215, 255), 2)
        cv2.putText(annotated, f"ID: {tid}",
                    (ax1, max(ay1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
        result['yellow_roi'] = (int(ax1), int(ay1), int(ax2), int(ay2))

    # Balanza roja — buscar en la MITAD INFERIOR (display cerca de cabina/chasis)
    bottom_half = crop[int(ch * 0.25):, :]
    weight_str, conf_w, unit = ocr_on_red(bottom_half)
    result['weight_reading']    = weight_str
    result['confidence_weight'] = conf_w
    result['weight_unit']       = unit

    r_bbox = find_red_roi(bottom_half)
    if r_bbox is not None:
        rx, ry, rw, rh = r_bbox
        ax1 = x1 + rx
        ay1 = y1 + int(ch * 0.25) + ry
        ax2 = ax1 + rw
        ay2 = ay1 + rh
        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 0, 255), 2)
        if weight_str != '—':
            cv2.putText(annotated, f"Peso: {weight_str} {unit}",
                        (ax1, min(annotated.shape[0] - 5, ay2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        result['red_roi'] = (int(ax1), int(ay1), int(ax2), int(ay2))

    # ── 3. Position score ────────────────────────────────────────────────────
    fh, fw = frame.shape[:2]
    bbox_cx = (x1 + x2) / 2
    ideal_cx = fw * 0.5
    offset_pct = abs(bbox_cx - ideal_cx) / fw * 100
    pos_score = max(0.0, 100.0 - offset_pct * 2.5)
    bbox_area_pct = (bbox_w * bbox_h) / (fw * fh) * 100
    if bbox_area_pct < 8:
        pos_score *= 0.6
    elif bbox_area_pct > 65:
        pos_score *= 0.5
    result['position_score'] = round(pos_score, 1)

    # Zona ideal
    cv2.rectangle(annotated,
                  (int(fw * 0.35), int(fh * 0.4)),
                  (int(fw * 0.65), int(fh * 0.95)),
                  (63, 185, 80), 2, cv2.LINE_AA)
    color = (63, 185, 80) if pos_score >= 70 else (248, 81, 73)
    cv2.putText(annotated, f"Pos: {pos_score:.0f}%  t={timestamp_s:.1f}s",
                (10, fh - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── 4. Encode ─────────────────────────────────────────────────────────────
    result['frame_b64'] = _encode_frame(annotated)
    return result


# ─── HELPERS DE ENCODE ────────────────────────────────────────────────────────

def _annotate_no_detection(frame: np.ndarray):
    h, w = frame.shape[:2]
    cv2.putText(frame, 'Sin camion detectado',
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)


def _encode_frame(frame: np.ndarray, max_w: int = 640) -> str:
    try:
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return base64.b64encode(buf).decode('utf-8')
    except Exception:
        return ''


# ─── DUST INDEX ───────────────────────────────────────────────────────────────

def compute_dust_index(frame: np.ndarray) -> float:
    """Proxy de polvo en el frame (0-100)."""
    if frame is None:
        return 0.0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        var  = float(np.var(gray))
        mean = float(np.mean(gray))
        gray_dom = 1.0 - (abs(mean - 128) / 128)
        var_inv  = max(0.0, 1.0 - var / 3000.0)
        dust = (gray_dom * 0.5 + var_inv * 0.5) * 100
        return float(max(0.0, min(100.0, dust)))
    except Exception:
        return 0.0


# ─── BATCH ────────────────────────────────────────────────────────────────────

def batch_analyze_frames(frames_by_ts: Dict[float, np.ndarray],
                          cycle_map: Dict[float, int],
                          event_map: Optional[Dict[float, str]] = None) -> List[Dict]:
    out = []
    for ts, frame in frames_by_ts.items():
        cid = cycle_map.get(ts, 0)
        ev  = event_map.get(ts, 'dump') if event_map else 'dump'
        out.append(analyze_frame(frame, ts, cid, ev))
    return out
