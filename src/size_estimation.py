import cv2
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import gc  # Para garbage collection

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CSV = ANNOTATIONS_DIR / "fruit_metadata.csv"

QUALITIES = ["good", "bad", "regular"]
FRUITS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]

model = YOLO("yolov8n.pt")

def is_valid_crop(crop, uniformity_threshold=0.85):
    if crop is None or crop.size == 0:
        return False
    
    h, w = crop.shape[:2]
    if h < 30 or w < 30:
        return False
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_hue = hist_hue / hist_hue.sum()
    
    if hist_hue.max() > uniformity_threshold:
        return False
    
    saturation = hsv[:, :, 1]
    sat_variance = np.var(saturation)
    if sat_variance < 100:
        return False
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (h * w)
    
    if edge_ratio < 0.02:
        return False
    
    return True

def get_fruit_yolo_class(fruit_name):
    mapping = {
        "apple": 47,
        "orange": 48,
        "banana": 52,
    }
    return mapping.get(fruit_name, None)

def detect_fruits_yolo_custom(image, fruit_name, confidence=0.3):
    results = model(image, conf=confidence, iou=0.3)
    target_class = get_fruit_yolo_class(fruit_name)
    
    boxes = []
    if target_class is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == target_class and conf > confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
    return boxes

def detect_fruits_saturation_based(image, min_area=800):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    sat_mask = saturation > 50
    val_mask = value > 40
    
    combined_mask = (sat_mask & val_mask).astype(np.uint8) * 255
    
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if 0.3 < aspect_ratio < 4.0:
                padding = 15
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2*padding)
                h = min(image.shape[0] - y, h + 2*padding)
                boxes.append((x, y, x + w, y + h))
    
    return boxes

def detect_fruits_edge_clustering_fast(image, min_area=800):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    if max(height, width) > 1200:
        scale = 800 / max(height, width)
        new_w = int(width * scale)
        new_h = int(height * scale)
        gray = cv2.resize(gray, (new_w, new_h))
        scale_back = 1.0 / scale
    else:
        scale_back = 1.0
    
    edges = cv2.Canny(gray, 30, 100)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area * (scale_back ** 2):
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if 0.3 < aspect_ratio < 4.0:
                x = int(x * scale_back)
                y = int(y * scale_back)
                w = int(w * scale_back)
                h = int(h * scale_back)
                padding = 15
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2*padding)
                h = min(image.shape[0] - y, h + 2*padding)
                boxes.append((x, y, x + w, y + h))
    
    return boxes

def detect_fruits_watershed_simple(image, min_area=800):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if 0.3 < aspect_ratio < 4.0:
                padding = 15
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2*padding)
                h = min(image.shape[0] - y, h + 2*padding)
                boxes.append((x, y, x + w, y + h))
    
    return boxes

def non_max_suppression(boxes, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(areas)
    
    keep = []
    while len(idxs) > 0:
        i = idxs[-1]
        keep.append(i)
        idxs = idxs[:-1]
        
        if len(idxs) == 0:
            break
        
        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[idxs] - intersection)
        
        idxs = idxs[iou <= iou_threshold]
    
    return [tuple(boxes[i]) for i in keep]

def process_quality_and_fruit(quality, fruit):
    raw_folder = RAW_DIR / quality / f"{fruit}_{quality}"
    
    if not raw_folder.exists():
        print(f"No existe: {raw_folder}")
        return []
    
    output_folder = PROCESSED_DIR / quality / fruit
    output_folder.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(raw_folder.glob(ext))
    
    if not image_files:
        print(f"No hay imágenes en {raw_folder}")
        return []
    
    metadata = []
    discarded_count = 0
    
    for idx, img_path in enumerate(tqdm(image_files, desc=f"{quality}/{fruit}")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error leyendo: {img_path}")
            continue
        
        height, width = img.shape[:2]
        boxes = []
        
        if fruit in ["apple", "orange", "banana"]:
            boxes = detect_fruits_yolo_custom(img, fruit, confidence=0.3)
        
        if len(boxes) == 0:
            boxes = detect_fruits_saturation_based(img, min_area=800)
        
        if len(boxes) == 0:
            boxes = detect_fruits_edge_clustering_fast(img, min_area=800)
        
        if len(boxes) == 0:
            boxes = detect_fruits_watershed_simple(img, min_area=800)
        
        if len(boxes) == 0:
            boxes = [(0, 0, width, height)]
        
        boxes = non_max_suppression(boxes, iou_threshold=0.4)
        
        num_fruits = len(boxes)
        img_pixels = width * height
        
        areas_raw = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes]
        max_area_raw = max(areas_raw) if areas_raw else 1
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue
            
            fruit_crop = img[y1:y2, x1:x2]
            w_px = x2 - x1
            h_px = y2 - y1
            area_px = w_px * h_px
            
            if not is_valid_crop(fruit_crop, uniformity_threshold=0.85):
                discarded_count += 1
                continue
            
            fruit_crop = cv2.resize(fruit_crop, (224, 224))
            
            output_name = f"{img_path.stem}_crop{i}.jpg"
            output_path = output_folder / output_name
            cv2.imwrite(str(output_path), fruit_crop)
            
            relative_size = area_px / max_area_raw if max_area_raw > 0 else 1.0
            coverage_ratio = area_px / img_pixels if img_pixels > 0 else 0
            
            metadata.append({
                "original_image": str(img_path.relative_to(RAW_DIR)),
                "quality": quality,
                "fruit": fruit,
                "crop_path": str(output_path.relative_to(BASE_DIR)),
                "x": x1, "y": y1,
                "width_px": w_px,
                "height_px": h_px,
                "area_px": area_px,
                "aspect_ratio": w_px / h_px if h_px > 0 else 0,
                "num_fruits_in_image": num_fruits,
                "relative_size_in_image": relative_size,
                "coverage_ratio": coverage_ratio,
                "img_width": width,
                "img_height": height
            })
        
        if idx % 10 == 0:
            gc.collect()
    
    if discarded_count > 0:
        print(f"  Descartados: {discarded_count} crops no válidos")
    
    return metadata

def main():
    print("Iniciando procesamiento de frutas...")
    all_metadata = []
    
    for quality in QUALITIES:
        for fruit in FRUITS:
            print(f"\nProcesando {quality}/{fruit}...")
            metadata = process_quality_and_fruit(quality, fruit)
            all_metadata.extend(metadata)
            gc.collect()
    
    if not all_metadata:
        print("No se procesó ninguna imagen. Revisa las rutas.")
        return
    
    df = pd.DataFrame(all_metadata)
    
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nProcesamiento completo!")
    print(f"Total recortes válidos: {len(df)}")
    print(f"CSV guardado en: {METADATA_CSV}")
    print(f"Recortes en: {PROCESSED_DIR}")
    
    print("\nResumen por calidad y fruta:")
    summary = df.groupby(["quality", "fruit"]).size().unstack(fill_value=0)
    print(summary)

if __name__ == "__main__":
    main()