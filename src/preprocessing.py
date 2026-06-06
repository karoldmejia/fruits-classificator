import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
METADATA_CSV = ANNOTATIONS_DIR / "fruit_metadata.csv"
OUTPUT_CSV = ANNOTATIONS_DIR / "preprocessed_metadata.csv"

TARGET_SIZE = (224, 224)
COLOR_PRIMARY = '#FF5E8A'

def normalize_size_category(df):
    """Asigna tamaño basado en área normalizada (percentiles por fruta)"""
    df['normalized_area'] = df['area_px'] / (df['img_width'] * df['img_height'])
    df['size_category'] = 'mediano'
    
    for fruit in df['fruit'].unique():
        fruit_mask = df['fruit'] == fruit
        areas = df.loc[fruit_mask, 'normalized_area']
        
        if len(areas) > 10:
            p33 = areas.quantile(0.33)
            p66 = areas.quantile(0.66)
            
            df.loc[fruit_mask & (df['normalized_area'] < p33), 'size_category'] = 'pequeño'
            df.loc[fruit_mask & (df['normalized_area'] > p66), 'size_category'] = 'grande'
    
    return df

def apply_class_balancing(df):
    """Identifica clases con desbalance crítico (IR > 3)"""
    imbalance_alert = []
    
    for fruit in df['fruit'].unique():
        fruit_data = df[df['fruit'] == fruit]
        class_counts = fruit_data['quality'].value_counts()
        
        if len(class_counts) > 1:
            ir = class_counts.max() / class_counts.min()
            if ir > 3:
                min_class = class_counts.idxmin()
                imbalance_alert.append({
                    'fruit': fruit,
                    'IR': ir,
                    'minority_class': min_class,
                    'minority_count': class_counts[min_class],
                    'majority_count': class_counts.max()
                })
    
    return imbalance_alert

def get_augmentation_pipeline():
    """Data augmentation específico para clases minoritarias"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.4)
    ])

def augment_minority_classes(df, metadata):
    """Aplica augmentation a clases minoritarias para balancear"""
    imbalance_alert = apply_class_balancing(metadata)
    augmented_records = []
    
    for alert in imbalance_alert:
        fruit = alert['fruit']
        minority_class = alert['minority_class']
        target_count = alert['majority_count']
        current_count = alert['minority_count']
        needed = target_count - current_count
        
        if needed <= 0:
            continue
        
        print(f"Balanceando {fruit}/{minority_class}: {current_count} -> {target_count} (+{needed})")
        
        minority_samples = metadata[(metadata['fruit'] == fruit) & 
                                     (metadata['quality'] == minority_class)]
        
        aug_pipeline = get_augmentation_pipeline()
        samples_needed = needed
        samples_per_original = max(1, needed // len(minority_samples))
        
        for idx, row in minority_samples.iterrows():
            img_path = BASE_DIR / row['crop_path']
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for aug_idx in range(samples_per_original):
                if len(augmented_records) >= samples_needed:
                    break
                
                augmented = aug_pipeline(image=img_rgb)
                aug_img = augmented['image']
                
                output_name = f"{Path(row['crop_path']).stem}_aug{aug_idx}.jpg"
                output_path = PROCESSED_DIR / row['quality'] / fruit / output_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), aug_img_bgr)
                
                new_record = row.to_dict()
                new_record['crop_path'] = str(output_path.relative_to(BASE_DIR))
                new_record['augmented'] = True
                augmented_records.append(new_record)
            
            if len(augmented_records) >= samples_needed:
                break
    
    return augmented_records

def extract_hsv_features(img_path):
    """Extrae características HSV robustas a iluminación"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    return {
        'hue_mean': hsv[:,:,0].mean(),
        'hue_std': hsv[:,:,0].std(),
        'saturation_mean': hsv[:,:,1].mean(),
        'saturation_std': hsv[:,:,1].std(),
        'value_mean': hsv[:,:,2].mean(),
        'value_std': hsv[:,:,2].std()
    }

def normalize_pixels(image):
    """Normalización de píxeles a [0,1] y estandarización"""
    img_normalized = image.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet stats
    std = np.array([0.229, 0.224, 0.225])
    
    img_standardized = (img_normalized - mean) / std
    return img_standardized

def preprocess_pipeline():
    print("PIPELINE DE PREPROCESAMIENTO\n")
    
    print("1. Cargando metadata...")
    df = pd.read_csv(METADATA_CSV)
    print(f"   Total recortes: {len(df)}")
    
    print("\n2. Calculando área normalizada...")
    df = normalize_size_category(df)
    print(f"   Tamaños asignados: {df['size_category'].value_counts().to_dict()}")
    
    print("\n3. Analizando desbalance...")
    imbalance_alert = apply_class_balancing(df)
    if imbalance_alert:
        print("   Desbalance crítico detectado:")
        for alert in imbalance_alert:
            print(f"      - {alert['fruit']}: IR={alert['IR']:.1f} (minoría: {alert['minority_class']})")
    else:
        print("   No se detectó desbalance crítico")
    
    print("\n4. Aplicando data augmentation a clases minoritarias...")
    augmented_records = augment_minority_classes(df, df)
    print(f"   Imágenes aumentadas: {len(augmented_records)}")
    
    print("\n5. Extrayendo características HSV...")
    hsv_features = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="HSV"):
        crop_path = BASE_DIR / row['crop_path']
        hsv_feat = extract_hsv_features(crop_path)
        if hsv_feat:
            hsv_features.append(hsv_feat)
    
    hsv_df = pd.DataFrame(hsv_features)
    df_hsv = pd.concat([df.iloc[:len(hsv_df)].reset_index(drop=True), hsv_df], axis=1)
    
    print("\n6. Guardando dataset preprocesado...")
    if augmented_records:
        df_augmented = pd.DataFrame(augmented_records)
        df_final = pd.concat([df_hsv, df_augmented], ignore_index=True)
    else:
        df_final = df_hsv
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("\nRESUMEN FINAL")
    print(f"Dataset original: {len(df)} muestras")
    print(f"Dataset balanceado: {len(df_final)} muestras")
    print(f"Crecimiento: +{len(df_final) - len(df)} muestras ({100*(len(df_final)-len(df))/len(df):.1f}%)")
    
    print("\nDistribución final por calidad:")
    print(df_final.groupby(['fruit', 'quality']).size().unstack(fill_value=0))
    
    print("\nDistribución de tamaños asignados:")
    print(df_final.groupby(['fruit', 'size_category']).size().unstack(fill_value=0))
    
    print(f"\nPreprocesamiento completado")
    print(f"   CSV guardado: {OUTPUT_CSV}")
    
    return df_final

if __name__ == "__main__":
    df_preprocessed = preprocess_pipeline()