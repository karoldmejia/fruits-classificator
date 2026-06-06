**Fruits Classificator** es un sistema de clasificación automática de frutas basado en técnicas de Visión por Computador, Machine Learning y Deep Learning. El proyecto fue desarrollado con el objetivo de automatizar procesos de inspección y clasificación en contextos agroindustriales, permitiendo identificar simultáneamente la **calidad comercial** y el **tamaño** de diferentes frutas a partir de imágenes digitales.

La investigación compara el desempeño de modelos clásicos de aprendizaje supervisado, específicamente **Support Vector Machines (SVM)** y **XGBoost**, con arquitecturas de **Redes Neuronales Convolucionales (CNN)**, evaluando su capacidad para resolver tareas de clasificación multiclase en condiciones reales.

Integrantes:
- Adri Jhoanny Martinez Murillo (A00400842)
- Johan Stiven Guzmán (A00401480)
- Karold Lizeth Mejia Orozco (A00401806)

---

## Objetivos del proyecto

El trabajo busca diseñar e implementar un pipeline completo de clasificación automática de frutas que abarque desde la detección y segmentación de los objetos hasta la evaluación de modelos predictivos. De manera específica, se pretende:

* Clasificar frutas según su calidad comercial (*good*, *regular* y *bad*).
* Clasificar frutas según categorías de tamaño (*pequeño*, *mediano* y *grande*).
* Comparar enfoques de aprendizaje clásico y aprendizaje profundo.
* Analizar el impacto de las características geométricas y cromáticas sobre el rendimiento de los modelos.
* Proponer una solución viable para escenarios de automatización industrial.

---

## Conjunto de datos

El conjunto de datos está compuesto por imágenes de seis tipos de frutas: manzana (*apple*), banano (*banana*), guayaba (*guava*), limón (*lime*), naranja (*orange*) y granada (*pomegranate*).

Cada muestra posee una etiqueta de calidad asignada manualmente y una etiqueta de tamaño generada automáticamente mediante el análisis del área normalizada de la fruta dentro de la imagen. Para ello se utilizaron percentiles específicos por tipo de fruta, garantizando una categorización consistente entre especies con dimensiones naturalmente diferentes.

Tras el proceso de preprocesamiento y balanceo, el dataset final quedó conformado por **36.848 muestras**.

---

## Metodología

El pipeline desarrollado se compone de varias etapas consecutivas. Inicialmente se realizó la detección y segmentación de frutas utilizando una combinación de técnicas basadas en YOLOv8, análisis de saturación en el espacio HSV, detección de contornos y segmentación por *watershed*. Posteriormente se eliminaron recortes inválidos y todas las imágenes fueron normalizadas a una resolución de **224 × 224 píxeles**.

A partir de los recortes obtenidos se extrajeron características geométricas y cromáticas. Entre las variables geométricas se incluyeron el área en píxeles, la relación de aspecto y la proporción de cobertura de la imagen. En cuanto a las variables cromáticas, se calcularon estadísticas de primer y segundo orden sobre los canales HSV, incluyendo medias y desviaciones estándar de tono, saturación y brillo.

Con el fin de reducir el impacto del desbalance de clases, se aplicaron técnicas de *data augmentation* sobre las categorías minoritarias mediante rotaciones, reflejos, cambios de brillo y contraste, adición de ruido y transformaciones geométricas leves.

---

## Modelos evaluados

Se implementaron tres enfoques de clasificación:

- **Support Vector Machine (SVM):** Modelo clásico basado en funciones *kernel*, utilizado para construir hiperplanos de separación en espacios de alta dimensión a partir de las características geométricas y cromáticas extraídas durante el preprocesamiento.
- **XGBoost:** Modelo de aprendizaje basado en *Gradient Boosting* sobre árboles de decisión. Su capacidad para capturar relaciones no lineales entre variables lo convirtió en una de las alternativas más sólidas para la clasificación tabular.
- **Convolutional Neural Network (CNN):** Arquitectura de aprendizaje profundo entrenada directamente sobre las imágenes, eliminando la necesidad de diseñar manualmente las características utilizadas para la clasificación.

---

## Resultados

La evaluación experimental mostró comportamientos diferenciados según la tarea analizada.

### Clasificación de calidad

| Modelo  | F1-Score Macro |
| ------- | -------------- |
| SVM     | 0.9319         |
| XGBoost | 0.9494         |
| CNN     | **0.9497**     |

La CNN obtuvo el mejor desempeño global para la clasificación de calidad, mostrando una mayor capacidad para identificar patrones complejos asociados a defectos superficiales, manchas y variaciones de textura.

### Clasificación de tamaño

| Modelo  | F1-Score Macro |
| ------- | -------------- |
| SVM     | 0.9590         |
| XGBoost | **0.9813**     |
| CNN     | 0.9181         |

En la tarea de tamaño, XGBoost alcanzó los mejores resultados al aprovechar directamente descriptores geométricos explícitos como el área y la proporción de cobertura.

---

## Discusión

Los resultados obtenidos evidencian que cada paradigma posee fortalezas específicas. Mientras las redes convolucionales sobresalen en problemas donde la información espacial y textural es determinante, los modelos tabulares mantienen una ventaja significativa cuando existen variables geométricas altamente informativas y fácilmente cuantificables.

Por esta razón, la investigación concluye que la estrategia más adecuada para un entorno industrial consiste en una arquitectura híbrida, donde **XGBoost** sea responsable de la clasificación de tamaño y la **CNN** realice la evaluación de calidad superficial. Esta combinación permite aprovechar simultáneamente la precisión geométrica de los modelos clásicos y la capacidad de representación visual de las redes neuronales.

---

## Estructura del proyecto

```text
fruits-classificator/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── notebooks/
├── models/
├── src/
├── results/
│
├── requirements.txt
└── README.md
```

---

## Tecnologías utilizadas

El desarrollo del proyecto se realizó utilizando Python como lenguaje principal y un ecosistema de bibliotecas especializadas en visión por computador, aprendizaje automático y aprendizaje profundo. Entre las herramientas empleadas destacan OpenCV, NumPy, Pandas, Scikit-Learn, XGBoost, TensorFlow/Keras, Albumentations, YOLOv8 y Matplotlib.