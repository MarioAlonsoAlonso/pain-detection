# Pain Detection System

Código del Trabajo Fin de Grado *"Detección automática de dolor en pacientes no comunicativos mediante visión por computador y datos sintéticos generados con inteligencia artificial"* (memoria en [`TFG - Alonso Alonso, Mario.pdf`](<TFG - Alonso Alonso, Mario.pdf>)).

El sistema clasifica el nivel de dolor que expresa un rostro en **tres categorías**, siguiendo el indicador de musculatura facial de la escala ESCID, a partir del vídeo de una cámara:

| Clase | Significado |
|-------|-------------|
| **0** | Sin dolor |
| **1** | Dolor moderado |
| **2** | Dolor intenso |

Incluye la generación del dataset sintético con Veo 3.1, el preprocesado de los vídeos, el entrenamiento en dos fases, la evaluación y un prototipo de inferencia en tiempo real con webcam.

---

## Arquitectura

```
Vídeo (T = 16 fotogramas, uno cada 250 ms)
  │
  ├─── Rama visual ──── EfficientNet-B0 (ImageNet) ─────► 1 280-D / fotograma
  │
  └─── Rama landmarks ─ MediaPipe FaceMesh (478 pts) ───►   256-D / fotograma
                        MLP: 1434 → 512 → 256

Fusión por fotograma: [1280 ∥ 256] = 1 536-D
Temporal:             LSTM unidireccional (2 capas, 512 unidades)
                      → último estado oculto → 512-D
Clasificador:         Linear(512→256)→BN→GELU→Dropout
                      Linear(256→128)→BN→GELU→Dropout
                      Linear(128→3) → logits
```

---

## Requisitos

- **Python 3.10, 3.11 o 3.12** (probado con 3.11.9 en Windows 11). Con Python 3.13 o posterior la instalación falla, porque MediaPipe 0.10.14 no está disponible para esas versiones.
- **Windows o Linux.** En macOS no hay versión de PyTorch con CUDA y la instalación falla.
- **GPU NVIDIA** con un driver reciente (compatible con CUDA 12.8) y unos 6 GB de VRAM para la Fase 2 del entrenamiento. La Fase 1 y la inferencia son ligeras; sin GPU la Fase 2 no es viable en un tiempo razonable.
- **Conexión a internet la primera vez**: torchvision descarga los pesos de EfficientNet-B0.
- **Webcam** para la inferencia en tiempo real.
- **OpenFace 2.x** solo si se quiere repetir la validación con PSPI (`scripts/run_openface.py` y `scripts/pain_analysis.py`).
- **Google Colab y un proyecto de Google Cloud** solo si se quiere generar un dataset nuevo (ver [Generar el dataset](#0--generar-el-dataset-opcional-google-colab)).

### Lo que no viene en el repositorio

`data/` y `checkpoints/` están en `.gitignore`, así que hay que conseguirlos aparte:

- **Los vídeos** (unos 3 300 clips sintéticos de 4 s, unos 1 100 por clase, generados con Veo 3.1). Deben colocarse en `data/raw/0`, `data/raw/1` y `data/raw/2`. Se pueden generar de nuevo con el notebook `scripts/Notebook_Generacion_Sintetica.ipynb` (paso 0), pero Veo 3.1 no da el mismo vídeo dos veces, así que el dataset resultante será parecido, no idéntico.
- **El modelo entrenado.** `evaluate.py` e `inference.py` necesitan `checkpoints/best_model.pth`. Si no se dispone de él, hay que entrenar primero (pasos 2 a 5).

---

## Instalación

Crear el entorno con Python 3.10–3.12. Si el `python` por defecto es más nuevo, en Windows se puede elegir la versión con el lanzador `py`:

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

La instalación descarga unos 3 GB (PyTorch con CUDA).

Comprobar que PyTorch ve la GPU (debe imprimir `True` y el nombre de la tarjeta):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Si imprime `False`, se ha instalado PyTorch sin soporte CUDA y todo se ejecutará en la CPU, muchísimo más lento (la Fase 2 deja de ser viable). `train.py` también lo avisa al empezar (`Device: cpu`). Se arregla reinstalando PyTorch desde el repositorio de CUDA:

```bash
pip install --force-reinstall --no-deps torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

## Estructura del proyecto

```
├── data/                          (no incluido)
│   ├── raw/{0,1,2}/               ← vídeos por clase
│   ├── processed/
│   │   ├── dataset_npy/           ← generado por preprocess.py
│   │   └── features_cache.h5      ← generado por cache_features.py
│   └── openface/{0,1,2}/          ← CSV de OpenFace (solo para pain_analysis.py)
├── src/                           ← librería: configuración, modelo y datos
├── scripts/                       ← generación del dataset (Colab) y utilidades de análisis
├── preprocess.py                  ┐
├── cache_features.py              │
├── train.py                       │ pipeline principal
├── evaluate.py                    │
├── inference.py                   ┘
├── checkpoints/                   (no incluido) pesos del modelo
├── logs/                          ← una carpeta por entrenamiento (texto, métricas, TensorBoard)
├── results/                       ← informes y figuras
└── requirements.txt
```

Todos los comandos se ejecutan desde la raíz del proyecto, salvo el notebook de generación, que se ejecuta en Google Colab.

---

## Uso paso a paso

### 0 — Generar el dataset (opcional, Google Colab)

Solo hace falta si no se dispone de los vídeos. **El notebook [`scripts/Notebook_Generacion_Sintetica.ipynb`](scripts/Notebook_Generacion_Sintetica.ipynb) debe ejecutarse en Google Colab**, no en local: usa la autenticación de Colab (`google.colab.auth`) y su descarga de ficheros, y no forma parte del entorno de `requirements.txt`.

Requisitos:

- Un proyecto de **Google Cloud** con facturación (o créditos) y la API de **Vertex AI** habilitada, con acceso a Veo 3.1.
- Un **bucket de Cloud Storage** en ese proyecto, donde Veo deja los vídeos antes de descargarlos.

Pasos:

1. Abrir el notebook en Colab (*Archivo → Subir cuaderno*).
2. En la celda de configuración, sustituir `PROJECT_ID` y `GCS_BUCKET` por el proyecto y el bucket propios.
3. Ejecutar las celdas en orden. La primera pide iniciar sesión con la cuenta de Google del proyecto.

Cada ejecución completa genera 108 vídeos de 4 s (todas las combinaciones de 2 sexos, 3 rangos de edad, 6 etnias y 3 niveles de dolor), con peinado, forma de la cara, rasgos, iluminación, enfoque y textura de piel aleatorios. Los vídeos se guardan por clase y la última celda los descarga en un zip con las carpetas `0/`, `1/` y `2/`, que se descomprime en `data/raw/`. Para llegar a unos 1 100 vídeos por clase hay que repetir la ejecución unas 30 veces. Si una sesión de Colab se corta, al volver a ejecutar la celda de generación se saltan los vídeos que ya estén en la carpeta local de la sesión.

### 1 — Colocar los vídeos

```
data/raw/0/*.mp4    ← sin dolor
data/raw/1/*.mp4    ← dolor moderado
data/raw/2/*.mp4    ← dolor intenso
```

Formatos admitidos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

### 2 — Preprocesar

De cada vídeo se extraen 16 fotogramas equiespaciados, se detecta la cara con MediaPipe FaceMesh (478 landmarks), se recorta y se redimensiona a 224×224.

```bash
python preprocess.py
```

Genera `data/processed/dataset_npy/`:

```
frames.npy       (N, 16, 224, 224, 3)  uint8
landmarks.npy    (N, 16, 1434)         float32
labels.npy       (N,)                  int32
metadata.json
```

Los vídeos en los que no se detecta cara en el primer fotograma se descartan. Opciones: `--raw_dir`, `--output_dir`, `--num_frames`, `--frame_size`.

### 3 — Pre-computar las características de EfficientNet

Se hace una sola vez. Permite que la Fase 1 no tenga que pasar los fotogramas por EfficientNet en cada época.

```bash
python cache_features.py
```

Genera `data/processed/features_cache.h5`. Hay que repetirlo si se vuelve a preprocesar o se cambian las etiquetas.

### 4 — Entrenar la Fase 1 (EfficientNet congelado)

Solo se entrenan el codificador de landmarks, la LSTM y el clasificador, sobre las características del paso 3.

```bash
python train.py --freeze_backbone
```

Por defecto: 20 épocas, early stopping con paciencia 5. Guarda `checkpoints/best_model_phase1.pth`.

### 5 — Entrenar la Fase 2 (ajuste fino completo)

Se descongela EfficientNet y se entrena todo el modelo con aumentación de datos, partiendo de la Fase 1.

```bash
python train.py --transfer_weights checkpoints/best_model_phase1.pth
```

Por defecto: 50 épocas, early stopping con paciencia 12. Guarda `checkpoints/best_model.pth`.

Opciones comunes a ambas fases: `--epochs`, `--patience`, `--lr`, `--batch_size`, `--weight_decay`.

**Continuar un entrenamiento interrumpido.** Al final de cada época se guarda `latest.pth` (o `latest_phase1.pth`), así que si el entrenamiento se corta se pierde como mucho la época en curso. Para continuar, con las mismas opciones que la ejecución original:

```bash
python train.py --resume checkpoints/latest.pth                            # Fase 2
python train.py --freeze_backbone --resume checkpoints/latest_phase1.pth   # Fase 1
```

Se recupera todo el estado: pesos, optimizador, *learning rate*, mejor modelo hasta el momento y su época, contador de paciencia del early stopping, escala de la precisión mixta y generadores aleatorios. La única diferencia con no haberlo cortado es el dropout entre las capas de la LSTM: cuDNN lo gestiona con un estado aleatorio interno que PyTorch no permite guardar. Por eso, a partir de la época reanudada, los resultados son estadísticamente equivalentes pero no idénticos bit a bit. La parte reanudada se registra en una carpeta nueva de `logs/`.

### 6 — Logs del entrenamiento

Cada ejecución de `train.py` guarda todo en su propia carpeta, `logs/<fecha_hora>_<phase1|phase2>/`:

| Fichero | Contenido |
|---|---|
| `train.log` | El mismo texto que sale por consola, incluido el error si la ejecución falla. |
| `config.json` | Comando ejecutado, argumentos, todos los hiperparámetros de `src/config.py`, GPU y versiones, tamaño de los splits y pesos de clase. |
| `metrics.csv` | Una fila por época: pérdida y precisión de entrenamiento y validación, *learning rate* y duración. Se escribe al acabar cada época. |
| `summary.json` | Resultado final: mejor época, mejor pérdida de validación, pérdida y precisión en test, si paró por early stopping y duración. |
| `events.out.tfevents.*` | Las mismas curvas para TensorBoard, más el resultado en test. |

Cada checkpoint guarda el nombre de la carpeta de logs que lo produjo, y `evaluate.py` lo muestra al cargarlo.

Para ver las curvas mientras entrena, o comparar ejecuciones:

```bash
tensorboard --logdir logs
```

La precisión de entrenamiento queda por debajo de la real por efecto de MixUp.

### 7 — Evaluar

```bash
python evaluate.py
```

Evalúa `checkpoints/best_model.pth` en validación y test, y guarda en `results/`:

- `{val,test}_report.txt` — precision, recall, F1 y AUC por clase
- `{val,test}_confusion_matrix.png`
- `{val,test}_roc_curves.png`

### 8 — Inferencia en tiempo real

```bash
python inference.py              # webcam por defecto
python inference.py --source 1   # otra webcam
```

Cada 250 ms se toma un fotograma y se predice sobre los últimos 16 (4 segundos), por lo que la primera predicción tarda 4 segundos. La ventana muestra el recorte de la cara con el color de la clase (verde, amarillo o rojo), la probabilidad de cada clase, los FPS y dos barras de calibración: la inclinación de la cámara y la distancia a la cara respecto a las del dataset.

| Tecla | Acción |
|-------|--------|
| `q` | Salir |
| `r` | Vaciar el buffer |

Opciones: `--checkpoint`, `--smooth` (factor de la media móvil exponencial, 0.35 por defecto) y `--majority_vote`.

---

## Qué hace cada fichero

### `src/` — librería

| Fichero | Función |
|---|---|
| `src/config.py` | Rutas e hiperparámetros, en un único sitio. |
| `src/model.py` | El modelo: EfficientNet-B0 y MLP de landmarks por fotograma, LSTM y clasificador. |
| `src/dataset.py` | Lectura del dataset preprocesado o de la caché, reparto estratificado y DataLoaders. |

### Raíz — pipeline principal

| Fichero | Función |
|---|---|
| `preprocess.py` | Convierte los vídeos en fotogramas recortados, landmarks y etiquetas (NPY). |
| `cache_features.py` | Pasa el dataset por EfficientNet congelado y guarda las características para la Fase 1. |
| `train.py` | Entrenamiento en dos fases, con aumentación, MixUp, early stopping, checkpoints y logs por ejecución. |
| `evaluate.py` | Métricas, matriz de confusión y curvas ROC de un checkpoint en validación y test. |
| `inference.py` | Prototipo en tiempo real con webcam. |

### `scripts/` — generación del dataset, análisis y visualización

Los `.py` se ejecutan desde la raíz, por ejemplo `python scripts/visualize_raw.py`, y guardan las figuras en `results/`. El notebook se ejecuta en Google Colab.

| Fichero | Función | Necesita |
|---|---|---|
| `Notebook_Generacion_Sintetica.ipynb` | Genera los vídeos del dataset con Veo 3.1 (paso 0). **Solo en Google Colab.** | Proyecto de Google Cloud con Vertex AI y un bucket (la generación cuesta dinero) |
| `visualize_raw.py` | Fotogramas sin procesar de un vídeo aleatorio por clase. | `data/raw` |
| `visualize_landmarks.py` | Recortes con la malla de landmarks superpuesta. | `data/raw` |
| `visualize_frames.py` | Cuadrícula de clips aleatorios del dataset preprocesado, tal como entran al modelo. Imprime la semilla usada para poder repetir la selección con `--seed`. | paso 2 |
| `visualize_occlusions.py` | Muestra las oclusiones de boca y entrecejo usadas como aumentación. | paso 2 |
| `visualize_pitch.py` | Histograma de la inclinación de la cámara respecto a la cara en el dataset (Ilustración 4 de la memoria); su mediana es la referencia de inclinación de `inference.py`. | paso 2 |
| `visualize_z_landmarks.py` | Histogramas de la profundidad de frente y barbilla. | paso 2 |
| `compute_face_size.py` | Tamaño medio de la cara respecto al fotograma; es la referencia de distancia de `inference.py`. | `data/raw` |
| `corrupt_labels.py` | Comprobación de cordura: intercambia etiquetas entre las clases 0 y 2 (`--restore` las recupera). | paso 2 |
| `run_openface.py` | Ejecuta OpenFace sobre los vídeos de `data/raw` y guarda un CSV de AUs por vídeo en `data/openface`. | `data/raw` y OpenFace |
| `pain_analysis.py` | Validación del dataset con OpenFace: PSPI por vídeo e histogramas de las AUs por clase. | `data/openface` |

**Sobre `corrupt_labels.py`:** modifica `labels.npy`, pero la Fase 1 lee las etiquetas de la caché. Tras corromper o restaurar las etiquetas hay que volver a ejecutar `cache_features.py`.

**Sobre la validación con OpenFace:** `pain_analysis.py` necesita un CSV de OpenFace 2.x por vídeo en `data/openface/{0,1,2}`, que se generan con `run_openface.py`:

1. Descargar `OpenFace_2.2.0_win_x64.zip` de las *releases* de [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/releases) y descomprimirlo.
2. Dentro de esa carpeta, ejecutar `download_models.ps1` en PowerShell (descarga los modelos que no vienen en el zip). Si `FeatureExtraction.exe` no arranca, instalar el *Visual C++ Redistributable* x64.
3. Desde la raíz del proyecto:

```bash
python scripts/run_openface.py --openface "C:/ruta/OpenFace_2.2.0_win_x64" --per_class 300
python scripts/pain_analysis.py
```

`--per_class 300` toma 300 vídeos aleatorios por clase, como en la memoria; sin esa opción procesa todos. Los vídeos que ya tienen CSV se saltan, así que se puede interrumpir y relanzar.

---

## Configuración

Los hiperparámetros están en [`src/config.py`](src/config.py).

**Modelo y datos**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `NUM_FRAMES` | 16 | Fotogramas por vídeo |
| `FRAME_SIZE` | 224 | Resolución del recorte (px) |
| `VISUAL_DIM` | 1280 | Salida de EfficientNet-B0 |
| `LANDMARK_DIM` | 1434 | 478 landmarks × 3 coordenadas |
| `LANDMARK_HIDDEN_DIM` / `LANDMARK_OUT_DIM` | 512 / 256 | MLP de landmarks |
| `LSTM_HIDDEN_DIM` / `LSTM_NUM_LAYERS` | 512 / 2 | LSTM |
| `DROPOUT` | 0.4 | Dropout (0.2 en la última capa oculta del clasificador) |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | 0.7 / 0.15 / 0.15 | Reparto estratificado |

**Entrenamiento**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `BATCH_SIZE` | 8 | Tamaño de lote |
| `LEARNING_RATE` | 1e-4 | Tasa de aprendizaje inicial (AdamW, coseno) |
| `WEIGHT_DECAY` | 1e-2 | Regularización |
| `PHASE1_NUM_EPOCHS` / `PHASE1_EARLY_STOPPING_PATIENCE` | 20 / 5 | Fase 1 |
| `NUM_EPOCHS` / `EARLY_STOPPING_PATIENCE` | 50 / 12 | Fase 2 |
| `ORDINAL_ALPHA` | 0.5 | Peso de la penalización por distancia entre clases en la pérdida |
| `SEED` | 42 | Semilla |

La pérdida es entropía cruzada con una penalización proporcional a la distancia entre la clase predicha y la real, de modo que confundir la clase 0 con la 2 penaliza el doble que confundir clases contiguas.

**Aumentación (Fase 2)**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `AUG_FLIP_PROB` | 0.5 | Volteo horizontal |
| `AUG_ROTATION_DEGREES` | 10.0 | Rotación aleatoria ±grados |
| `AUG_TRANSLATE` | 0.10 | Traslación máxima (fracción de la imagen) |
| `AUG_TEMPORAL_JITTER` | 2 | Desplazamiento temporal máximo (±fotogramas) |
| `AUG_MIXUP_ALPHA` | 0.4 | MixUp (también en Fase 1) |
| `WEBCAM_AUG` | `False` | Degradaciones de webcam (desactivado) |

Además, en GPU: ruido gaussiano, brillo y contraste (50 %), desenfoque gaussiano (30 %) y oclusiones (50 %).

---

## Reproducibilidad

- Semillas fijadas en `train.py` (Python, NumPy, PyTorch y CUDA).
- El reparto en entrenamiento, validación y test es determinista para una misma `SEED`.
- Cada checkpoint guarda los hiperparámetros del modelo bajo la clave `config`.

## Problemas conocidos

- En Windows, `train.py` termina con el código de salida `-1073740791` (`0xC0000409`) aunque el entrenamiento haya ido bien. Lo provoca cuDNN al cerrarse el proceso, por usar una LSTM de dos capas con dropout entre capas después de haber hecho algún `backward`. Ocurre cuando los checkpoints ya están guardados, así que no afecta a los resultados: si aparece la línea `Training complete` en la salida, el entrenamiento es válido. La única consecuencia es que no conviene encadenar comandos con `&&` detrás de `train.py`. Se ha mantenido así porque las alternativas que evitan el error hacen el entrenamiento más lento.
