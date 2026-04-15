# Clasificacion de Imagenes con CIFAR-10

Cuaderno de clasificacion de imagenes usando el dataset CIFAR-10 y redes neuronales convolucionales con Keras y TensorFlow.

## Descripcion

El proyecto entrena una CNN para clasificar imagenes de 32x32 pixeles en 10 categorias diferentes usando el dataset CIFAR-10. Se desarrollaron dos enfoques: una CNN entrenada desde cero y un modelo basado en ResNet50 con Transfer Learning y Fine-tuning.

- Streamlit: [Ver aplicacion](https://modelo-de-prediccion-de-imagenes-de-cifar10-7gnqumuynzkj84lvdw.streamlit.app/)

## Clases

| Etiqueta | Clase |
|----------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Estructura del proyecto

```
cifar10/
│
├── modelo_cifar10.keras                     # Modelo CNN entrenado desde cero
├── modelo_final_cifar10.keras              # Modelo ResNet50 con fine-tuning
├── notebook.ipynb                           # Cuaderno CNN desde cero
└── notebook_resnet_transfer_learning.ipynb  # Cuaderno ResNet50
```

## Requisitos

- Python 3.12
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Instalacion

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

---

## Cuadernos

### 1. CNN desde cero (`notebook.ipynb`)

Red neuronal convolucional entrenada desde cero con tres bloques de convolucion.

**Flujo:**
1. Carga del dataset CIFAR-10
2. Normalizacion de imagenes
3. Construccion del modelo CNN
4. Entrenamiento con early stopping y data augmentation
5. Evaluacion sobre el conjunto de test
6. Visualizacion de resultados

**Arquitectura:**

| Capa | Tipo | Detalles |
|------|------|----------|
| 1 | RandomFlip | Volteo horizontal |
| 2 | RandomRotation | Rotacion hasta 10% |
| 3 | RandomZoom | Zoom hasta 10% |
| 4 | Conv2D + BN + MaxPool | 64 filtros, kernel 3x3 |
| 5 | Conv2D + BN + MaxPool | 128 filtros, kernel 3x3 |
| 6 | Conv2D + BN + MaxPool | 256 filtros, kernel 3x3 |
| 7 | Flatten | - |
| 8 | Dense + BN + Dropout | 512 neuronas, dropout 0.5 |
| 9 | Dense + BN + Dropout | 256 neuronas, dropout 0.5 |
| 10 | Dense (salida) | 10 neuronas, softmax |

- Entrada: imagenes de 32x32x3
- Optimizador: Adam con learning rate 0.0001
- Funcion de perdida: sparse_categorical_crossentropy

**Tecnicas aplicadas:**
- Normalizacion de pixeles al rango [0, 1]
- Data augmentation (flip, rotation, zoom)
- Batch Normalization en cada bloque
- Dropout 0.5 en capas densas
- Early stopping con patience 5
- Model checkpoint guardando el mejor modelo

**Resultados:**

| Metrica | Valor |
|---------|-------|
| Test accuracy | 77% |
| Epochs entrenados | ~39 |

---

### 2. ResNet50 con Transfer Learning y Fine-tuning (`notebook_resnet_transfer_learning.ipynb`)

Modelo basado en ResNet50 preentrenado en ImageNet, adaptado para CIFAR-10 mediante transfer learning y fine-tuning en dos fases.

**Flujo:**
1. Carga del dataset CIFAR-10
2. Construccion del modelo con ResNet50 como base
3. Fase 1 — Transfer Learning: base congelada, entrenamiento de la cabeza clasificadora
4. Fase 2 — Fine-tuning: descongelamiento de las ultimas 30 capas con LR reducido
5. Evaluacion sobre el conjunto de test
6. Visualizacion de resultados y matriz de confusion

**Arquitectura:**

| Capa | Tipo | Detalles |
|------|------|----------|
| 1 | RandomFlip | Volteo horizontal |
| 2 | RandomRotation | Rotacion hasta 10% |
| 3 | RandomZoom | Zoom hasta 10% |
| 4 | Resizing | Redimension a 224x224 |
| 5 | Lambda (preprocess_input) | Preprocesamiento ResNet50 |
| 6 | ResNet50 | Base preentrenada en ImageNet |
| 7 | GlobalAveragePooling2D | Reduccion espacial |
| 8 | Dropout | 0.5 |
| 9 | Dense (salida) | 10 neuronas, softmax |

- Entrada: imagenes de 32x32x3
- Optimizador fase 1: Adam con learning rate 1e-4
- Optimizador fase 2: Adam con learning rate 1e-5
- Funcion de perdida: sparse_categorical_crossentropy

**Tecnicas aplicadas:**
- Transfer Learning con pesos de ImageNet
- Fine-tuning de las ultimas 30 capas de ResNet50
- Data augmentation dentro del modelo (flip, rotation, zoom)
- Preprocesamiento especifico de ResNet50 con `preprocess_input`
- Early stopping con patience 3 en ambas fases

**Resultados:**

| Metrica | Valor |
|---------|-------|
| Test accuracy | 93% |
| Epochs fase 1 (Transfer Learning) | 5 |
| Epochs fase 2 (Fine-tuning) | 10 |

---

## Comparacion de modelos

| Modelo | Test Accuracy | Epochs totales | Tecnica |
|--------|--------------|----------------|---------|
| CNN desde cero | 77% | ~39 | Entrenamiento completo |
| ResNet50 | 93% | ~15 | Transfer Learning + Fine-tuning |
