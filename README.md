# Clasificacion de Imagenes con CIFAR-10

Cuaderno de clasificacion de imagenes usando el dataset CIFAR-10 y redes neuronales convolucionales con Keras y TensorFlow.

## Descripcion

El proyecto entrena una CNN para clasificar imagenes de 32x32 pixeles en 10 categorias diferentes usando el dataset CIFAR-10.

- Streamlit: [Ver aplicacion]()

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
├── modelo_cifar10.keras   # Modelo entrenado guardado
└── notebook.ipynb         # Cuaderno principal
```

## Requisitos

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Instalacion

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

## Flujo del proyecto

1. Carga del dataset CIFAR-10
2. Normalizacion de imagenes
3. Construccion del modelo CNN
4. Entrenamiento con early stopping y data augmentation
5. Evaluacion sobre el conjunto de test
6. Visualizacion de resultados

## Arquitectura del modelo

Red neuronal convolucional secuencial con tres bloques de convolucion y dos capas densas.

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

## Resultados

| Metrica | Valor |
|---------|-------|
| Test accuracy | 76% |
| Epochs entrenados | ~42 |

## Tecnicas aplicadas

- Normalizacion de pixeles al rango [0, 1]
- Data augmentation (flip, rotation, zoom)
- Batch Normalization en cada bloque
- Dropout 0.5 en capas densas
- Early stopping con patience 5
- Model checkpoint guardando el mejor modelo
# Modelo-de-Prediccion-de-Imagenes-de-Cifar10
# Modelo-de-Prediccion-de-Imagenes-de-Cifar10
