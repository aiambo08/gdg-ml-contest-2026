# GDG on Campus UAM · ML Contest 2026: entrega

Clasificador de series temporales multicanal para el ML Contest de GDG on Campus UAM (2026). Cada muestra es una serie
`(4, 256)` en float32, que puede traer NaN, más los metadatos `sample_rate`, `group` y `level`. Hay que asignarla a una de
las clases 0–7 dentro de las restricciones de la evaluación: 1 hilo de CPU, sin GPU, y la nota crece con el acierto y con
el número de muestras procesadas.

La solución es un **único kernel compilado con numba** que, para cada muestra, limpia la serie, calcula 228 features
resumen y aplica una **MLP pequeña destilada de un ensamble de CNN 1D**. Tarda unos 20 µs por muestra. Si el kernel no
está disponible, responde automáticamente la entrega anterior (una CNN 1D).

## Resultados

Medidos en Google Colab (CPU Intel Xeon a 2,2 GHz, 1 hilo) con el `harness.py` oficial.

| Versión | Modelo | Acierto en `test_demo` | Latencia en el harness | Latencia directa |
|---|---|---|---|---|
| **v6 (esta entrega)** | Features + MLP 228→64→8 (numba) | 0,8275 | **0,049 ms** | 20 µs |
| v5 (respaldo, `fallback_v5/`) | CNN 1D de 4 capas | 0,8675 | 0,273 ms | 167 µs |

La nota crece aproximadamente con `acierto³ · √n` (n = número de muestras procesadas). Por eso un modelo ~5,5× más
rápido con 4 puntos menos de acierto sale ganando: con nuestra validación interna, la v6 da una nota estimada de unas
**2× la de la v5**.

Verificación de la entrega (desde un clon limpio de este repositorio):
- harness sobre `test_demo`, sobre los datos de entrenamiento y con `warmup_enabled=false`: 3/3 ejecuciones correctas
  en cada caso;
- en Colab: 27/27 ejecuciones del harness correctas; preparación en 13–14 s (límite: 120 s); una sola especialización
  de numba;
- 15 casos límite (todo NaN, valores infinitos, metadatos raros o ausentes, otras formas y tipos de array):
  `predict` nunca lanza excepciones.

## Estructura

```
.
├── model.py            # clase Model: kernel numba + MLP; carga el respaldo si hace falta
├── hf_weights.npz      # pesos de la MLP, normalización de features y logits de referencia (sin pickle)
├── requirements.txt    # numpy>=1.24, numba>=0.59
└── fallback_v5/        # entrega anterior (CNN 1D), autocontenida
    ├── model.py
    ├── weights.npz
    └── config.json
```

Todas las rutas son relativas a `model.py` (`Path(__file__)`), así que el repositorio funciona desde cualquier directorio.

## Uso

```bash
pip install -r requirements.txt

# desde la carpeta de materiales del concurso (ml-contest-materials/)
python harness.py evaluate --model <ruta>/model.py --source file:data/test_demo.npz
```

Interfaz que espera el harness:

```python
from model import Model

m = Model()            # carga los pesos, compila el kernel y lo comprueba (unos segundos, fuera del tiempo puntuable)
y = m.predict(sample)  # sample.series: (4, 256) float32; sample.meta: {"sample_rate", "group", "level"}
                       # devuelve un int en 0..7 (o -1 si algo falla)
```

## Cómo funciona

### 1. Modelo de los datos

El análisis exploratorio apunta a que cada serie se genera, aproximadamente, como

```
x = level + gscale(group) · s(t / sample_rate) + ruido        con gscale: A = 1, B = 2, C = 4
```

Por eso los metadatos **solo se usan para normalizar la señal** y nunca como pistas directas de la clase: la escala por
canal absorbe `level` y la ganancia del grupo, y `sample_rate = 2` se diezma ×2 para trabajar en tiempo físico. `level`
no se usa. El grupo entra en la MLP como un one-hot para que pueda corregir pequeñas diferencias de forma.

Cada clase corresponde a un mecanismo distinto, y las features están pensadas para captarlos:

| Clase | Mecanismo |
|---|---|
| 0 | tono base |
| 1 | el canal 1 es el canal 0 con retardo |
| 2 | tendencia lineal |
| 3 | modulación de amplitud |
| 4 | saltos hacia arriba con decaimiento |
| 5 | la clase 4 invertida en el tiempo |
| 6 y 7 | oscilaciones no estacionarias (las más difíciles de separar) |

### 2. Preprocesado (dentro del kernel, sin reservar memoria)

1. Interpolación lineal de NaN e infinitos (extremos: valor más cercano; canal vacío: 0).
2. Escala robusta por canal mediante winsorización (sin ordenar el array).
3. Eliminación de picos: los puntos con |z| > 8 se reinterpolan y se recortan.
4. Diezmado ×2 si `sample_rate == 2`.
5. Estandarización por canal.

### 3. Features (228)

- correlaciones cruzadas ch0–ch1 y autocorrelaciones a 11 retardos (de 1 a 48);
- estadísticas de los incrementos a retardos 1, 2, 4, 8 y 16;
- energía por ventanas y perfiles de energía a lo largo de la serie;
- 48 bins de una DFT con ventana de Hann sobre ch0 + ch1 (tablas de senos y cosenos precalculadas);
- energía de Teager;
- one-hot del grupo.

Las features se recortan a los límites fijados en el entrenamiento y se estandarizan.

### 4. Clasificador

MLP de una capa oculta (228 → 64 → 8, ReLU), con un juego de pesos por `sample_rate`. La predicción es el argmax de los
logits.

### 5. Entrenamiento

El test tiene una distribución distinta de la del entrenamiento, y esa fue la principal dificultad del reto:

- **Atajos del train.** El conjunto de entrenamiento tiene atajos que el test no tiene: `level` depende de la clase,
  hay un offset en un canal en algunas clases y hay canales intercambiados en parte de otras. Antes de entrenar se
  deshacen.
- **Simulador del test.** Reproduce el desplazamiento observado en `test_demo`: ganancia log-normal, ruido t₄ de nivel
  aleatorio, picos, más NaN, `level` uniforme y clases balanceadas. Con él, un clasificador adversario train-vs-test
  pasa de AUC 0,92 a 0,53.
- **Datos de entrenamiento.** La MLP se entrena con varias copias simuladas de cada muestra.
- **Destilación.** La MLP se destila de un ensamble de CNN 1D (las de las versiones anteriores, con BatchNorm por
  `sample_rate` y grupo), que acierta más pero es mucho más lenta.
- **Selección de modelos.** Se usa un pseudo-test (validación + corrección de atajos + simulador). `test_demo` solo se
  usa para comprobar, porque con 400 muestras su intervalo de confianza es demasiado ancho para elegir entre modelos.

### 6. Rendimiento y robustez

- **Un solo kernel por muestra.** `predict` hace una única llamada al kernel compilado, con búferes preasignados y sin
  copias. El kernel se compila durante la preparación para el tipo exacto de array que entrega el harness (float32 de
  solo lectura), y después se desactiva la compilación: ninguna llamada puntuable puede pagar una recompilación.
- **Comprobación al arrancar.** En la preparación, el kernel se ejecuta sobre unas entradas de referencia guardadas en
  `hf_weights.npz` y sus logits se comparan con los esperados.
- **Respaldo automático.** Si numba no está instalado, el kernel no compila o los logits no coinciden, `Model` carga
  `fallback_v5/` (la CNN de la v5, con su propia inferencia numba y una ruta numpy pura).
- **Entradas raras.** Otra forma o tipo de array, o metadatos ausentes o inesperados, pasan por una ruta lenta que los
  normaliza. Cualquier error devuelve `-1` (abstención) en lugar de lanzar una excepción.

## Historial de versiones

| Versión | Cambio | Acierto en demo |
|---|---|---|
| v1 | CNN 1D (26k parámetros) en numpy puro | 0,8625 |
| v2 | Misma red, inferencia 1,6× más rápida (logits idénticos) | 0,8625 |
| v4.1 | BatchNorm por (`sample_rate`, grupo) + inferencia fusionada en numba | 0,870 |
| v5 | CNN destilada de un ensamble de 4 CNN | 0,8675 |
| **v6** | **Features + MLP destilada en un kernel numba (~5,5× más rápida en el harness)** | **0,8275** |

## Requisitos

- `numpy>=1.24`
- `numba>=0.59` (sin numba el modelo sigue funcionando: responde el respaldo, más lento)
