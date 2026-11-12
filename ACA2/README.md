# ACA N.º 2 · Deserción estudiantil con redes neuronales

Aprendizaje Automático · Especialización en IA · CUN · Docente: Dr. Hamilton Rivera Flor
Pilar: Educación · Dataset: UCI *Predict Students Dropout and Academic Success*

## Estado del proyecto

| Paso del enunciado | Estado | Archivo |
|---|---|---|
| 1. Exploración y gestión SQL | Hecho | `ACA2_Paso1_SQL_Desercion.ipynb` |
| 2. Preparación y pipeline | Hecho | `ACA2_Paso2_Preparacion_Desercion.ipynb` |
| 3. Modelado neuronal | Hecho | `ACA2_Paso3_4_Modelado_Evaluacion.ipynb` |
| 4. Evaluación y métricas | Hecho | `ACA2_Paso3_4_Modelado_Evaluacion.ipynb` |
| 5. App en Streamlit | **Pendiente** | — |
| Póster tipo infografía (APA 7) | **Pendiente** | — |
| Video de sustentación (3 a 5 min) | **Pendiente** | — |

## Archivos

```
ACA2/
├── ACA2_Paso1_SQL_Desercion.ipynb          # base SQLite, índices, transacciones y EDA con SQL
├── ACA2_Paso2_Preparacion_Desercion.ipynb  # partición 70/15/15, one-hot, escalado, SMOTE-NC
├── ACA2_Paso3_4_Modelado_Evaluacion.ipynb  # MLP, GRU, curvas, métricas y matriz de confusión
├── artefactos_paso2.zip   # datos preparados, preprocesador.joblib, preprocesamiento.py, config, desercion.db
└── artefactos_paso3.zip   # modelo_desercion.keras, metricas_paso3_4.json, figuras para el póster
artefactos/
└── desercion.db           # base SQLite del Paso 1 (la descarga el notebook del Paso 2)
```

## Resultados (Pasos 3 y 4)

**Modelo final:** red híbrida GRU + densa (28.675 parámetros). Una GRU de 32 unidades lee los dos semestres como secuencia; una capa densa procesa las variables estáticas; salida *softmax* de 3 clases. Entrenada con Adam, pesos por clase, *dropout*, L2, *batch normalization* y *early stopping*.

| Métrica en prueba (664 estudiantes) | Valor |
|---|---|
| F1 macro | 0,709 (IC 95 %: 0,672 – 0,743) |
| Exactitud | 0,752 |
| Pérdida | 0,588 |
| AUC macro (OvR) | 0,895 |
| F1 Abandono / Sigue matriculado / Se gradúa | 0,756 / 0,521 / 0,848 |

**Comparación en la misma prueba (F1 macro):** GRU híbrida 0,709 · MLP 0,727 · Gradient Boosting (ACA 1) 0,704 · Regresión logística 0,696. Todos dentro del intervalo de confianza: no se puede afirmar que uno sea mejor que otro.

### Lo que hay que saber para sustentar
- **La red iguala al modelo de la ACA 1, no lo supera.**
- **GRU y MLP empataron.** La GRU se eligió con validación (promedio de 5 semillas: 0,712 frente a 0,709). En prueba el MLP sacó más, pero el modelo no se cambia mirando la prueba.
- **SMOTE-NC no mejoró** frente a los pesos por clase.
- **Una red más grande no mejoró:** el límite está en los datos.
- **«Sigue matriculado» es la clase débil** (precisión 0,45): es un estado sin resolver, no un desenlace.

## Cómo reproducir
1. Ejecutar el Paso 1 (genera `desercion.db`) y el Paso 2 (genera `artefactos_paso2.zip`).
2. Para los Pasos 3 y 4, dejar `artefactos_paso2.zip` en la misma carpeta del notebook.

> Los Pasos 3 y 4 se ejecutaron con Keras 3 sobre JAX, scikit-learn 1.9.1 y numpy 2.5.3. En otro entorno los decimales cambian y, por ser un empate, el modelo elegido puede pasar a ser el MLP; en ese caso hay que revisar los textos de interpretación.

## Pendiente

**Paso 5 · App de Streamlit (15 %)**
- `app.py`: formulario de entrada, carga de `modelo_desercion.keras` y `preprocesador.joblib`, separación en secuencia y variables estáticas (función `a_secuencia` del notebook) y salida con las 3 probabilidades.
- `requirements.txt` con las versiones exactas (keras, jax, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5, joblib, streamlit). Con otras versiones el preprocesador puede no cargar.
- `preprocesamiento.py` junto a `app.py` (el preprocesador lo importa).
- Reducir el formulario: son 36 variables; dejar 8 a 10 campos clave y el resto con valores por defecto, o cargar estudiantes de ejemplo.
- Instrucciones de uso y despliegue en Streamlit Community Cloud; el enlace va en el póster.

**Póster (15 %):** integrantes, objetivo, pregunta orientadora, dataset, diagrama del flujo, arquitectura, métricas, vista previa de la app, conclusiones, enlaces de la app y del video, referencias en APA 7. Las figuras están en `artefactos_paso3.zip`.

**Video (15 %):** 3 a 5 minutos con problema, arquitectura, métricas y demostración en vivo de la app y del póster.

**Por decidir en el grupo**
- Confirmar con el docente si se entregan tres notebooks o uno solo (el enunciado habla de «notebook» en singular).
- Unificar el nombre del grupo en los archivos.
