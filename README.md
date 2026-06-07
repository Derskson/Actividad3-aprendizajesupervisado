# Actividad 3 — Predicción de congestión en estaciones

Breve proyecto didáctico que genera un dataset sintético y entrena un árbol de decisión
(`DecisionTreeClassifier`) para predecir si una estación estará **congestionada**.

---

Contenido
- `actividad3_arbol_decision.py`: script principal que crea el dataset, entrena el modelo,
  evalúa métricas, guarda `dataset_estaciones.csv` y `arbol_decision.png`, y muestra predicciones de ejemplo.

Archivos generados al ejecutar
- `dataset_estaciones.csv` — dataset sintético (se crea al ejecutar el script).
- `arbol_decision.png` — imagen del árbol de decisión (se crea al ejecutar el script).

Requisitos
- Python 3.8 o superior
- Paquetes: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

Instalación (ejemplo)
```bash
pip install pandas numpy scikit-learn matplotlib
```

Ejecución
```bash
python actividad3_arbol_decision.py
```

Explicación rápida de las columnas del dataset
- `hora`: hora del día (0–23).
- `pasajeros`: número estimado de pasajeros en la estación (entero).
- `clima`: condición meteorológica categórica (`soleado`, `nublado`, `lluvia`).
- `congestion`: etiqueta objetivo (`si` = congestionada, `no` = no congestionada).

Qué muestra el script
- Genera y guarda el dataset sintético en `dataset_estaciones.csv`.
- Entrena un `DecisionTreeClassifier` y muestra la `accuracy` y la matriz de confusión.
- Guarda una visualización del árbol en `arbol_decision.png`.
- Imprime tres predicciones de ejemplo y una breve explicación y conclusión.

Notas
- El proyecto es intencionalmente simple y didáctico. Para uso real hay que
  recopilar datos reales, añadir variables relevantes y validar con más rigor.
- Para modificar la complejidad del modelo, edita el parámetro `max_depth` en
  `actividad3_arbol_decision.py`.

---

Si quieres, puedo ejecutar el script aquí y pegar los resultados (accuracy, matriz de confusión) y la imagen generada.
