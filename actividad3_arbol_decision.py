#!/usr/bin/env python3
"""
Actividad universitaria: Aprendizaje supervisado con un árbol de decisión.
Predice si una estación estará 'congestionada' a partir de 'hora', 'pasajeros' y 'clima'.

Pasos realizados por el script:
1) Genera un dataset sintético (~100 registros) y lo guarda en 'dataset_estaciones.csv'.
2) Carga el dataset con pandas.
3) Prepara los datos (codifica variables categóricas y la etiqueta).
4) Entrena un DecisionTreeClassifier.
5) Evalúa el modelo (accuracy y matriz de confusión).
6) Guarda una gráfica del árbol en 'arbol_decision.png'.
7) Realiza tres predicciones de ejemplo y muestra una explicación y una conclusión.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


def generar_dataset(n_samples=100, seed=42):
    """Genera un dataset sintético y devuelve un DataFrame."""
    np.random.seed(seed)
    hours = list(range(24))
    # Más peso en horas punta (matutina y vespertina)
    weights = []
    for h in hours:
        if h in [7, 8, 9, 17, 18, 19]:
            weights.append(4)
        elif h in [6, 10, 16, 20]:
            weights.append(2)
        else:
            weights.append(1)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    hora = np.random.choice(hours, size=n_samples, p=weights)
    climas = ['soleado', 'nublado', 'lluvia']
    clima = np.random.choice(climas, size=n_samples, p=[0.5, 0.3, 0.2])

    pasajeros = []
    for i, h in enumerate(hora):
        # Base de pasajeros depende de si es hora punta
        if h in [7, 8, 9, 17, 18, 19]:
            base = int(np.random.normal(300, 80))
        else:
            base = int(np.random.normal(100, 50))
        # ruido y límite mínimo
        cantidad = max(5, base + int(np.random.normal(0, 20)))
        pasajeros.append(cantidad)
    pasajeros = np.array(pasajeros)

    # Generar etiqueta 'congestion' usando una regla probabilística simple
    congestion = []
    for p, h, c in zip(pasajeros, hora, clima):
        peak = h in [7, 8, 9, 17, 18, 19]
        rain = (c == 'lluvia')
        # puntuación simple que combina pasajeros, hora punta y lluvia
        score = (p / 500.0) + (0.3 if peak else 0.0) + (0.2 if rain else 0.0)
        prob = min(1.0, score)
        label = 'si' if np.random.rand() < prob else 'no'
        congestion.append(label)

    df = pd.DataFrame({
        'hora': hora,
        'pasajeros': pasajeros,
        'clima': clima,
        'congestion': congestion
    })
    return df


def preparar_datos(df):
    """Prepara X e y para entrenamiento (codifica variables categóricas)."""
    # mapear la etiqueta a 0/1
    df = df.copy()
    df['congestion_num'] = df['congestion'].map({'no': 0, 'si': 1})
    # variables predictoras: hora, pasajeros y dummies de clima
    X = df[['hora', 'pasajeros']].copy()
    clima_dummies = pd.get_dummies(df['clima'], prefix='clima')
    X = pd.concat([X, clima_dummies], axis=1)
    y = df['congestion_num']
    return X, y


def entrenar_arbol(X_train, y_train, max_depth=5, seed=42):
    """Entrena un DecisionTreeClassifier y devuelve el modelo."""
    clf = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
    clf.fit(X_train, y_train)
    return clf


def preparar_muestra(sample, feature_names):
    """Convierte un diccionario de ejemplo a un DataFrame con las mismas columnas que X."""
    df_s = pd.DataFrame([sample])
    df_s = pd.concat([df_s[['hora', 'pasajeros']], pd.get_dummies(df_s['clima'], prefix='clima')], axis=1)
    # asegurar columnas faltantes (p. ej. si el ejemplo no contiene cierta categoría)
    for col in feature_names:
        if col not in df_s.columns:
            df_s[col] = 0
    df_s = df_s[feature_names]
    return df_s


def main():
    # 1-2. Generar dataset y guardarlo en CSV
    df = generar_dataset(n_samples=100, seed=42)
    csv_path = 'dataset_estaciones.csv'
    df.to_csv(csv_path, index=False)
    print(f"Dataset sintético creado y guardado en '{csv_path}' ({len(df)} registros).")

    # 3. Explicación breve de las columnas
    print("\nDescripción de columnas:")
    print("- hora: hora del día (0-23).")
    print("- pasajeros: número estimado de pasajeros en la estación.")
    print("- clima: condición meteorológica ('soleado','nublado','lluvia').")
    print("- congestion: etiqueta objetivo ('si' = congestionada, 'no' = no congestionada).")

    # 4. Cargar dataset con pandas
    df = pd.read_csv(csv_path)
    print("\nPrimeras filas del dataset:")
    print(df.head(8).to_string(index=False))

    # 5. Preparar datos para entrenamiento
    X, y = preparar_datos(df)
    feature_names = X.columns.tolist()

    # 7. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6 & 8. Crear y entrenar DecisionTreeClassifier
    clf = entrenar_arbol(X_train, y_train, max_depth=5, seed=42)

    # 9. Evaluar: Accuracy y matriz de confusión
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(f"\nAccuracy en conjunto de prueba: {accuracy:.3f}")
    print("Matriz de confusión (filas: verdad, columnas: predicción):")
    print(cm)

    # 10. Graficar el árbol de decisión
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_names, class_names=['no', 'si'], filled=True, rounded=True)
    plt.title("Árbol de decisión para predecir congestión")
    plt.tight_layout()
    plot_path = 'arbol_decision.png'
    plt.savefig(plot_path, dpi=200)
    print(f"\nGráfica del árbol guardada en '{plot_path}'.")

    # 11. Tres ejemplos de predicción
    ejemplos = [
        {'hora': 8, 'pasajeros': 450, 'clima': 'lluvia'},     # hora punta y muchos pasajeros -> 'si'
        {'hora': 14, 'pasajeros': 60, 'clima': 'soleado'},    # fuera de punta y pocos pasajeros -> 'no'
        {'hora': 18, 'pasajeros': 180, 'clima': 'nublado'}   # hora punta con pasajeros moderados -> incierto
    ]
    print("\nPredicciones de ejemplo:")
    for ej in ejemplos:
        Xs = preparar_muestra(ej, feature_names)
        pred = clf.predict(Xs)[0]
        proba = clf.predict_proba(Xs)[0][1] if hasattr(clf, "predict_proba") else None
        proba_str = f"(prob={proba:.2f})" if proba is not None else ""
        print(f"- Entrada: {ej} -> Predicción: {'si' if pred==1 else 'no'} {proba_str}")

    # 12. Explicación sencilla de resultados
    print("\nExplicación sencilla:")
    print("- El modelo utiliza 'hora', 'pasajeros' y 'clima' para decidir si hay congestión.")
    print("- El 'accuracy' indica la proporción de predicciones correctas en el conjunto de prueba.")
    print("- La matriz de confusión muestra verdaderos/falsos positivos y negativos.")
    print("- En general, el árbol captura reglas simples como 'muchos pasajeros' y 'hora punta' -> congestión.")

    # 13. Conclusión (máx. 5 párrafos)
    conclusion = [
        "Se generó un dataset sintético y se entrenó un árbol de decisión para predecir congestión en estaciones. "
        "El dataset incorpora variables relevantes (hora, pasajeros, clima) y la etiqueta se creó con una regla probabilística.",
        "El modelo es interpretable: permite visualizar qué características y umbrales conducen a predecir congestión. "
        "En este experimento sencillo, las variables con más peso suelen ser el número de pasajeros y si la hora es punta.",
        "Las métricas (accuracy y matriz de confusión) dan una idea del rendimiento sobre los datos sintéticos; en escenarios reales se necesitarían más datos, validación cruzada y pruebas contra datos en producción.",
        "Como pasos siguientes se recomienda recolectar datos reales, probar modelos más robustos (p. ej. Random Forest) y añadir características adicionales (capacidad de estación, eventos especiales, trabajos en vías) para mejorar la predicción."
    ]
    print("\nConclusión:")
    for p in conclusion:
        print("\n" + p)


if __name__ == "__main__":
    main()
