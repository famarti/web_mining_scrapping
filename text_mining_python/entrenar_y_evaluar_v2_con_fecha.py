import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

VECTORS_FILE = "vectores_con_fecha.joblib"
TARGETS_FILE = "targets_con_fecha.joblib"
DATES_FILE = "fechas.joblib"

htmls = joblib.load(VECTORS_FILE).toarray()
targets = np.array(joblib.load(TARGETS_FILE))
fechas = np.array(joblib.load(DATES_FILE), dtype='datetime64')

fecha_inicio_train = np.datetime64('2018-08-01')
fecha_fin_train = np.datetime64('2024-09-30')
fecha_inicio_test = np.datetime64('2024-10-15')
fecha_fin_test = np.datetime64('2024-10-31')

X_train = htmls[(fechas >= fecha_inicio_train) & (fechas <= fecha_fin_train)]
y_train = targets[(fechas >= fecha_inicio_train) & (fechas <= fecha_fin_train)]

X_test = htmls[(fechas >= fecha_inicio_test) & (fechas <= fecha_fin_test)]
y_test = targets[(fechas >= fecha_inicio_test) & (fechas <= fecha_fin_test)]

random_forest = RandomForestClassifier(random_state=3654, class_weight='balanced')

param_dist_rf = {
    'n_estimators': np.arange(100, 1001, 100),  
    'max_depth': [10, 20, 30, None],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'bootstrap': [True, False]  
}

# Obtener fecha y hora actuales para nombrar archivos
fecha_hora_actual = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"output_temporal_{fecha_hora_actual}.txt"

random_search_rf = RandomizedSearchCV(
    random_forest, param_distributions=param_dist_rf, n_iter=50, 
    cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='accuracy'
)

# Redirigir la salida estándar a un archivo .txt
with open(output_file, 'w') as f:
    print(f"Output generado el {fecha_hora_actual}\n", file=f)

    random_search_rf.fit(X_train, y_train)
    best_rf = random_search_rf.best_estimator_

    # Predicciones y evaluación en el conjunto de test
    preds_rf = best_rf.predict(X_test)

    # Evaluar precisión
    accuracy_rf = accuracy_score(y_test, preds_rf)
    print(f"Accuracy: {accuracy_rf}", file=f)

    # Matriz de confusión
    mat_conf_rf = confusion_matrix(y_test, preds_rf)
    print(f"Matriz de Confusión:\n{mat_conf_rf}", file=f)

    # Reporte de clasificación
    report_rf = classification_report(y_test, preds_rf)
    print(f"Reporte de Clasificación:\n{report_rf}", file=f)

    # Guardar la matriz de confusión en un archivo PNG
    print(f"Guardando matriz de confusión en archivo PNG...", file=f)

plt.figure(figsize=(10, 7))
sns.heatmap(mat_conf_rf, annot=True, fmt='d', cmap='Blues')
plt.title(f"Matriz de Confusión Random Forest - {fecha_hora_actual}")
plt.ylabel('Actual')
plt.xlabel('Predicho')
png_file = f"confusion_temporal_{fecha_hora_actual}.png"
plt.savefig(png_file)

print(f"Matriz de confusión guardada en: {png_file}")
