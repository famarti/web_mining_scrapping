import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from datetime import datetime

VECTORS_FILE = "vectores.joblib"
TARGETS_FILE = "targets.joblib"
htmls = joblib.load(VECTORS_FILE)
targets = joblib.load(TARGETS_FILE)

htmls = htmls.toarray()

targets = np.array(targets)

fecha_hora_actual = datetime.now().strftime('%Y%m%d_%H%M%S')

output_file = f"output_{fecha_hora_actual}.txt"

with open(output_file, 'w') as f:
    print(f"Output generado el {fecha_hora_actual}\n", file=f)

    CANT_FOLDS_CV = 5
    skf = StratifiedKFold(n_splits=CANT_FOLDS_CV)

    # Configurar el clasificador Random Forest
    random_forest = RandomForestClassifier(random_state=6447, class_weight='balanced')

    param_dist_rf = {
        'n_estimators': np.arange(100, 1001, 100),  
        'max_depth': [10, 20, 30, None],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'bootstrap': [True, False]  
    }

    random_search_rf = RandomizedSearchCV(
        random_forest, param_distributions=param_dist_rf, n_iter=50, 
        cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='accuracy'
    )

    accuracy_promedio_rf = 0
    mat_conf_promedio = None

    for fold, (train_index, test_index) in enumerate(skf.split(htmls, targets)):
        # Usar los índices de la validación cruzada para dividir los datos
        X_train, X_test = htmls[train_index], htmls[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        # Optimización de hiperparámetros con RandomizedSearch
        random_search_rf.fit(X_train, y_train)
        best_rf = random_search_rf.best_estimator_

        # Hacer predicciones
        preds_rf = best_rf.predict(X_test)

        # Evaluar precisión
        accuracy_rf = accuracy_score(y_test, preds_rf)
        accuracy_promedio_rf += accuracy_rf
        print(f"Fold {fold + 1} - Random Forest Accuracy: {accuracy_rf}\n", file=f)

        # Matriz de confusión
        mat_conf_rf = confusion_matrix(y_test, preds_rf)
        print(f"Matriz de Confusión Fold {fold + 1}:\n{mat_conf_rf}\n", file=f)

        # Acumular matrices de confusión
        if mat_conf_promedio is None:
            mat_conf_promedio = mat_conf_rf
        else:
            mat_conf_promedio += mat_conf_rf

        # Reporte de clasificación
        report_rf = classification_report(y_test, preds_rf, target_names=random_search_rf.classes_)
        print(f"Reporte de Clasificación Fold {fold + 1}:\n{report_rf}\n", file=f)

    # Calcular el promedio de la precisión
    accuracy_promedio_rf /= CANT_FOLDS_CV
    print(f"\nPromedio Accuracy Random Forest: {accuracy_promedio_rf}\n", file=f)

    plt.figure(figsize=(10, 7))
    sns.heatmap(mat_conf_promedio, annot=True, fmt='d', cmap='Blues', xticklabels=random_search_rf.classes_, yticklabels=random_search_rf.classes_)
    plt.title(f"Matriz de Confusión Promedio Random Forest - {fecha_hora_actual}")
    plt.ylabel('Actual')
    plt.xlabel('Predicho')
    png_file = f"confusion_{fecha_hora_actual}.png"
    plt.savefig(png_file)
    print(f"Matriz de confusión guardada en: {png_file}", file=f)

    print(f"Output completado. Resultados guardados en: {output_file}", file=f)
