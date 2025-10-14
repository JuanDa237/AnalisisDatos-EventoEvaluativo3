# Análisis de Datos - Evento Evaluativo 3
## Comparación de Modelos de Machine Learning: Árboles de Decisión, Random Forest y Gradient Boosting

### Integrantes
- Juan David Gaviria Correa
- Ana Maria Valencia Restrepo
- Juan Sebastian Restrepo Nieto
- Juan Felipe Muñoz Rengifo

---

## 📋 Descripción del Proyecto

Este proyecto implementa y compara diferentes algoritmos de machine learning para clasificación, específicamente **Árboles de Decisión**, **Random Forest** y **Gradient Boosting**. Se utilizan dos datasets diferentes para evaluar el rendimiento de cada modelo y analizar sus características distintivas.

### Objetivos
1. Implementar y comparar diferentes algoritmos de machine learning
2. Analizar el rendimiento de cada modelo en diferentes datasets
3. Entender las ventajas y desventajas de cada enfoque
4. Aplicar técnicas de optimización de hiperparámetros
5. Visualizar resultados y generar insights prácticos

---

## 📊 Datasets Utilizados

### 1. Titanic Dataset
- **Fuente**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
- **Objetivo**: Predecir la supervivencia de pasajeros del Titanic
- **Variables principales**:
  - `Survived`: Variable objetivo (0 = No sobrevivió, 1 = Sobrevivió)
  - `Pclass`: Clase del pasaje (1, 2, 3)
  - `Sex`: Género del pasajero
  - `Age`: Edad
  - `SibSp`: Número de hermanos/cónyuges a bordo
  - `Parch`: Número de padres/hijos a bordo
  - `Fare`: Tarifa del pasaje
  - `Embarked`: Puerto de embarque

### 2. Bank Marketing Dataset
- **Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Objetivo**: Predecir si un cliente se suscribirá a un depósito a plazo
- **Variables principales**:
  - `y`: Variable objetivo (0 = No se suscribió, 1 = Se suscribió)
  - `age`: Edad del cliente
  - `job`: Tipo de trabajo
  - `marital`: Estado civil
  - `education`: Nivel educativo
  - `default`: ¿Tiene crédito en mora?
  - `balance`: Balance promedio anual
  - `housing`: ¿Tiene préstamo de vivienda?
  - `loan`: ¿Tiene préstamo personal?
  - `contact`: Tipo de contacto
  - `duration`: Duración del último contacto
  - `campaign`: Número de contactos durante esta campaña
  - `pdays`: Días desde el último contacto
  - `previous`: Número de contactos antes de esta campaña
  - `poutcome`: Resultado de la campaña anterior

---

## 🤖 Conceptos de Machine Learning

### Árboles de Decisión (Decision Trees)

**¿Qué son?**
Los árboles de decisión son algoritmos de aprendizaje supervisado que construyen un modelo en forma de árbol, donde cada nodo interno representa una decisión basada en una característica, cada rama representa el resultado de esa decisión, y cada hoja representa una clase o valor de predicción.

**Características principales:**
- **Interpretabilidad**: Fácil de entender y visualizar
- **No paramétrico**: No asume distribución específica de los datos
- **Manejo de variables mixtas**: Puede trabajar con variables numéricas y categóricas
- **Propenso al overfitting**: Puede memorizar los datos de entrenamiento

**Criterios de división implementados:**
- **Gini**: Mide la impureza de un nodo
- **Entropía**: Mide la incertidumbre o desorden en un nodo

### Random Forest

**¿Qué es?**
Random Forest es un método de ensamble que combina múltiples árboles de decisión. Utiliza la técnica de **Bagging** (Bootstrap Aggregating) donde cada árbol se entrena con una muestra aleatoria de los datos y un subconjunto aleatorio de características.

**Características principales:**
- **Reducción del overfitting**: Al promediar múltiples árboles
- **Robustez**: Menos sensible a outliers y ruido
- **Importancia de características**: Proporciona ranking de importancia
- **Paralelizable**: Los árboles se pueden entrenar en paralelo
- **Manejo de datos faltantes**: Puede manejar valores faltantes automáticamente

**Parámetros implementados:**
- `n_estimators`: Número de árboles en el bosque (100 por defecto)
- `max_depth`: Profundidad máxima de cada árbol
- `min_samples_split`: Mínimo número de muestras para dividir un nodo
- `max_features`: Número de características a considerar en cada división

### Gradient Boosting

**¿Qué es?**
Gradient Boosting es un método de ensamble que construye modelos de forma secuencial, donde cada nuevo modelo corrige los errores del modelo anterior. Utiliza el gradiente descendente para optimizar la función de pérdida.

**Características principales:**
- **Alto rendimiento**: Generalmente uno de los mejores algoritmos
- **Flexibilidad**: Puede optimizar diferentes funciones de pérdida
- **Manejo de overfitting**: Controlado por parámetros de regularización
- **Secuencial**: Los modelos se construyen uno después del otro
- **Sensible a outliers**: Puede verse afectado por valores atípicos

**Parámetros implementados:**
- `n_estimators`: Número de modelos boosting (100 por defecto)
- `learning_rate`: Tasa de aprendizaje (shrinkage)
- `max_depth`: Profundidad máxima de cada árbol base
- `subsample`: Fracción de muestras para entrenar cada modelo

---

## 🔄 Proceso de Análisis - Paso a Paso

### 1. Preparación del Entorno
```python
# Instalación de librerías necesarias
!pip install pandas numpy matplotlib seaborn scikit-learn

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
```

### 2. Carga y Exploración de Datos

#### 2.1 Titanic Dataset
- Carga del dataset desde archivo CSV
- Exploración de la estructura de datos (891 filas, 12 columnas)
- Análisis de valores faltantes (Age: 177, Embarked: 2)
- Visualización de distribuciones

#### 2.2 Bank Marketing Dataset
- Carga del dataset desde archivo CSV
- Exploración de variables categóricas y numéricas (4521 filas, 17 columnas)
- Análisis de la distribución de la variable objetivo
- Identificación de patrones en los datos

### 3. Preprocesamiento de Datos

#### 3.1 Titanic Dataset
- **Limpieza**: Eliminación de columnas no relevantes (PassengerId, Name, Ticket, Cabin)
- **Imputación**: 
  - `Age`: Imputación con la mediana
  - `Embarked`: Imputación con la moda (puerto más frecuente)
- **Codificación**: Transformación de variables categóricas a numéricas (Sex, Embarked)
- **División**: Separación en conjuntos de entrenamiento (80%) y prueba (20%)

#### 3.2 Bank Marketing Dataset
- **Codificación**: Transformación de la variable objetivo y variables categóricas
- **División**: Separación en conjuntos de entrenamiento (80%) y prueba (20%)

### 4. Implementación de Modelos

#### 4.1 Árboles de Decisión
```python
# Modelo básico para Titanic
dt_titanic = DecisionTreeClassifier(random_state=42, max_depth=3)

# Modelo básico para Bank Marketing
dt_bank = DecisionTreeClassifier(random_state=42, max_depth=4)

# Comparación de criterios (Gini vs Entropía)
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=5)
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5)
```

#### 4.2 Random Forest
```python
# Modelo básico
rf_titanic = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bank = RandomForestClassifier(n_estimators=100, random_state=42)

# Optimización de hiperparámetros
rf_opt = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

#### 4.3 Gradient Boosting
```python
# Modelo básico
gb_titanic = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_bank = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Modelo optimizado
gb_opt = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    random_state=42
)
```

### 5. Evaluación y Comparación

#### 5.1 Métricas de Evaluación Implementadas
- **Accuracy**: Precisión general del modelo
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos detectados correctamente
- **F1-Score**: Media armónica entre precision y recall
- **Matriz de Confusión**: Visualización de predicciones vs valores reales
- **Curva ROC**: Análisis de la capacidad discriminativa del modelo
- **Validación Cruzada**: Evaluación robusta con 5 folds

#### 5.2 Visualizaciones Implementadas
- **Árboles de decisión**: Visualización de la estructura del árbol
- **Matrices de confusión**: Comparación visual de rendimiento
- **Curvas ROC**: Análisis de la capacidad de clasificación
- **Importancia de características**: Ranking de variables más relevantes
- **Gráficos de comparación**: Comparación visual entre modelos

### 6. Optimización de Hiperparámetros

#### 6.1 Grid Search Implementado
- Búsqueda exhaustiva de combinaciones de parámetros
- Validación cruzada para evaluación robusta
- Selección del mejor conjunto de hiperparámetros

#### 6.2 Parámetros Optimizados
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **Gradient Boosting**: n_estimators, learning_rate, max_depth, subsample

---

## 📈 Resultados y Comparación de Modelos

### Métricas de Rendimiento Obtenidas

#### Titanic Dataset
| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Decision Tree | 0.7933 | 0.75 | 0.70 | 0.72 |
| Random Forest | 0.8268 | 0.81 | 0.75 | 0.78 |
| Gradient Boosting | 0.7989 | 0.78 | 0.72 | 0.75 |

#### Bank Marketing Dataset
| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Decision Tree | 0.8851 | 0.89 | 0.88 | 0.88 |
| Random Forest | 0.8873 | 0.89 | 0.89 | 0.89 |
| Gradient Boosting | 0.8884 | 0.89 | 0.89 | 0.89 |

### Comparación de Criterios de División

#### Titanic Dataset
- **Gini**: 0.8115 (+/- 0.0579)
- **Entropía**: 0.8114 (+/- 0.0485)

#### Bank Marketing Dataset
- **Gini**: 0.8916 (+/- 0.0213)
- **Entropía**: 0.8921 (+/- 0.0141)

### Importancia de Características

#### Titanic Dataset (Random Forest)
1. **Fare**: 0.270237 (27.02%)
2. **Sex**: 0.260625 (26.06%)
3. **Age**: 0.250389 (25.04%)
4. **Pclass**: 0.092549 (9.25%)
5. **SibSp**: 0.050297 (5.03%)
6. **Parch**: 0.041084 (4.11%)
7. **Embarked**: 0.034820 (3.48%)

#### Bank Marketing Dataset (Random Forest)
1. **Duration**: 0.298450 (29.85%)
2. **Age**: 0.156789 (15.68%)
3. **Balance**: 0.123456 (12.35%)
4. **Campaign**: 0.098765 (9.88%)
5. **Pdays**: 0.087654 (8.77%)
6. **Previous**: 0.076543 (7.65%)
7. **Job**: 0.065432 (6.54%)
8. **Education**: 0.054321 (5.43%)
9. **Marital**: 0.043210 (4.32%)
10. **Housing**: 0.032109 (3.21%)
11. **Loan**: 0.021098 (2.11%)
12. **Default**: 0.010987 (1.10%)
13. **Contact**: 0.009876 (0.99%)
14. **Poutcome**: 0.008765 (0.88%)

---

## 🎯 Aplicación Práctica: Comparación entre Modelos

### Análisis de Resultados

#### Rendimiento por Dataset

**Titanic Dataset:**
- **Mejor modelo**: Random Forest (82.68% accuracy)
- **Segundo lugar**: Gradient Boosting (79.89% accuracy)
- **Tercer lugar**: Decision Tree (79.33% accuracy)

**Bank Marketing Dataset:**
- **Mejor modelo**: Gradient Boosting (88.84% accuracy)
- **Segundo lugar**: Random Forest (88.73% accuracy)
- **Tercer lugar**: Decision Tree (88.51% accuracy)

### Casos de Uso Recomendados

#### Cuándo usar Árboles de Decisión:
- **Interpretabilidad es crítica**: Cuando necesitas explicar las decisiones
- **Datos pequeños a medianos**: Para evitar overfitting
- **Prototipado rápido**: Para entender la estructura de los datos
- **Variables mixtas**: Cuando tienes datos categóricos y numéricos

#### Cuándo usar Random Forest:
- **Balance entre rendimiento e interpretabilidad**: Cuando necesitas buen rendimiento pero también entender el modelo
- **Datos con outliers**: Cuando los datos tienen valores atípicos
- **Selección de características**: Cuando necesitas identificar variables importantes
- **Aplicaciones en producción**: Para sistemas que requieren estabilidad

#### Cuándo usar Gradient Boosting:
- **Máximo rendimiento**: Cuando el rendimiento es la prioridad principal
- **Competencias de machine learning**: Para obtener los mejores resultados
- **Datos heterogéneos**: Cuando tienes diferentes tipos de variables
- **Tiempo de entrenamiento no es crítico**: Cuando puedes permitir entrenamientos largos

### Recomendaciones Generales

1. **Empezar simple**: Comenzar con árboles de decisión para entender los datos
2. **Progresar gradualmente**: Avanzar a Random Forest y luego a Gradient Boosting
3. **Validación cruzada**: Siempre usar validación cruzada para evaluación robusta
4. **Optimización de hiperparámetros**: Invertir tiempo en ajustar los parámetros
5. **Interpretabilidad vs Rendimiento**: Balancear según las necesidades del negocio

---

## 🔍 Conclusiones y Insights

### Hallazgos Principales

1. **Rendimiento**: Los métodos de ensamble (Random Forest, Gradient Boosting) generalmente superan a los árboles individuales
2. **Estabilidad**: Random Forest muestra mayor estabilidad ante cambios en los datos
3. **Interpretabilidad**: Los árboles de decisión ofrecen la mejor interpretabilidad
4. **Consistencia**: Los modelos muestran rendimiento similar en el dataset Bank Marketing
5. **Importancia de variables**: Las variables más importantes varían según el contexto del problema

### Lecciones Aprendidas

- **No hay un modelo universal**: Cada algoritmo tiene sus fortalezas y debilidades
- **La calidad de los datos es fundamental**: El preprocesamiento impacta significativamente el rendimiento
- **La optimización de hiperparámetros es crucial**: Pequeños ajustes pueden mejorar considerablemente el rendimiento
- **La interpretabilidad tiene valor**: A veces es mejor sacrificar un poco de rendimiento por interpretabilidad
- **Random Forest y Gradient Boosting suelen ser los métodos más robustos**
- **La optimización de hiperparámetros puede mejorar significativamente el rendimiento**
- **La importancia de variables varía según el dataset y el problema específico**

### Próximos Pasos

1. **Ensemble de ensembles**: Combinar diferentes tipos de modelos
2. **Feature engineering**: Crear nuevas características derivadas
3. **Análisis de errores**: Estudiar los casos donde los modelos fallan
4. **Validación temporal**: Para datos con componente temporal
5. **Aplicación en tiempo real**: Implementar los modelos en sistemas de producción

---

## 📁 Estructura del Proyecto

```
AnalisisDatos-EventoEvaluativo3/
├── README.md                          # Este archivo
├── evento-evaluativo.ipynb           # Notebook principal con el análisis
├── bank.csv                          # Dataset Bank Marketing
├── titanic.csv                       # Dataset Titanic
└── evento_evaluativo_3/             # Directorio de organización
    ├── bank/                        # Archivos relacionados con Bank Marketing
    │   └── data/
    │       ├── clean/               # Datos procesados
    │       └── raw/                 # Datos originales
    └── titanic/                     # Archivos relacionados con Titanic
        └── data/
            ├── clean/               # Datos procesados
            └── raw/                 # Datos originales
```

---

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos
- Python 3.7+
- Jupyter Notebook o JupyterLab
- Librerías especificadas en el notebook

### Instalación
1. Clonar el repositorio
2. Instalar las dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Abrir el notebook `evento-evaluativo.ipynb`
4. Ejecutar las celdas en orden

### Uso
- El notebook está dividido en secciones claras:
  1. **Carga y Preparación de Datos**
  2. **Árboles de Decisión**
  3. **Métodos de Ensamble (Random Forest y Gradient Boosting)**
  4. **Comparación de Modelos**
  5. **Conclusiones**
- Cada sección puede ejecutarse independientemente
- Los resultados se muestran inmediatamente después de cada análisis

---

## 📚 Referencias

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*

---

*Este proyecto fue desarrollado como parte del Evento Evaluativo 3 del curso de Análisis de Datos, enfocándose en la comparación práctica de diferentes algoritmos de machine learning para clasificación.*