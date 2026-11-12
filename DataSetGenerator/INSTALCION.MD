
INSTALACIÓN DE PYTHON Y ENTORNO VIRTUAL
1.1. Instalar Python
Opción A: Python oficial (Recomendado)

# Descargar desde: https://www.python.org/downloads/
# Versión recomendada: 3.10 o 3.11
# Durante la instalación: MARCAR "Add Python to PATH"



python --version
# Debe mostrar: Python 3.10.x o superior

pip --version


# Crear carpeta del proyecto
mkdir skytravel-analytics
cd skytravel-analytics

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate

# Verás algo como: (venv) C:\...\skytravel-analytics>


```text

# Análisis de datos
pandas==2.1.0
numpy==1.24.3

# Visualización
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Machine Learning
scikit-learn==1.3.0
xgboost==1.7.6
imbalanced-learn==0.11.0

# Conexión a base de datos (elige según tu DB)
# PostgreSQL:
psycopg2-binary==2.9.7
# MySQL:
# mysql-connector-python==8.1.0
# SQL Server:
# pyodbc==4.0.39

# Jupyter (opcional pero recomendado)
jupyter==1.0.0
jupyterlab==4.0.5

# Utilidades
python-dotenv==1.0.0

```

pip install -r requirements.txt

```py
# Archivo: verificar_instalacion.py
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib
import seaborn as sns

print("✅ pandas:", pd.__version__)
print("✅ numpy:", np.__version__)
print("✅ scikit-learn:", sklearn.__version__)
print("✅ xgboost:", xgb.__version__)
print("✅ matplotlib:", matplotlib.__version__)
print("✅ seaborn:", sns.__version__)
print("\n🎉 Todas las librerías instaladas correctamente!")
```

python test.py


configurar proyecto

crear .env

```env
# .env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=skytravel_dw
DB_USER=analista
DB_PASSWORD=tu_password_aqui
DB_TYPE=postgresql  # o mysql, sqlserver
```


**EJECUTAR**
# 1. Activar entorno virtual
venv\Scripts\activate

# 2. Verificar instalación
python verificar_instalacion.py

# 3. Probar conexión a SQL Server
python test_conexion.py

# 4. Cargar datos en Python
python cargar_datos.py

**OPCIÓN 2: JupyterLab / Jupyter Notebook (Navegador)**

La forma clásica que se abre en tu navegador web (Chrome, Edge, etc.).
Paso 1: Activa tu entorno virtual
Abre tu terminal (CMD o PowerShell en Windows) y activa el entorno:
```
cd skytravel-analytics
venv\Scripts\activate
```
**Paso 2: Instala Jupyter**
```
pip install jupyterlab
```
**Paso 3: Inicia el servidor**

jupyter lab

