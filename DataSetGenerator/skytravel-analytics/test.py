# Archivo: verificar_instalacion.py
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib
import seaborn as sns

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgb.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
print("\nTodas las librerías instaladas correctamente!")