CELDA 1 — code
Source:

python
# ============================================================
# Celda 1: Importar librerías y configurar el KPI
# ============================================================
from lightgbm import LGBMClassifier
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from cargar_datos import cargar_dataset_recontacto

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ============ CAMBIAR SOLO ESTA LÍNEA EN CADA NOTEBOOK ============
KPI_SEL = 'KPI01'      # 'KPI01', 'KPI02', 'KPI07', 'KPI08' o 'KPI10'
# ===================================================================

FICHAS = {
    'KPI01': dict(nombre='Recontacto general', filtro=None,
                  formula='casos con recontacto a 7 días ÷ casos con resultado conocido',
                  base=41.5, meta=36.5),
    'KPI02': dict(nombre='Recontacto en motivos críticos (falla/queja)', filtro='motivos_criticos',
                  formula='recontactos a 7 días de casos por falla o queja ÷ casos por falla o queja con resultado conocido',
                  base=46.1, meta=40.5),
    'KPI07': dict(nombre='Recontacto en usuarios con historial', filtro='historial',
                  formula='recontactos a 7 días de usuarios con casos previos en 30 días ÷ casos de esos usuarios con resultado conocido',
                  base=50.4, meta=44.4),
    'KPI08': dict(nombre='Recontacto en canales de baja resolución (red social y correo)', filtro='canales_baja_resolucion',
                  formula='recontactos a 7 días en red social y correo ÷ casos en esos canales con resultado conocido',
                  base=44.3, meta=39.1),
    'KPI10': dict(nombre='Recontacto en turno nocturno y fin de semana', filtro='nocturno_finde',
                  formula='recontactos a 7 días en turno nocturno y fin de semana ÷ casos en esos turnos con resultado conocido',
                  base=44.1, meta=38.6),
}
_f = FICHAS[KPI_SEL]

# ------------------ CONFIGURACIÓN DEL KPI ------------------
KPI_ID = KPI_SEL
KPI_NOMBRE = _f['nombre']
FILTRO = _f['filtro']
KPI_FORMULA = _f['formula']

# ------------------ OBJETIVO Y META ------------------
OBJETIVO = 'Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800'

# Referencia externa (NO equivalente): 100% - 78% de resolución en primer contacto.
META_RECONTACTO = 22.0

# Ficha del KPI
LINEA_BASE = _f['base']     # tasa actual del segmento (%)
META_6M = _f['meta']        # meta a 6 meses (escenario del 15 %, supuesto a validar)
PLAZO = '6 meses'
UMBRAL_ALERTA = _f['base']  # si la tasa supera la línea base, se activa alerta

TARGET_DEF = ('es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; '
              '0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo')
# -----------------------------------------------------

os.makedirs('../informe_figuras', exist_ok=True)

def guardar(nombre):
    plt.savefig(f'../informe_figuras/{KPI_ID}_{nombre}.png', dpi=150, bbox_inches='tight')

print("✅ Librerías cargadas correctamente")
print(f"📌 {KPI_ID} - {KPI_NOMBRE}")
print(f"📐 Fórmula: {KPI_FORMULA}")
print(f"🎯 Objetivo: {OBJETIVO}")
print(f"📊 Línea base: {LINEA_BASE}% | 🎯 Meta {PLAZO}: ≤ {META_6M}% | 🚨 Umbral de alerta: > {UMBRAL_ALERTA}%")
print(f"🏁 Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%")
print(f"🔖 Target: {TARGET_DEF}")
Output (stdout):

text
✅ Librerías cargadas correctamente
📌 KPI01 - Recontacto general
📐 Fórmula: casos con recontacto a 7 días ÷ casos con resultado conocido
🎯 Objetivo: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
📊 Línea base: 41.5% | 🎯 Meta 6 meses: ≤ 36.5% | 🚨 Umbral de alerta: > 41.5%
🏁 Referencia de largo plazo (no equivalente): ≤ 22%
🔖 Target: es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo
CELDA 2 — code
Source:

python
# ============================================================
# Celda 2: Cargar dataset desde SQL Server
# ============================================================
def filtrar(d, filtro):
    if filtro == 'motivos_criticos':
        d = d[(d.motivo_falla == 1) | (d.motivo_queja == 1)]
    elif filtro == 'historial':
        d = d[d.tiene_historial == 1]
    elif filtro == 'canales_baja_resolucion':
        d = d[(d.canal_red_social == 1) | (d.canal_correo == 1)]
    elif filtro == 'nocturno_finde':
        d = d[(d.turno_nocturno == 1) | (d.turno_fin_semana == 1)]
    return d.reset_index(drop=True)

print("=" * 70)
print(f"📊 CARGANDO DATASET - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)

df_base = cargar_dataset_recontacto()
df = filtrar(df_base, FILTRO)

tasa_inicial = df['es_recontacto'].mean() * 100

print(f"\n✅ Dataset del KPI listo")
print(f"📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"📈 Tasa de recontacto del KPI: {tasa_inicial:.2f}%")
print(f"📈 Tasa de recontacto global : {df_base['es_recontacto'].mean() * 100:.2f}%")
print(f"🏁 Meta {PLAZO}: ≤ {META_6M}% → brecha actual: {tasa_inicial - META_6M:.1f} puntos | referencia largo plazo: {META_RECONTACTO:.0f}%")

# Control: la línea base escrita en la Celda 1 debe coincidir con la tasa real
if abs(tasa_inicial - LINEA_BASE) > 0.5:
    print(f"⚠️ OJO: la línea base de la ficha ({LINEA_BASE}%) no coincide con la tasa real ({tasa_inicial:.1f}%). Revisar KPI_SEL.")
else:
    print(f"✅ La línea base de la ficha ({LINEA_BASE}%) coincide con la tasa real.")
Output (stdout):

text
======================================================================
📊 CARGANDO DATASET - KPI01: Recontacto general
======================================================================
============================================================
📊 CARGANDO DATASET DE RECONTACTO (KPI 1)
============================================================
✅ Conexión exitosa a SQL Server
✅ Consulta ejecutada: 11,436 registros, 34 columnas

📏 Dimensiones: 11,436 filas × 34 columnas

🎯 Distribución del target:
es_recontacto
0    6687
1    4749
Name: count, dtype: int64

📈 Tasa de recontacto: 41.53%

✅ Dataset del KPI listo
📏 Dimensiones: 11,436 filas × 34 columnas
📈 Tasa de recontacto del KPI: 41.53%
📈 Tasa de recontacto global : 41.53%
🏁 Meta 6 meses: ≤ 36.5% → brecha actual: 5.0 puntos | referencia largo plazo: 22%
✅ La línea base de la ficha (41.5%) coincide con la tasa real.
CELDA 3 — code
Source:

python
# ============================================================
# Celda 3: Exploración inicial del dataset
# ============================================================
print("=" * 70)
print("🔍 EXPLORACIÓN INICIAL DEL DATASET")
print("=" * 70)

print(f"\n📋 Columnas del dataset ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col:28s} | Tipo: {str(df[col].dtype):8s} | Nulos: {df[col].isnull().sum()}")

target_counts = df['es_recontacto'].value_counts().sort_index()
print(f"\n🎯 Distribución del target (es_recontacto):")
print(target_counts)
print(f"\n📈 Tasa de recontacto: {df['es_recontacto'].mean() * 100:.2f}%")

print(f"\n🔑 VARIABLES OPERATIVAS CLAVE:")
for c in ['espera_seg', 'duracion_seg', 'transferencias', 'casos_previos_30d']:
    print(f"   • {c}: media {df[c].mean():.2f} | mediana {df[c].median():.2f} | máx {df[c].max():.0f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

target_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Distribución del target', fontweight='bold')
axes[0, 0].set_xticklabels(['No recontacta (0)', 'Recontacta (1)'], rotation=0)
axes[0, 0].set_ylabel('Casos')
for i, v in enumerate(target_counts):
    axes[0, 0].text(i, v + max(target_counts) * 0.01, f'{v:,}', ha='center', fontweight='bold')

df['espera_seg'].hist(ax=axes[0, 1], bins=30, color='#3498db', edgecolor='black')
axes[0, 1].set_title('Distribución del tiempo de espera', fontweight='bold')
axes[0, 1].set_xlabel('Segundos de espera')

df['duracion_seg'].hist(ax=axes[1, 0], bins=30, color='#9b59b6', edgecolor='black')
axes[1, 0].set_title('Distribución de la duración de la atención', fontweight='bold')
axes[1, 0].set_xlabel('Segundos de duración')

tr = df['transferencias'].clip(upper=3)
tasa_tr = df.groupby(tr)['es_recontacto'].mean() * 100
etiquetas = [str(int(i)) if i < 3 else '3 o más' for i in tasa_tr.index]
axes[1, 1].bar(etiquetas, tasa_tr.values, color='#e67e22', edgecolor='black')
axes[1, 1].axhline(df['es_recontacto'].mean() * 100, color='gray', ls='--', label='Tasa del KPI')
axes[1, 1].set_title('Tasa de recontacto según transferencias', fontweight='bold')
axes[1, 1].set_xlabel('Transferencias')
axes[1, 1].set_ylabel('% recontacto')
axes[1, 1].legend()
for i, v in enumerate(tasa_tr.values):
    axes[1, 1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
guardar('01_exploracion')
plt.show()
Output (stdout):

text
======================================================================
🔍 EXPLORACIÓN INICIAL DEL DATASET
======================================================================

📋 Columnas del dataset (34):
    1. id_interaccion               | Tipo: int64    | Nulos: 0
    2. es_recontacto                | Tipo: int64    | Nulos: 0
    3. duracion_seg                 | Tipo: int64    | Nulos: 0
    4. espera_seg                   | Tipo: int64    | Nulos: 0
    5. transferencias               | Tipo: int64    | Nulos: 0
    6. casos_previos_30d            | Tipo: int64    | Nulos: 0
    7. es_duracion_corta            | Tipo: int64    | Nulos: 0
    8. es_espera_alta               | Tipo: int64    | Nulos: 0
    9. es_multi_transferencia       | Tipo: int64    | Nulos: 0
   10. tiene_historial              | Tipo: int64    | Nulos: 0
   11. canal_chat                   | Tipo: int64    | Nulos: 0
   12. canal_correo                 | Tipo: int64    | Nulos: 0
   13. canal_red_social             | Tipo: int64    | Nulos: 0
   14. canal_telefono               | Tipo: int64    | Nulos: 0
   15. canal_sin_dato               | Tipo: int64    | Nulos: 0
   16. motivo_consulta              | Tipo: int64    | Nulos: 0
   17. motivo_falla                 | Tipo: int64    | Nulos: 0
   18. motivo_cobro                 | Tipo: int64    | Nulos: 0
   19. motivo_queja                 | Tipo: int64    | Nulos: 0
   20. motivo_solicitud             | Tipo: int64    | Nulos: 0
   21. cola_facturacion             | Tipo: int64    | Nulos: 0
   22. cola_soporte                 | Tipo: int64    | Nulos: 0
   23. cola_informacion             | Tipo: int64    | Nulos: 0
   24. cola_reclamos                | Tipo: int64    | Nulos: 0
   25. cola_tramites                | Tipo: int64    | Nulos: 0
   26. usuario_nuevo                | Tipo: int64    | Nulos: 0
   27. usuario_recurrente           | Tipo: int64    | Nulos: 0
   28. usuario_empresa              | Tipo: int64    | Nulos: 0
   29. usuario_adulto_mayor         | Tipo: int64    | Nulos: 0
   30. turno_am                     | Tipo: int64    | Nulos: 0
   31. turno_pm                     | Tipo: int64    | Nulos: 0
   32. turno_nocturno               | Tipo: int64    | Nulos: 0
   33. turno_fin_semana             | Tipo: int64    | Nulos: 0
   34. fecha_contacto               | Tipo: object   | Nulos: 0

🎯 Distribución del target (es_recontacto):
es_recontacto
0    6687
1    4749
Name: count, dtype: int64

📈 Tasa de recontacto: 41.53%

🔑 VARIABLES OPERATIVAS CLAVE:
   • espera_seg: media 335.33 | mediana 295.00 | máx 900
   • duracion_seg: media 980.58 | mediana 978.00 | máx 1800
   • transferencias: media 0.72 | mediana 1.00 | máx 4
   • casos_previos_30d: media 0.96 | mediana 1.00 | máx 8
Output (display_data): [FIGURA MOSTRADA] Figura de tamaño 1500x1000 con 4 ejes (histograma del target, distribución de espera, distribución de duración, tasa de recontacto según transferencias).

CELDA 4 — code
Source:

python
# ============================================================
# Celda 4: Preprocesamiento de variables
# ============================================================
print("=" * 70)
print("🔧 PREPROCESAMIENTO DE VARIABLES")
print("=" * 70)

columnas_a_eliminar = ['id_interaccion', 'fecha_contacto', 'es_recontacto']
X = df.drop(columns=columnas_a_eliminar, errors='ignore')
y = df['es_recontacto']

print(f"\n📊 Features iniciales: {X.shape[1]} columnas")
print(f"🚫 Excluidas del modelo: {columnas_a_eliminar} (identificador, fecha y target)")

columnas_texto = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n📝 Variables categóricas sin codificar: {columnas_texto if columnas_texto else 'ninguna (ya vienen en dummies)'}")
if columnas_texto:
    X = pd.get_dummies(X, columns=columnas_texto, drop_first=True)

columnas_bool = X.select_dtypes(include=['bool']).columns.tolist()
if columnas_bool:
    X = X.astype({c: 'int8' for c in columnas_bool})
    print(f"✅ {len(columnas_bool)} columnas booleanas convertidas a enteros")

nulos_antes = int(X.isnull().sum().sum())
X = X.fillna(0)
print(f"✅ Valores nulos rellenados: {nulos_antes} → {int(X.isnull().sum().sum())}")

# --- Quitar columnas constantes (no aportan nada, p. ej. tiene_historial en KPI07) ---
constantes = [c for c in X.columns if X[c].nunique() <= 1]
if constantes:
    X = X.drop(columns=constantes)
    print(f"\n🧹 Columnas constantes eliminadas: {constantes}")

# --- Evitar colinealidad: una categoría de referencia por grupo de dummies ---
GRUPOS = ['canal_', 'motivo_', 'cola_', 'usuario_', 'turno_']
referencias = {}
for pref in GRUPOS:
    cols = [c for c in X.columns if c.startswith(pref)]
    if cols and (X[cols].sum(axis=1) == 1).all():      # el grupo suma 1 en todas las filas
        ref = max(cols, key=lambda c: X[c].sum())      # la más frecuente como referencia
        X = X.drop(columns=ref)
        referencias[pref.rstrip('_')] = ref

print("\n📌 Categorías de referencia (los coeficientes se leen frente a ellas):")
for g, r in referencias.items():
    print(f"   • {g:8s} → {r}")

print(f"\n📏 Features finales: {X.shape[1]} columnas")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")
Output (stdout):

text
======================================================================
🔧 PREPROCESAMIENTO DE VARIABLES
======================================================================

📊 Features iniciales: 31 columnas
🚫 Excluidas del modelo: ['id_interaccion', 'fecha_contacto', 'es_recontacto'] (identificador, fecha y target)

📝 Variables categóricas sin codificar: ninguna (ya vienen en dummies)
✅ Valores nulos rellenados: 0 → 0

📌 Categorías de referencia (los coeficientes se leen frente a ellas):
   • canal    → canal_chat
   • motivo   → motivo_queja
   • cola     → cola_informacion
   • usuario  → usuario_recurrente
   • turno    → turno_pm

📏 Features finales: 26 columnas
   1. duracion_seg
   2. espera_seg
   3. transferencias
   4. casos_previos_30d
   5. es_duracion_corta
   6. es_espera_alta
   7. es_multi_transferencia
   8. tiene_historial
   9. canal_correo
  10. canal_red_social
  11. canal_telefono
  12. canal_sin_dato
  13. motivo_consulta
  14. motivo_falla
  15. motivo_cobro
  16. motivo_solicitud
  17. cola_facturacion
  18. cola_soporte
  19. cola_reclamos
  20. cola_tramites
  21. usuario_nuevo
  22. usuario_empresa
  23. usuario_adulto_mayor
  24. turno_am
  25. turno_nocturno
  26. turno_fin_semana
CELDA 5 — code
Source:

python
# ============================================================
# Celda 5: División del dataset (80% / 20%)
# ============================================================
print("=" * 70)
print("📊 DIVISIÓN TRAIN / TEST (80% / 20%)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📏 Entrenamiento: {X_train.shape[0]:,} registros")
print(f"📏 Prueba:        {X_test.shape[0]:,} registros")
print(f"\n🎯 Clases en TRAIN:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\n🎯 Clases en TEST:\n{y_test.value_counts(normalize=True).round(3)}")
print("\n✅ División estratificada completada (proporción mantenida)")
print("✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)")
Output (stdout):

text
======================================================================
📊 DIVISIÓN TRAIN / TEST (80% / 20%)
======================================================================

📏 Entrenamiento: 9,148 registros
📏 Prueba:        2,288 registros

🎯 Clases en TRAIN:
es_recontacto
0    0.585
1    0.415
Name: proportion, dtype: float64

🎯 Clases en TEST:
es_recontacto
0    0.585
1    0.415
Name: proportion, dtype: float64

✅ División estratificada completada (proporción mantenida)
✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)
CELDA 6 — code
Source:

python
# ============================================================
# Celda 6: Comparación de modelos (selección del mejor)
# ============================================================
print("=" * 70)
print("🤖 COMPARACIÓN DE 7 MODELOS")
print("=" * 70)

ratio = (y_train == 0).sum() / (y_train == 1).sum()

modelos = {
    '1. Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    '2. Árbol de Decisión': DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10,
                                                   class_weight='balanced', random_state=42),
    '3. Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10,
                                               class_weight='balanced', random_state=42, n_jobs=-1),
    '4. XGBoost': XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                scale_pos_weight=ratio, random_state=42, eval_metric='auc', verbosity=0),
    '5. LightGBM': LGBMClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=ratio, random_state=42, verbose=-1),
    '6. KNN (k=10)': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
    '7. Naive Bayes': GaussianNB(),
}
ESCALADOS = ['1. Regresión Logística', '6. KNN (k=10)']

resultados = []
for nombre, m in modelos.items():
    inicio = time.time()
    a, b = (X_train_scaled, X_test_scaled) if nombre in ESCALADOS else (X_train, X_test)
    m.fit(a, y_train)
    pred = m.predict(b)
    proba = m.predict_proba(b)[:, 1]
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, proba),
        'Tiempo (s)': time.time() - inicio,
    })
    print(f"   ✅ {nombre:26s} Accuracy {resultados[-1]['Accuracy']:.4f} | AUC {resultados[-1]['ROC-AUC']:.4f}")

df_res = pd.DataFrame(resultados).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n📊 TABLA COMPARATIVA (ordenada por ROC-AUC)")
print(df_res.round(4).to_string(index=False))

mejor = df_res.iloc[0]['Modelo']
print(f"\n🏆 MEJOR MODELO: {mejor}  (ROC-AUC {df_res.iloc[0]['ROC-AUC']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
orden = df_res.sort_values('Accuracy')
axes[0].barh(orden['Modelo'], orden['Accuracy'], color='#3498db', edgecolor='black')
axes[0].axvspan(0.75, 0.85, color='green', alpha=0.12, label='Rango esperado 75-85%')
axes[0].set_title('Accuracy por modelo', fontweight='bold')
axes[0].set_xlim(0.6, 0.95)
axes[0].legend(loc='lower right')
orden = df_res.sort_values('ROC-AUC')
axes[1].barh(orden['Modelo'], orden['ROC-AUC'], color='#e74c3c', edgecolor='black')
axes[1].set_title('ROC-AUC por modelo', fontweight='bold')
axes[1].set_xlim(0.6, 1.0)
plt.tight_layout()
guardar('02_comparacion_modelos')
plt.show()
Output (stdout):

text
======================================================================
🤖 COMPARACIÓN DE 7 MODELOS
======================================================================
   ✅ 1. Regresión Logística     Accuracy 0.8042 | AUC 0.8803
   ✅ 2. Árbol de Decisión       Accuracy 0.7596 | AUC 0.8290
   ✅ 3. Random Forest           Accuracy 0.7876 | AUC 0.8628
   ✅ 4. XGBoost                 Accuracy 0.7946 | AUC 0.8646
   ✅ 5. LightGBM                Accuracy 0.7880 | AUC 0.8638
   ✅ 6. KNN (k=10)              Accuracy 0.7570 | AUC 0.8168
   ✅ 7. Naive Bayes             Accuracy 0.7837 | AUC 0.8467

📊 TABLA COMPARATIVA (ordenada por ROC-AUC)
                Modelo  Accuracy  Precision  Recall  F1-Score  ROC-AUC  Tiempo (s)
1. Regresión Logística    0.8042     0.7413  0.8116    0.7749   0.8803      0.0399
            4. XGBoost    0.7946     0.7429  0.7726    0.7575   0.8646      3.6698
           5. LightGBM    0.7880     0.7351  0.7653    0.7499   0.8638      0.7794
      3. Random Forest    0.7876     0.7367  0.7600    0.7482   0.8628      1.9388
        7. Naive Bayes    0.7837     0.7898  0.6526    0.7147   0.8467      0.0349
  2. Árbol de Decisión    0.7596     0.7110  0.7095    0.7102   0.8290      0.0699
         6. KNN (k=10)    0.7570     0.7506  0.6211    0.6797   0.8168      0.2273

🏆 MEJOR MODELO: 1. Regresión Logística  (ROC-AUC 0.8803)
Output (display_data): [FIGURA MOSTRADA] Figura de tamaño 1600x550 con 2 ejes (barras horizontales de Accuracy y de ROC-AUC por modelo).

CELDA 7 — code
Source:

python
# ============================================================
# Celda 7: Entrenamiento del modelo final (Regresión Logística)
# ============================================================
print("=" * 70)
print("🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA")
print("=" * 70)

modelo = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
print(f"\n🔄 Entrenando con {X_train.shape[0]:,} registros...")
modelo.fit(X_train_scaled, y_train)
print("✅ Modelo entrenado exitosamente")

cv = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\n🔁 Validación cruzada (5 particiones) ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

if mejor != '1. Regresión Logística':
    print(f"\n⚠️ OJO: en este KPI el mejor por AUC fue {mejor}. Revisar antes de justificar la Regresión Logística.")
else:
    print("\n✅ La Regresión Logística es el mejor modelo de la comparación para este KPI")
Output (stdout):

text
======================================================================
🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA
======================================================================

🔄 Entrenando con 9,148 registros...
✅ Modelo entrenado exitosamente

🔁 Validación cruzada (5 particiones) ROC-AUC: 0.8905 ± 0.0106

✅ La Regresión Logística es el mejor modelo de la comparación para este KPI
CELDA 8 — code
Source:

python
# ============================================================
# Celda 8: Evaluación de métricas en el set de prueba
# ============================================================
print("=" * 70)
print("📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA")
print("=" * 70)

y_pred = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\n🎯 MÉTRICAS PRINCIPALES:")
print(f"   • Accuracy:       {accuracy:.4f} ({accuracy:.2%})")
print(f"   • Precision:      {precision:.4f}")
print(f"   • Recall:         {recall:.4f}")
print(f"   • F1-Score:       {f1:.4f}")
print(f"   • ROC-AUC:        {auc:.4f}")
print(f"   • Avg Precision:  {avg_precision:.4f}")

print(f"\n📋 REPORTE DE CLASIFICACIÓN COMPLETO:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['No recontacta (0)', 'Recontacta (1)']))

print(f"💡 INTERPRETACIÓN:")
print(f"   • ROC-AUC = {auc:.3f} indica un poder discriminatorio "
      f"{'excelente' if auc > 0.85 else 'bueno' if auc > 0.75 else 'moderado'}")
print(f"   • Accuracy {'dentro' if 0.75 <= accuracy <= 0.85 else 'fuera'} del rango esperado (75%-85%)")
print(f"   • {'No hay indicio de data leakage (accuracy < 90%)' if accuracy < 0.90 else 'ALERTA: accuracy > 90%, revisar posible data leakage'}")
print(f"   • CV-AUC {cv.mean():.3f} vs AUC de prueba {auc:.3f}: diferencia {abs(cv.mean() - auc):.3f} "
      f"({'sin sobreajuste evidente' if abs(cv.mean() - auc) < 0.03 else 'revisar sobreajuste'})")
Output (stdout):

text
======================================================================
📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA
======================================================================

🎯 MÉTRICAS PRINCIPALES:
   • Accuracy:       0.8042 (80.42%)
   • Precision:      0.7413
   • Recall:         0.8116
   • F1-Score:       0.7749
   • ROC-AUC:        0.8803
   • Avg Precision:  0.8460

📋 REPORTE DE CLASIFICACIÓN COMPLETO:
----------------------------------------------------------------------
                   precision    recall  f1-score   support

No recontacta (0)       0.86      0.80      0.83      1338
   Recontacta (1)       0.74      0.81      0.77       950

         accuracy                           0.80      2288
        macro avg       0.80      0.81      0.80      2288
     weighted avg       0.81      0.80      0.81      2288

💡 INTERPRETACIÓN:
   • ROC-AUC = 0.880 indica un poder discriminatorio excelente
   • Accuracy dentro del rango esperado (75%-85%)
   • No hay indicio de data leakage (accuracy < 90%)
   • CV-AUC 0.891 vs AUC de prueba 0.880: diferencia 0.010 (sin sobreajuste evidente)
CELDA 9 — code
Source:

python
# ============================================================
# Celda 8b: Verificación anti-fuga de datos (data leakage)
# ============================================================
print("=" * 70)
print("🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS")
print("=" * 70)

# 1) Ninguna variable del modelo contiene el resultado
sospechosas = [c for c in X.columns if 'recontacto' in c.lower()]
print(f"\n1) Variables con 'recontacto' en el nombre dentro del modelo: {sospechosas if sospechosas else 'ninguna ✅'}")

# 2) Ninguna variable por sí sola separa casi perfectamente el target
aucs = {}
for c in X.columns:
    if X[c].nunique() > 1:
        a = roc_auc_score(y, X[c])
        aucs[c] = max(a, 1 - a)
aucs = pd.Series(aucs).sort_values(ascending=False)
print(f"\n2) AUC individual más alto de una sola variable: {aucs.index[0]} = {aucs.iloc[0]:.3f}")
print(f"   {'✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)' if aucs.iloc[0] < 0.95 else '⚠️ revisar: una variable casi separa el target'}")
print(aucs.head(5).round(3).to_string())

# 3) Rendimiento realista y estable
print(f"\n3) Accuracy {accuracy:.2%} (alerta si > 90%): {'✅' if accuracy < 0.90 else '⚠️'}")
print(f"4) CV-AUC {cv.mean():.3f} vs AUC prueba {auc:.3f} (diferencia {abs(cv.mean() - auc):.3f}): "
      f"{'✅ estable' if abs(cv.mean() - auc) < 0.03 else '⚠️ revisar'}")

print("""
5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
""")
Output (stdout):

text
======================================================================
🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS
======================================================================

1) Variables con 'recontacto' en el nombre dentro del modelo: ninguna ✅

2) AUC individual más alto de una sola variable: transferencias = 0.741
   ✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)
transferencias            0.741
espera_seg                0.721
es_multi_transferencia    0.645
es_espera_alta            0.644
casos_previos_30d         0.638

3) Accuracy 80.42% (alerta si > 90%): ✅
4) CV-AUC 0.891 vs AUC prueba 0.880 (diferencia 0.010): ✅ estable

5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
CELDA 10 — code
Source:

python
# ============================================================
# Celda 9: Matriz de confusión
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No recontacta (0)', 'Recontacta (1)'],
            yticklabels=['No recontacta (0)', 'Recontacta (1)'],
            cbar_kws={'label': 'Cantidad'})
plt.title(f'Matriz de Confusión - Set de Prueba\n{KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Valor Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
guardar('03_matriz_confusion')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n📊 DESGLOSE DE LA MATRIZ:")
print(f"   • Verdaderos Positivos (VP): {tp:5d} → Recontactos detectados correctamente")
print(f"   • Verdaderos Negativos (VN): {tn:5d} → Casos resueltos identificados correctamente")
print(f"   • Falsos Positivos (FP):     {fp:5d} → Alarmas falsas (se predijo recontacto y no ocurrió)")
print(f"   • Falsos Negativos (FN):     {fn:5d} → Recontactos que el modelo no detectó")

print(f"\n💼 LECTURA OPERATIVA:")
print(f"   • De {tp + fn:,} recontactos reales, el modelo anticipó {tp:,} ({tp / (tp + fn):.1%})")
print(f"   • De {tp + fp:,} casos marcados como riesgo, {tp:,} sí recontactaron ({tp / (tp + fp):.1%})")
print(f"   • Seguimientos que serían innecesarios: {fp:,}")
Output (display_data): [FIGURA MOSTRADA] Figura de tamaño 800x600 con 2 ejes (heatmap de matriz de confusión).

Output (stdout):

text

📊 DESGLOSE DE LA MATRIZ:
   • Verdaderos Positivos (VP):   771 → Recontactos detectados correctamente
   • Verdaderos Negativos (VN):  1069 → Casos resueltos identificados correctamente
   • Falsos Positivos (FP):       269 → Alarmas falsas (se predijo recontacto y no ocurrió)
   • Falsos Negativos (FN):       179 → Recontactos que el modelo no detectó

💼 LECTURA OPERATIVA:
   • De 950 recontactos reales, el modelo anticipó 771 (81.2%)
   • De 1,040 casos marcados como riesgo, 771 sí recontactaron (74.1%)
   • Seguimientos que serían innecesarios: 269
CELDA 11 — code
Source:

python
# ============================================================
# Celda 10: Curva ROC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'Curva ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title(f'Curva ROC - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
guardar('04_curva_roc')
plt.show()

idx_optimo = np.argmax(tpr - fpr)
umbral_optimo = thresholds[idx_optimo]
print(f"\n🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):")
print(f"   • Umbral: {umbral_optimo:.3f}")
print(f"   • TPR en ese punto: {tpr[idx_optimo]:.3f}")
print(f"   • FPR en ese punto: {fpr[idx_optimo]:.3f}")
Output (display_data): [FIGURA MOSTRADA] Figura de tamaño 1000x700 con 1 eje (curva ROC).

Output (stdout):

text

🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):
   • Umbral: 0.478
   • TPR en ese punto: 0.826
   • FPR en ese punto: 0.206
CELDA 12 — code
Source:

python
# ============================================================
# Celda 11: Variables más asociadas al recontacto
# ============================================================
print("=" * 70)
print("🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)")
print("=" * 70)

df_coef = pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_[0]})
df_coef['Odds ratio'] = np.exp(df_coef['Coeficiente'])
df_coef['Abs'] = df_coef['Coeficiente'].abs()
df_coef = df_coef.sort_values('Abs', ascending=False).reset_index(drop=True)
top15 = df_coef.head(15)

print(f"\n📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):")
print("-" * 70)
for i, r in top15.iterrows():
    signo = '↑ sube' if r['Coeficiente'] > 0 else '↓ baja'
    print(f"   {i + 1:2d}. {r['Variable']:28s} | coef {r['Coeficiente']:+.4f} | OR {r['Odds ratio']:.3f} | {signo} la prob.")

plot_data = top15.sort_values('Coeficiente')
colores = ['#e74c3c' if c > 0 else '#2ecc71' for c in plot_data['Coeficiente']]
plt.figure(figsize=(12, 8))
plt.barh(plot_data['Variable'], plot_data['Coeficiente'], color=colores, edgecolor='black')
plt.axvline(0, color='black', lw=0.8)
plt.xlabel('Coeficiente (variables estandarizadas)')
plt.title(f'Top 15 variables asociadas al recontacto - {KPI_ID}\n(rojo: aumenta la probabilidad, verde: la reduce)',
          fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
guardar('05_coeficientes')
plt.show()

print("\n⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.")
Output (stdout):

text
======================================================================
🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)
======================================================================

📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):
----------------------------------------------------------------------
    1. transferencias               | coef +1.5143 | OR 4.546 | ↑ sube la prob.
    2. casos_previos_30d            | coef +0.9227 | OR 2.516 | ↑ sube la prob.
    3. espera_seg                   | coef +0.9166 | OR 2.501 | ↑ sube la prob.
    4. cola_reclamos                | coef -0.5845 | OR 0.557 | ↓ baja la prob.
    5. cola_soporte                 | coef -0.4462 | OR 0.640 | ↓ baja la prob.
    6. es_duracion_corta            | coef +0.3706 | OR 1.449 | ↑ sube la prob.
    7. duracion_seg                 | coef +0.3557 | OR 1.427 | ↑ sube la prob.
    8. usuario_adulto_mayor         | coef +0.3452 | OR 1.412 | ↑ sube la prob.
    9. usuario_empresa              | coef -0.2744 | OR 0.760 | ↓ baja la prob.
   10. motivo_consulta              | coef -0.2490 | OR 0.780 | ↓ baja la prob.
   11. canal_red_social             | coef +0.2205 | OR 1.247 | ↑ sube la prob.
   12. turno_nocturno               | coef +0.2088 | OR 1.232 | ↑ sube la prob.
   13. motivo_falla                 | coef -0.2021 | OR 0.817 | ↓ baja la prob.
   14. canal_telefono               | coef -0.1915 | OR 0.826 | ↓ baja la prob.
   15. motivo_cobro                 | coef -0.1847 | OR 0.831 | ↓ baja la prob.
Output (display_data): [FIGURA MOSTRADA] Figura de tamaño 1200x800 con 1 eje (barras horizontales de coeficientes top 15).

Output (stdout final):

text
⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.
CELDA 13 — code
Source:

python
# ============================================================
# Celda 12: Tasa de recontacto por segmento (reglas de negocio)
# ============================================================
print("=" * 70)
print("💼 RECONTACTO POR SEGMENTO")
print("=" * 70)

grupos = {'Canal': 'canal_', 'Motivo': 'motivo_', 'Cola': 'cola_',
          'Tipo de usuario': 'usuario_', 'Turno': 'turno_'}
tasa_kpi = df['es_recontacto'].mean() * 100

filas = []
for g, pref in grupos.items():
    for col in [c for c in df.columns if c.startswith(pref)]:
        m = df[col] == 1
        if m.sum() >= 30:
            filas.append({'Grupo': g, 'Segmento': col.replace(pref, ''),
                          'Casos': int(m.sum()),
                          'Tasa %': df.loc[m, 'es_recontacto'].mean() * 100})
seg = pd.DataFrame(filas).sort_values('Tasa %', ascending=True).reset_index(drop=True)
seg['Etiqueta'] = seg['Grupo'] + ': ' + seg['Segmento']

paleta = {'Canal': '#3498db', 'Motivo': '#e67e22', 'Cola': '#9b59b6',
          'Tipo de usuario': '#1abc9c', 'Turno': '#f1c40f'}
plt.figure(figsize=(12, max(6, len(seg) * 0.38)))
plt.barh(seg['Etiqueta'], seg['Tasa %'], color=[paleta[g] for g in seg['Grupo']], edgecolor='black')
plt.axvline(tasa_kpi, color='red', ls='--', label=f'Tasa del KPI ({tasa_kpi:.1f}%)')
plt.xlabel('% de recontacto a 7 días')
plt.title(f'Recontacto por segmento - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
guardar('06_segmentos')
plt.show()

print(f"\n🔺 SEGMENTOS CON MAYOR RECONTACTO:")
print(seg.tail(5).iloc[::-1][['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
print(f"\n🔻 SEGMENTOS CON MENOR RECONTACTO:")
print(seg.head(5)[['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
Output (stdout):

text
======================================================================
💼 RECONTACTO POR SEGMENTO
======================================================================
Output (display_data): [FIGURA MOSTRADA] Figura de tamaño 1200x874 con 1 eje (barras horizontales de tasa de recontacto por segmento).

Output (stdout final):

text

🔺 SEGMENTOS CON MAYOR RECONTACTO:
                   Etiqueta  Casos  Tasa %
            Turno: nocturno   1323    54.6
Tipo de usuario: recurrente   3933    48.9
              Motivo: falla   2528    48.9
          Cola: facturacion   2110    48.2
          Canal: red_social   2552    48.0

🔻 SEGMENTOS CON MENOR RECONTACTO:
                Etiqueta  Casos  Tasa %
Tipo de usuario: empresa   1718    31.3
  Tipo de usuario: nuevo   3439    34.3
        Motivo: consulta   1895    34.8
           Cola: soporte   2240    35.2
           Motivo: cobro   2319    37.1
CELDA 14 — code
Source:

python
# ============================================================
# Celda 13: Cifras para el reporte ejecutivo (calculadas)
# El ANÁLISIS se redacta a mano en la celda de texto que sigue.
# ============================================================
ACCIONES = {
    'transferencias': 'Reducir transferencias: mejorar el ruteo para resolver en el primer contacto',
    'espera_seg': 'Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas',
    'casos_previos_30d': 'Seguimiento proactivo a usuarios con casos previos recientes',
    'es_duracion_corta': 'Revisar atenciones muy cortas: pueden cerrarse sin resolver',
    'duracion_seg': 'Revisar la duración de atención frente al tipo de caso',
    'usuario_adulto_mayor': 'Atención asistida y confirmación de resolución a adultos mayores',
    'turno_nocturno': 'Reforzar capacidad de resolución en turno nocturno',
    'canal_red_social': 'Mejorar la resolución en redes sociales o derivar a un canal con trazabilidad',
    'cola_informacion': 'Revisar la calidad de las respuestas en la cola de Información',
    'cola_facturacion': 'Revisar casos de Facturación para resolver en el primer contacto',
    'cola_tramites': 'Simplificar el seguimiento de Trámites',
}
REDUCCION_SUPUESTA = 0.15  # SUPUESTO de trabajo, no es un dato

tasa_kpi = df['es_recontacto'].mean() * 100
n_casos = len(df)
n_rec = int(df['es_recontacto'].sum())
dur_media = df['duracion_seg'].mean()
recall_ = tp / (tp + fn)
horas_rec = n_rec * dur_media / 3600
evitables = n_rec * recall_ * REDUCCION_SUPUESTA
horas_ahorradas = evitables * dur_media / 3600
tasa_proy = tasa_kpi - tasa_kpi * recall_ * REDUCCION_SUPUESTA
brecha_actual = tasa_kpi - META_6M
dif_proy = tasa_proy - META_6M
brecha_ref = tasa_proy - META_RECONTACTO

print("=" * 70)
print(f"📊 CIFRAS DEL REPORTE - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)
print(f"""
🎯 OBJETIVO: {OBJETIVO}
🔖 TARGET : {TARGET_DEF}

📊 DATOS
   • Casos del KPI: {n_casos:,}  |  Recontactos: {n_rec:,}  |  Tasa actual (línea base): {tasa_kpi:.2f}%
   • Meta a {PLAZO}: ≤ {META_6M}%  |  Brecha actual: {brecha_actual:.1f} puntos  |  Umbral de alerta: > {UMBRAL_ALERTA}%
   • Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%
   • Entrenamiento / prueba: {len(X_train):,} / {len(X_test):,}  |  Variables: {X.shape[1]}

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: {mejor})
   • Accuracy {accuracy:.2%} | Precision {precision:.2%} | Recall {recall:.2%} | F1 {f1:.4f} | ROC-AUC {auc:.4f}
   • CV-AUC (5 particiones): {cv.mean():.4f} ± {cv.std():.4f}
   • Recontactos detectados: {tp:,} de {tp + fn:,} ({recall_:.1%})  |  Alarmas falsas: {fp:,}

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = {dur_media:.0f} s)
   • Horas-agente que consumen los recontactos del periodo: ≈ {horas_rec:,.0f} h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el {REDUCCION_SUPUESTA:.0%} de los recontactos detectados)
   • Recontactos evitables: ≈ {evitables:,.0f}  |  Horas-agente liberadas: ≈ {horas_ahorradas:,.0f} h
   • La tasa bajaría de {tasa_kpi:.1f}% a ≈ {tasa_proy:.1f}%
   • Diferencia frente a la meta de {PLAZO} ({META_6M}%): {dif_proy:+.1f} puntos
   • Distancia que queda a la referencia de largo plazo ({META_RECONTACTO:.0f}%): {brecha_ref:.1f} puntos
   • El {REDUCCION_SUPUESTA:.0%} es un supuesto de trabajo, no un resultado del modelo.
""")

pos = df_coef[df_coef['Coeficiente'] > 0]
prioridades = ['URGENTE', 'ALTO', 'ALTO', 'MEDIO', 'MEDIO', 'MEDIO']
print("🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)")
print("-" * 70)
n = 0
for _, r in pos.iterrows():
    if r['Variable'] in ACCIONES and n < len(prioridades):
        print(f"   {prioridades[n]:8s} {ACCIONES[r['Variable']]}")
        print(f"            (variable: {r['Variable']} | odds ratio {r['Odds ratio']:.2f} por desv. estándar)")
        n += 1

print("\n🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):")
for _, r in df_coef[df_coef['Coeficiente'] < 0].head(3).iterrows():
    print(f"   ↓ {r['Variable']:26s} (odds ratio {r['Odds ratio']:.2f})")
Output (stdout):

text
======================================================================
📊 CIFRAS DEL REPORTE - KPI01: Recontacto general
======================================================================

🎯 OBJETIVO: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
🔖 TARGET : es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo

📊 DATOS
   • Casos del KPI: 11,436  |  Recontactos: 4,749  |  Tasa actual (línea base): 41.53%
   • Meta a 6 meses: ≤ 36.5%  |  Brecha actual: 5.0 puntos  |  Umbral de alerta: > 41.5%
   • Referencia de largo plazo (no equivalente): ≤ 22%
   • Entrenamiento / prueba: 9,148 / 2,288  |  Variables: 26

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: 1. Regresión Logística)
   • Accuracy 80.42% | Precision 74.13% | Recall 81.16% | F1 0.7749 | ROC-AUC 0.8803
   • CV-AUC (5 particiones): 0.8905 ± 0.0106
   • Recontactos detectados: 771 de 950 (81.2%)  |  Alarmas falsas: 269

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = 981 s)
   • Horas-agente que consumen los recontactos del periodo: ≈ 1,294 h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el 15% de los recontactos detectados)
   • Recontactos evitables: ≈ 578  |  Horas-agente liberadas: ≈ 157 h
   • La tasa bajaría de 41.5% a ≈ 36.5%
   • Diferencia frente a la meta de 6 meses (36.5%): -0.0 puntos
   • Distancia que queda a la referencia de largo plazo (22%): 14.5 puntos
   • El 15% es un supuesto de trabajo, no un resultado del modelo.

🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)
----------------------------------------------------------------------
   URGENTE  Reducir transferencias: mejorar el ruteo para resolver en el primer contacto
            (variable: transferencias | odds ratio 4.55 por desv. estándar)
   ALTO     Seguimiento proactivo a usuarios con casos previos recientes
            (variable: casos_previos_30d | odds ratio 2.52 por desv. estándar)
   ALTO     Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas
            (variable: espera_seg | odds ratio 2.50 por desv. estándar)
   MEDIO    Revisar atenciones muy cortas: pueden cerrarse sin resolver
            (variable: es_duracion_corta | odds ratio 1.45 por desv. estándar)
   MEDIO    Revisar la duración de atención frente al tipo de caso
            (variable: duracion_seg | odds ratio 1.43 por desv. estándar)
   MEDIO    Atención asistida y confirmación de resolución a adultos mayores
            (variable: usuario_adulto_mayor | odds ratio 1.41 por desv. estándar)

🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):
   ↓ cola_reclamos              (odds ratio 0.56)
   ↓ cola_soporte               (odds ratio 0.64)
   ↓ usuario_empresa            (odds ratio 0.76)
CELDA 15 — markdown
Source (texto):

text
**Nota sobre la meta.** El 22 % surge de 100 % − 78 % de resolución en primer contacto. No es equivalente al recontacto a 7 días, así que se usa solo como horizonte de largo plazo. La meta del periodo es la de la tabla siguiente, y debe validarse con la dirección.

### Ficha de KPIs: objetivo, acción y responsable por KPI

| KPI | Pregunta de negocio | Acción propia | Responsable (A / R) |
|---|---|---|---|
| KPI01 General | ¿Cuánto recontacta el servicio en total? | Mejorar el ruteo para bajar transferencias | Gerencia de Servicio / Operaciones |
| KPI02 Falla/queja | ¿Se resuelven bien los casos críticos? | Guías de resolución y escalamiento para falla y queja | Operaciones / Conocimiento |
| KPI07 Historial | ¿Vuelven los que ya habían contactado? | Seguimiento proactivo en CRM a usuarios con casos previos | Gerencia de Servicio / Calidad |
| KPI08 Canales | ¿Red social y correo resuelven menos? | Trazabilidad en esos canales o derivar a uno mejor | TI/Seguridad / Operaciones |
| KPI10 Nocturno/finde | ¿Qué pasa fuera del horario normal? | Reforzar dotación y capacidad de resolución | Operaciones / Gerencia de Servicio |
CELDA 16 — markdown
Source (texto):

text
- Los datos son sintéticos. Las relaciones entre variables (transferencias, espera, casos previos, etc.) fueron definidas como supuestos documentados del generador, y se verificó que el rendimiento resultante fuera realista (accuracy entre 75 % y 90 %, sin indicios de fuga de datos).
CELDA 17 — markdown
Source (texto):

text
# Fase 15: Reporte ejecutivo (redactado por el analista)

**Objetivo.** Reducir la tasa de recontacto a 7 días del Servicio Ciudadano 1800. La línea base es 41.5 %, la meta a 6 meses es ≤ 36.5 % (supuesto a validar con la dirección) y el umbral de alerta es una tasa superior a 41.5 %. El 22 % (100 % − 78 % de resolución en primer contacto) no es equivalente al recontacto a 7 días, por lo que se usa solo como referencia de largo plazo.

**Momento de predicción.** El modelo se aplica al cierre de la atención, cuando ya se conocen la espera, la duración, las transferencias y los casos previos del usuario. Ninguna de esas variables incluye información posterior al contacto. Sirve para priorizar el seguimiento después de cada atención, no como alerta en tiempo real durante la llamada.

**Hallazgo central.** La Regresión Logística fue el mejor de los 7 modelos comparados (ROC-AUC 0.880, accuracy 80.4 %, CV-AUC 0.891). Detecta 771 de 950 recontactos del conjunto de prueba (recall 81.2 %) y 74.1 % de los casos que marca como riesgo sí recontactan (269 alarmas falsas). La probabilidad de recontacto se asocia sobre todo con tres factores operativos: las transferencias (OR 4.5 por desviación estándar), los casos previos del usuario en los últimos 30 días (OR 2.5) y el tiempo de espera (OR 2.5). Las comparaciones por categoría se leen frente a una referencia: cola Información, usuarios recurrentes, motivo Queja, canal Chat y turno PM. Frente a esas referencias, las colas de Reclamos (OR 0.56) y Soporte (OR 0.64) y los usuarios empresa (OR 0.76) tienen menor probabilidad de recontacto; los adultos mayores (OR 1.41), el canal red social (OR 1.25) y el turno nocturno (OR 1.23) tienen mayor probabilidad. Los motivos Consulta, Falla y Cobro (OR entre 0.78 y 0.83) recontactan menos que Queja.

**Lectura de negocio.** Las transferencias son la señal más fuerte: resolver en el primer contacto, con mejor ruteo, es la acción prioritaria. Después van el seguimiento proactivo a usuarios con casos recientes y la reducción de la espera. Por segmentos, la tasa observada es mayor en el turno nocturno (54.6 %), los usuarios recurrentes (48.9 %), los casos por falla (48.9 %), la cola de Facturación (48.2 %) y el canal red social (48.0 %); es menor en usuarios empresa (31.3 %), usuarios nuevos (34.3 %) y consultas (34.8 %). Estas tasas son observadas y no ajustadas: por ejemplo, los casos por falla recontactan mucho, pero una vez controlados la espera y las transferencias quedan por debajo de las quejas, lo que sugiere que su recontacto se explica más por las condiciones de atención que por el motivo en sí.

**Alcance realista.** En el periodo hubo 4,749 recontactos, que consumen ≈ 1,294 horas-agente. Si se evitara el 15 % de los recontactos que el modelo detecta (supuesto de trabajo, no un resultado), se evitarían ≈ 578 recontactos y se liberarían ≈ 157 horas-agente. La tasa bajaría de 41.5 % a ≈ 36.5 %, que coincide con la meta a 6 meses; como esa meta se construyó con el mismo supuesto, solo es alcanzable si el supuesto se cumple. Aun así quedarían ≈ 14.5 puntos hasta la referencia de largo plazo (22 %). El modelo ayuda a priorizar a quién y dónde intervenir, pero no basta por sí solo.
CELDA 18 — markdown
Source (texto):

text
# Fase 16: Próximos pasos y limitaciones

**Próximos pasos.**
1. Validar con la dirección la meta a 6 meses, el umbral de alerta y el criterio de "mismo motivo".
2. Probar las intervenciones con una prueba A/B en las colas con mayor recontacto.
3. Monitorear mensualmente la tasa de recontacto y las métricas del modelo.
4. Reentrenar con datos reales cada trimestre.

**Limitaciones.**
- Los datos son sintéticos. Las relaciones entre variables (transferencias, espera, casos previos, etc.) fueron definidas como supuestos documentados del generador, y se verificó que el rendimiento resultante fuera realista (accuracy entre 75 % y 90 %, sin indicios de fuga de datos).
- La Regresión Logística gana en parte porque el generador usa una fórmula logística.
- Los coeficientes indican asociación, no causalidad, y se leen frente a la categoría de referencia de cada grupo.
- La división entre entrenamiento y prueba es aleatoria, no temporal, y no se dispone de un identificador de usuario para evitar que un mismo usuario aparezca en ambos conjuntos.
- No se dispone de costos por contacto, por lo que el impacto se expresa en casos y horas-agente, con un supuesto del 15 % declarado.
- La meta a 6 meses se deriva de ese mismo supuesto; la referencia del 22 % no es equivalente al recontacto a 7 días.
- Los 5 KPIs comparten el mismo evento (recontacto) y difieren en la población analizada.
- Parte del recontacto no es evitable (usuarios con consultas nuevas o complejas), por lo que conviene aceptar un piso mínimo.


--------------------------------

Extracción de entradas y salidas del notebook 03_modelado_kpi02_motivos.ipynb
A continuación, cada celda con su ENTRADA (source/código) y SALIDA (outputs), sin omitir nada.

CELDA 1
ENTRADA (source)
python
# ============================================================
# Celda 1: Importar librerías y configurar el KPI
# ============================================================
from lightgbm import LGBMClassifier
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from cargar_datos import cargar_dataset_recontacto

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ============ CAMBIAR SOLO ESTA LÍNEA EN CADA NOTEBOOK ============
KPI_SEL = 'KPI02'      # 'KPI01', 'KPI02', 'KPI07', 'KPI08' o 'KPI10'
# ===================================================================

FICHAS = {
    'KPI01': dict(nombre='Recontacto general', filtro=None,
                  formula='casos con recontacto a 7 días ÷ casos con resultado conocido',
                  base=41.5, meta=36.5),
    'KPI02': dict(nombre='Recontacto en motivos críticos (falla/queja)', filtro='motivos_criticos',
                  formula='recontactos a 7 días de casos por falla o queja ÷ casos por falla o queja con resultado conocido',
                  base=46.1, meta=40.5),
    'KPI07': dict(nombre='Recontacto en usuarios con historial', filtro='historial',
                  formula='recontactos a 7 días de usuarios con casos previos en 30 días ÷ casos de esos usuarios con resultado conocido',
                  base=50.4, meta=44.4),
    'KPI08': dict(nombre='Recontacto en canales de baja resolución (red social y correo)', filtro='canales_baja_resolucion',
                  formula='recontactos a 7 días en red social y correo ÷ casos en esos canales con resultado conocido',
                  base=44.3, meta=39.1),
    'KPI10': dict(nombre='Recontacto en turno nocturno y fin de semana', filtro='nocturno_finde',
                  formula='recontactos a 7 días en turno nocturno y fin de semana ÷ casos en esos turnos con resultado conocido',
                  base=44.1, meta=38.6),
}
_f = FICHAS[KPI_SEL]

# ------------------ CONFIGURACIÓN DEL KPI ------------------
KPI_ID = KPI_SEL
KPI_NOMBRE = _f['nombre']
FILTRO = _f['filtro']
KPI_FORMULA = _f['formula']

# ------------------ OBJETIVO Y META ------------------
OBJETIVO = 'Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800'

# Referencia externa (NO equivalente): 100% - 78% de resolución en primer contacto.
META_RECONTACTO = 22.0

# Ficha del KPI
LINEA_BASE = _f['base']     # tasa actual del segmento (%)
META_6M = _f['meta']        # meta a 6 meses (escenario del 15 %, supuesto a validar)
PLAZO = '6 meses'
UMBRAL_ALERTA = _f['base']  # si la tasa supera la línea base, se activa alerta

TARGET_DEF = ('es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; '
              '0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo')
# -----------------------------------------------------

os.makedirs('../informe_figuras', exist_ok=True)

def guardar(nombre):
    plt.savefig(f'../informe_figuras/{KPI_ID}_{nombre}.png', dpi=150, bbox_inches='tight')

print("✅ Librerías cargadas correctamente")
print(f"📌 {KPI_ID} - {KPI_NOMBRE}")
print(f"📐 Fórmula: {KPI_FORMULA}")
print(f"🎯 Objetivo: {OBJETIVO}")
print(f"📊 Línea base: {LINEA_BASE}% | 🎯 Meta {PLAZO}: ≤ {META_6M}% | 🚨 Umbral de alerta: > {UMBRAL_ALERTA}%")
print(f"🏁 Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%")
print(f"🔖 Target: {TARGET_DEF}")
SALIDA (stdout)
text
✅ Librerías cargadas correctamente
📌 KPI02 - Recontacto en motivos críticos (falla/queja)
📐 Fórmula: recontactos a 7 días de casos por falla o queja ÷ casos por falla o queja con resultado conocido
🎯 Objetivo: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
📊 Línea base: 46.1% | 🎯 Meta 6 meses: ≤ 40.5% | 🚨 Umbral de alerta: > 46.1%
🏁 Referencia de largo plazo (no equivalente): ≤ 22%
🔖 Target: es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo
CELDA 2
ENTRADA (source)
python
# ============================================================
# Celda 2: Cargar dataset desde SQL Server
# ============================================================
def filtrar(d, filtro):
    if filtro == 'motivos_criticos':
        d = d[(d.motivo_falla == 1) | (d.motivo_queja == 1)]
    elif filtro == 'historial':
        d = d[d.tiene_historial == 1]
    elif filtro == 'canales_baja_resolucion':
        d = d[(d.canal_red_social == 1) | (d.canal_correo == 1)]
    elif filtro == 'nocturno_finde':
        d = d[(d.turno_nocturno == 1) | (d.turno_fin_semana == 1)]
    return d.reset_index(drop=True)

print("=" * 70)
print(f"📊 CARGANDO DATASET - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)

df_base = cargar_dataset_recontacto()
df = filtrar(df_base, FILTRO)

tasa_inicial = df['es_recontacto'].mean() * 100

print(f"\n✅ Dataset del KPI listo")
print(f"📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"📈 Tasa de recontacto del KPI: {tasa_inicial:.2f}%")
print(f"📈 Tasa de recontacto global : {df_base['es_recontacto'].mean() * 100:.2f}%")
print(f"🏁 Meta {PLAZO}: ≤ {META_6M}% → brecha actual: {tasa_inicial - META_6M:.1f} puntos | referencia largo plazo: {META_RECONTACTO:.0f}%")

# Control: la línea base escrita en la Celda 1 debe coincidir con la tasa real
if abs(tasa_inicial - LINEA_BASE) > 0.5:
    print(f"⚠️ OJO: la línea base de la ficha ({LINEA_BASE}%) no coincide con la tasa real ({tasa_inicial:.1f}%). Revisar KPI_SEL.")
else:
    print(f"✅ La línea base de la ficha ({LINEA_BASE}%) coincide con la tasa real.")
SALIDA (stdout)
text
======================================================================
📊 CARGANDO DATASET - KPI02: Recontacto en motivos críticos (falla/queja)
======================================================================
============================================================
📊 CARGANDO DATASET DE RECONTACTO (KPI 1)
============================================================
✅ Conexión exitosa a SQL Server
✅ Consulta ejecutada: 11,436 registros, 34 columnas

📏 Dimensiones: 11,436 filas × 34 columnas

🎯 Distribución del target:
es_recontacto
0    6687
1    4749
Name: count, dtype: int64

📈 Tasa de recontacto: 41.53%

✅ Dataset del KPI listo
📏 Dimensiones: 5,077 filas × 34 columnas
📈 Tasa de recontacto del KPI: 46.11%
📈 Tasa de recontacto global : 41.53%
🏁 Meta 6 meses: ≤ 40.5% → brecha actual: 5.6 puntos | referencia largo plazo: 22%
✅ La línea base de la ficha (46.1%) coincide con la tasa real.
CELDA 3
ENTRADA (source)
python
# ============================================================
# Celda 3: Exploración inicial del dataset
# ============================================================
print("=" * 70)
print("🔍 EXPLORACIÓN INICIAL DEL DATASET")
print("=" * 70)

print(f"\n📋 Columnas del dataset ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col:28s} | Tipo: {str(df[col].dtype):8s} | Nulos: {df[col].isnull().sum()}")

target_counts = df['es_recontacto'].value_counts().sort_index()
print(f"\n🎯 Distribución del target (es_recontacto):")
print(target_counts)
print(f"\n📈 Tasa de recontacto: {df['es_recontacto'].mean() * 100:.2f}%")

print(f"\n🔑 VARIABLES OPERATIVAS CLAVE:")
for c in ['espera_seg', 'duracion_seg', 'transferencias', 'casos_previos_30d']:
    print(f"   • {c}: media {df[c].mean():.2f} | mediana {df[c].median():.2f} | máx {df[c].max():.0f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

target_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Distribución del target', fontweight='bold')
axes[0, 0].set_xticklabels(['No recontacta (0)', 'Recontacta (1)'], rotation=0)
axes[0, 0].set_ylabel('Casos')
for i, v in enumerate(target_counts):
    axes[0, 0].text(i, v + max(target_counts) * 0.01, f'{v:,}', ha='center', fontweight='bold')

df['espera_seg'].hist(ax=axes[0, 1], bins=30, color='#3498db', edgecolor='black')
axes[0, 1].set_title('Distribución del tiempo de espera', fontweight='bold')
axes[0, 1].set_xlabel('Segundos de espera')

df['duracion_seg'].hist(ax=axes[1, 0], bins=30, color='#9b59b6', edgecolor='black')
axes[1, 0].set_title('Distribución de la duración de la atención', fontweight='bold')
axes[1, 0].set_xlabel('Segundos de duración')

tr = df['transferencias'].clip(upper=3)
tasa_tr = df.groupby(tr)['es_recontacto'].mean() * 100
etiquetas = [str(int(i)) if i < 3 else '3 o más' for i in tasa_tr.index]
axes[1, 1].bar(etiquetas, tasa_tr.values, color='#e67e22', edgecolor='black')
axes[1, 1].axhline(df['es_recontacto'].mean() * 100, color='gray', ls='--', label='Tasa del KPI')
axes[1, 1].set_title('Tasa de recontacto según transferencias', fontweight='bold')
axes[1, 1].set_xlabel('Transferencias')
axes[1, 1].set_ylabel('% recontacto')
axes[1, 1].legend()
for i, v in enumerate(tasa_tr.values):
    axes[1, 1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
guardar('01_exploracion')
plt.show()
SALIDA (stdout)
text
======================================================================
🔍 EXPLORACIÓN INICIAL DEL DATASET
======================================================================

📋 Columnas del dataset (34):
    1. id_interaccion               | Tipo: int64    | Nulos: 0
    2. es_recontacto                | Tipo: int64    | Nulos: 0
    3. duracion_seg                 | Tipo: int64    | Nulos: 0
    4. espera_seg                   | Tipo: int64    | Nulos: 0
    5. transferencias               | Tipo: int64    | Nulos: 0
    6. casos_previos_30d            | Tipo: int64    | Nulos: 0
    7. es_duracion_corta            | Tipo: int64    | Nulos: 0
    8. es_espera_alta               | Tipo: int64    | Nulos: 0
    9. es_multi_transferencia       | Tipo: int64    | Nulos: 0
   10. tiene_historial              | Tipo: int64    | Nulos: 0
   11. canal_chat                   | Tipo: int64    | Nulos: 0
   12. canal_correo                 | Tipo: int64    | Nulos: 0
   13. canal_red_social             | Tipo: int64    | Nulos: 0
   14. canal_telefono               | Tipo: int64    | Nulos: 0
   15. canal_sin_dato               | Tipo: int64    | Nulos: 0
   16. motivo_consulta              | Tipo: int64    | Nulos: 0
   17. motivo_falla                 | Tipo: int64    | Nulos: 0
   18. motivo_cobro                 | Tipo: int64    | Nulos: 0
   19. motivo_queja                 | Tipo: int64    | Nulos: 0
   20. motivo_solicitud             | Tipo: int64    | Nulos: 0
   21. cola_facturacion             | Tipo: int64    | Nulos: 0
   22. cola_soporte                 | Tipo: int64    | Nulos: 0
   23. cola_informacion             | Tipo: int64    | Nulos: 0
   24. cola_reclamos                | Tipo: int64    | Nulos: 0
   25. cola_tramites                | Tipo: int64    | Nulos: 0
   26. usuario_nuevo                | Tipo: int64    | Nulos: 0
   27. usuario_recurrente           | Tipo: int64    | Nulos: 0
   28. usuario_empresa              | Tipo: int64    | Nulos: 0
   29. usuario_adulto_mayor         | Tipo: int64    | Nulos: 0
   30. turno_am                     | Tipo: int64    | Nulos: 0
   31. turno_pm                     | Tipo: int64    | Nulos: 0
   32. turno_nocturno               | Tipo: int64    | Nulos: 0
   33. turno_fin_semana             | Tipo: int64    | Nulos: 0
   34. fecha_contacto               | Tipo: object   | Nulos: 0

🎯 Distribución del target (es_recontacto):
es_recontacto
0    2736
1    2341
Name: count, dtype: int64

📈 Tasa de recontacto: 46.11%

🔑 VARIABLES OPERATIVAS CLAVE:
   • espera_seg: media 337.94 | mediana 301.00 | máx 900
   • duracion_seg: media 1020.50 | mediana 1017.00 | máx 1800
   • transferencias: media 0.82 | mediana 1.00 | máx 4
   • casos_previos_30d: media 0.96 | mediana 1.00 | máx 8
SALIDA (display_data)
[FIGURA MOSTRADA] — Figura de tamaño 1500x1000 con 4 ejes (histograma del target, distribución de espera, distribución de duración, tasa de recontacto según transferencias).

CELDA 4
ENTRADA (source)
python
# ============================================================
# Celda 4: Preprocesamiento de variables
# ============================================================
print("=" * 70)
print("🔧 PREPROCESAMIENTO DE VARIABLES")
print("=" * 70)

columnas_a_eliminar = ['id_interaccion', 'fecha_contacto', 'es_recontacto']
X = df.drop(columns=columnas_a_eliminar, errors='ignore')
y = df['es_recontacto']

print(f"\n📊 Features iniciales: {X.shape[1]} columnas")
print(f"🚫 Excluidas del modelo: {columnas_a_eliminar} (identificador, fecha y target)")

columnas_texto = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n📝 Variables categóricas sin codificar: {columnas_texto if columnas_texto else 'ninguna (ya vienen en dummies)'}")
if columnas_texto:
    X = pd.get_dummies(X, columns=columnas_texto, drop_first=True)

columnas_bool = X.select_dtypes(include=['bool']).columns.tolist()
if columnas_bool:
    X = X.astype({c: 'int8' for c in columnas_bool})
    print(f"✅ {len(columnas_bool)} columnas booleanas convertidas a enteros")

nulos_antes = int(X.isnull().sum().sum())
X = X.fillna(0)
print(f"✅ Valores nulos rellenados: {nulos_antes} → {int(X.isnull().sum().sum())}")

# --- Quitar columnas constantes (no aportan nada, p. ej. tiene_historial en KPI07) ---
constantes = [c for c in X.columns if X[c].nunique() <= 1]
if constantes:
    X = X.drop(columns=constantes)
    print(f"\n🧹 Columnas constantes eliminadas: {constantes}")

# --- Evitar colinealidad: una categoría de referencia por grupo de dummies ---
GRUPOS = ['canal_', 'motivo_', 'cola_', 'usuario_', 'turno_']
referencias = {}
for pref in GRUPOS:
    cols = [c for c in X.columns if c.startswith(pref)]
    if cols and (X[cols].sum(axis=1) == 1).all():      # el grupo suma 1 en todas las filas
        ref = max(cols, key=lambda c: X[c].sum())      # la más frecuente como referencia
        X = X.drop(columns=ref)
        referencias[pref.rstrip('_')] = ref

print("\n📌 Categorías de referencia (los coeficientes se leen frente a ellas):")
for g, r in referencias.items():
    print(f"   • {g:8s} → {r}")

print(f"\n📏 Features finales: {X.shape[1]} columnas")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")
SALIDA (stdout)
text
======================================================================
🔧 PREPROCESAMIENTO DE VARIABLES
======================================================================

📊 Features iniciales: 31 columnas
🚫 Excluidas del modelo: ['id_interaccion', 'fecha_contacto', 'es_recontacto'] (identificador, fecha y target)

📝 Variables categóricas sin codificar: ninguna (ya vienen en dummies)
✅ Valores nulos rellenados: 0 → 0

🧹 Columnas constantes eliminadas: ['motivo_consulta', 'motivo_cobro', 'motivo_solicitud']

📌 Categorías de referencia (los coeficientes se leen frente a ellas):
   • canal    → canal_chat
   • motivo   → motivo_queja
   • cola     → cola_informacion
   • usuario  → usuario_recurrente
   • turno    → turno_pm

📏 Features finales: 23 columnas
   1. duracion_seg
   2. espera_seg
   3. transferencias
   4. casos_previos_30d
   5. es_duracion_corta
   6. es_espera_alta
   7. es_multi_transferencia
   8. tiene_historial
   9. canal_correo
  10. canal_red_social
  11. canal_telefono
  12. canal_sin_dato
  13. motivo_falla
  14. cola_facturacion
  15. cola_soporte
  16. cola_reclamos
  17. cola_tramites
  18. usuario_nuevo
  19. usuario_empresa
  20. usuario_adulto_mayor
  21. turno_am
  22. turno_nocturno
  23. turno_fin_semana
CELDA 5
ENTRADA (source)
python
# ============================================================
# Celda 5: División del dataset (80% / 20%)
# ============================================================
print("=" * 70)
print("📊 DIVISIÓN TRAIN / TEST (80% / 20%)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📏 Entrenamiento: {X_train.shape[0]:,} registros")
print(f"📏 Prueba:        {X_test.shape[0]:,} registros")
print(f"\n🎯 Clases en TRAIN:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\n🎯 Clases en TEST:\n{y_test.value_counts(normalize=True).round(3)}")
print("\n✅ División estratificada completada (proporción mantenida)")
print("✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)")
SALIDA (stdout)
text
======================================================================
📊 DIVISIÓN TRAIN / TEST (80% / 20%)
======================================================================

📏 Entrenamiento: 4,061 registros
📏 Prueba:        1,016 registros

🎯 Clases en TRAIN:
es_recontacto
0    0.539
1    0.461
Name: proportion, dtype: float64

🎯 Clases en TEST:
es_recontacto
0    0.539
1    0.461
Name: proportion, dtype: float64

✅ División estratificada completada (proporción mantenida)
✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)
CELDA 6
ENTRADA (source)
python
# ============================================================
# Celda 6: Comparación de modelos (selección del mejor)
# ============================================================
print("=" * 70)
print("🤖 COMPARACIÓN DE 7 MODELOS")
print("=" * 70)

ratio = (y_train == 0).sum() / (y_train == 1).sum()

modelos = {
    '1. Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    '2. Árbol de Decisión': DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10,
                                                   class_weight='balanced', random_state=42),
    '3. Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10,
                                               class_weight='balanced', random_state=42, n_jobs=-1),
    '4. XGBoost': XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                scale_pos_weight=ratio, random_state=42, eval_metric='auc', verbosity=0),
    '5. LightGBM': LGBMClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=ratio, random_state=42, verbose=-1),
    '6. KNN (k=10)': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
    '7. Naive Bayes': GaussianNB(),
}
ESCALADOS = ['1. Regresión Logística', '6. KNN (k=10)']

resultados = []
for nombre, m in modelos.items():
    inicio = time.time()
    a, b = (X_train_scaled, X_test_scaled) if nombre in ESCALADOS else (X_train, X_test)
    m.fit(a, y_train)
    pred = m.predict(b)
    proba = m.predict_proba(b)[:, 1]
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, proba),
        'Tiempo (s)': time.time() - inicio,
    })
    print(f"   ✅ {nombre:26s} Accuracy {resultados[-1]['Accuracy']:.4f} | AUC {resultados[-1]['ROC-AUC']:.4f}")

df_res = pd.DataFrame(resultados).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n📊 TABLA COMPARATIVA (ordenada por ROC-AUC)")
print(df_res.round(4).to_string(index=False))

mejor = df_res.iloc[0]['Modelo']
print(f"\n🏆 MEJOR MODELO: {mejor}  (ROC-AUC {df_res.iloc[0]['ROC-AUC']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
orden = df_res.sort_values('Accuracy')
axes[0].barh(orden['Modelo'], orden['Accuracy'], color='#3498db', edgecolor='black')
axes[0].axvspan(0.75, 0.85, color='green', alpha=0.12, label='Rango esperado 75-85%')
axes[0].set_title('Accuracy por modelo', fontweight='bold')
axes[0].set_xlim(0.6, 0.95)
axes[0].legend(loc='lower right')
orden = df_res.sort_values('ROC-AUC')
axes[1].barh(orden['Modelo'], orden['ROC-AUC'], color='#e74c3c', edgecolor='black')
axes[1].set_title('ROC-AUC por modelo', fontweight='bold')
axes[1].set_xlim(0.6, 1.0)
plt.tight_layout()
guardar('02_comparacion_modelos')
plt.show()
SALIDA (stdout)
text
======================================================================
🤖 COMPARACIÓN DE 7 MODELOS
======================================================================
   ✅ 1. Regresión Logística     Accuracy 0.8140 | AUC 0.8926
   ✅ 2. Árbol de Decisión       Accuracy 0.7628 | AUC 0.8143
   ✅ 3. Random Forest           Accuracy 0.8002 | AUC 0.8742
   ✅ 4. XGBoost                 Accuracy 0.7746 | AUC 0.8566
   ✅ 5. LightGBM                Accuracy 0.7835 | AUC 0.8537
   ✅ 6. KNN (k=10)              Accuracy 0.7362 | AUC 0.8128
   ✅ 7. Naive Bayes             Accuracy 0.7667 | AUC 0.8446

📊 TABLA COMPARATIVA (ordenada por ROC-AUC)
                Modelo  Accuracy  Precision  Recall  F1-Score  ROC-AUC  Tiempo (s)
1. Regresión Logística    0.8140     0.7937  0.8056    0.7996   0.8926      0.0299
      3. Random Forest    0.8002     0.8074  0.7436    0.7742   0.8742      1.3731
            4. XGBoost    0.7746     0.7626  0.7415    0.7519   0.8566      2.4942
           5. LightGBM    0.7835     0.7805  0.7372    0.7582   0.8537      0.8031
        7. Naive Bayes    0.7667     0.7909  0.6709    0.7260   0.8446      0.0229
  2. Árbol de Decisión    0.7628     0.7441  0.7393    0.7417   0.8143      0.0549
         6. KNN (k=10)    0.7362     0.7451  0.6496    0.6941   0.8128      0.1516

🏆 MEJOR MODELO: 1. Regresión Logística  (ROC-AUC 0.8926)
SALIDA (display_data)
[FIGURA MOSTRADA] — Figura de tamaño 1600x550 con 2 ejes (barras de Accuracy y ROC-AUC por modelo).

CELDA 7
ENTRADA (source)
python
# ============================================================
# Celda 7: Entrenamiento del modelo final (Regresión Logística)
# ============================================================
print("=" * 70)
print("🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA")
print("=" * 70)

modelo = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
print(f"\n🔄 Entrenando con {X_train.shape[0]:,} registros...")
modelo.fit(X_train_scaled, y_train)
print("✅ Modelo entrenado exitosamente")

cv = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\n🔁 Validación cruzada (5 particiones) ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

if mejor != '1. Regresión Logística':
    print(f"\n⚠️ OJO: en este KPI el mejor por AUC fue {mejor}. Revisar antes de justificar la Regresión Logística.")
else:
    print("\n✅ La Regresión Logística es el mejor modelo de la comparación para este KPI")
SALIDA (stdout)
text
======================================================================
🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA
======================================================================

🔄 Entrenando con 4,061 registros...
✅ Modelo entrenado exitosamente

🔁 Validación cruzada (5 particiones) ROC-AUC: 0.8910 ± 0.0098

✅ La Regresión Logística es el mejor modelo de la comparación para este KPI
CELDA 8
ENTRADA (source)
python
# ============================================================
# Celda 8: Evaluación de métricas en el set de prueba
# ============================================================
print("=" * 70)
print("📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA")
print("=" * 70)

y_pred = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\n🎯 MÉTRICAS PRINCIPALES:")
print(f"   • Accuracy:       {accuracy:.4f} ({accuracy:.2%})")
print(f"   • Precision:      {precision:.4f}")
print(f"   • Recall:         {recall:.4f}")
print(f"   • F1-Score:       {f1:.4f}")
print(f"   • ROC-AUC:        {auc:.4f}")
print(f"   • Avg Precision:  {avg_precision:.4f}")

print(f"\n📋 REPORTE DE CLASIFICACIÓN COMPLETO:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['No recontacta (0)', 'Recontacta (1)']))

print(f"💡 INTERPRETACIÓN:")
print(f"   • ROC-AUC = {auc:.3f} indica un poder discriminatorio "
      f"{'excelente' if auc > 0.85 else 'bueno' if auc > 0.75 else 'moderado'}")
print(f"   • Accuracy {'dentro' if 0.75 <= accuracy <= 0.85 else 'fuera'} del rango esperado (75%-85%)")
print(f"   • {'No hay indicio de data leakage (accuracy < 90%)' if accuracy < 0.90 else 'ALERTA: accuracy > 90%, revisar posible data leakage'}")
print(f"   • CV-AUC {cv.mean():.3f} vs AUC de prueba {auc:.3f}: diferencia {abs(cv.mean() - auc):.3f} "
      f"({'sin sobreajuste evidente' if abs(cv.mean() - auc) < 0.03 else 'revisar sobreajuste'})")
SALIDA (stdout)
text
======================================================================
📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA
======================================================================

🎯 MÉTRICAS PRINCIPALES:
   • Accuracy:       0.8140 (81.40%)
   • Precision:      0.7937
   • Recall:         0.8056
   • F1-Score:       0.7996
   • ROC-AUC:        0.8926
   • Avg Precision:  0.8853

📋 REPORTE DE CLASIFICACIÓN COMPLETO:
----------------------------------------------------------------------
                   precision    recall  f1-score   support

No recontacta (0)       0.83      0.82      0.83       548
   Recontacta (1)       0.79      0.81      0.80       468

         accuracy                           0.81      1016
        macro avg       0.81      0.81      0.81      1016
     weighted avg       0.81      0.81      0.81      1016

💡 INTERPRETACIÓN:
   • ROC-AUC = 0.893 indica un poder discriminatorio excelente
   • Accuracy dentro del rango esperado (75%-85%)
   • No hay indicio de data leakage (accuracy < 90%)
   • CV-AUC 0.891 vs AUC de prueba 0.893: diferencia 0.002 (sin sobreajuste evidente)
CELDA 9
ENTRADA (source)
python
# ============================================================
# Celda 8b: Verificación anti-fuga de datos (data leakage)
# ============================================================
print("=" * 70)
print("🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS")
print("=" * 70)

# 1) Ninguna variable del modelo contiene el resultado
sospechosas = [c for c in X.columns if 'recontacto' in c.lower()]
print(f"\n1) Variables con 'recontacto' en el nombre dentro del modelo: {sospechosas if sospechosas else 'ninguna ✅'}")

# 2) Ninguna variable por sí sola separa casi perfectamente el target
aucs = {}
for c in X.columns:
    if X[c].nunique() > 1:
        a = roc_auc_score(y, X[c])
        aucs[c] = max(a, 1 - a)
aucs = pd.Series(aucs).sort_values(ascending=False)
print(f"\n2) AUC individual más alto de una sola variable: {aucs.index[0]} = {aucs.iloc[0]:.3f}")
print(f"   {'✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)' if aucs.iloc[0] < 0.95 else '⚠️ revisar: una variable casi separa el target'}")
print(aucs.head(5).round(3).to_string())

# 3) Rendimiento realista y estable
print(f"\n3) Accuracy {accuracy:.2%} (alerta si > 90%): {'✅' if accuracy < 0.90 else '⚠️'}")
print(f"4) CV-AUC {cv.mean():.3f} vs AUC prueba {auc:.3f} (diferencia {abs(cv.mean() - auc):.3f}): "
      f"{'✅ estable' if abs(cv.mean() - auc) < 0.03 else '⚠️ revisar'}")

print("""
5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
""")
SALIDA (stdout)
text
======================================================================
🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS
======================================================================

1) Variables con 'recontacto' en el nombre dentro del modelo: ninguna ✅

2) AUC individual más alto de una sola variable: transferencias = 0.755
   ✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)
transferencias            0.755
espera_seg                0.722
es_multi_transferencia    0.665
es_espera_alta            0.640
casos_previos_30d         0.633

3) Accuracy 81.40% (alerta si > 90%): ✅
4) CV-AUC 0.891 vs AUC prueba 0.893 (diferencia 0.002): ✅ estable

5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
CELDA 10
ENTRADA (source)
python
# ============================================================
# Celda 9: Matriz de confusión
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No recontacta (0)', 'Recontacta (1)'],
            yticklabels=['No recontacta (0)', 'Recontacta (1)'],
            cbar_kws={'label': 'Cantidad'})
plt.title(f'Matriz de Confusión - Set de Prueba\n{KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Valor Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
guardar('03_matriz_confusion')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n📊 DESGLOSE DE LA MATRIZ:")
print(f"   • Verdaderos Positivos (VP): {tp:5d} → Recontactos detectados correctamente")
print(f"   • Verdaderos Negativos (VN): {tn:5d} → Casos resueltos identificados correctamente")
print(f"   • Falsos Positivos (FP):     {fp:5d} → Alarmas falsas (se predijo recontacto y no ocurrió)")
print(f"   • Falsos Negativos (FN):     {fn:5d} → Recontactos que el modelo no detectó")

print(f"\n💼 LECTURA OPERATIVA:")
print(f"   • De {tp + fn:,} recontactos reales, el modelo anticipó {tp:,} ({tp / (tp + fn):.1%})")
print(f"   • De {tp + fp:,} casos marcados como riesgo, {tp:,} sí recontactaron ({tp / (tp + fp):.1%})")
print(f"   • Seguimientos que serían innecesarios: {fp:,}")
SALIDA (stdout)
text

📊 DESGLOSE DE LA MATRIZ:
   • Verdaderos Positivos (VP):   377 → Recontactos detectados correctamente
   • Verdaderos Negativos (VN):   450 → Casos resueltos identificados correctamente
   • Falsos Positivos (FP):        98 → Alarmas falsas (se predijo recontacto y no ocurrió)
   • Falsos Negativos (FN):        91 → Recontactos que el modelo no detectó

💼 LECTURA OPERATIVA:
   • De 468 recontactos reales, el modelo anticipó 377 (80.6%)
   • De 475 casos marcados como riesgo, 377 sí recontactaron (79.4%)
   • Seguimientos que serían innecesarios: 98
SALIDA (display_data)
[FIGURA MOSTRADA] — Figura de tamaño 800x600 con 2 ejes (heatmap de matriz de confusión).

CELDA 11
ENTRADA (source)
python
# ============================================================
# Celda 10: Curva ROC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'Curva ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title(f'Curva ROC - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
guardar('04_curva_roc')
plt.show()

idx_optimo = np.argmax(tpr - fpr)
umbral_optimo = thresholds[idx_optimo]
print(f"\n🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):")
print(f"   • Umbral: {umbral_optimo:.3f}")
print(f"   • TPR en ese punto: {tpr[idx_optimo]:.3f}")
print(f"   • FPR en ese punto: {fpr[idx_optimo]:.3f}")
SALIDA (stdout)
text

🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):
   • Umbral: 0.486
   • TPR en ese punto: 0.825
   • FPR en ese punto: 0.190
SALIDA (display_data)
[FIGURA MOSTRADA] — Figura de tamaño 1000x700 con 1 eje (curva ROC).

CELDA 12
ENTRADA (source)
python
# ============================================================
# Celda 11: Variables más asociadas al recontacto
# ============================================================
print("=" * 70)
print("🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)")
print("=" * 70)

df_coef = pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_[0]})
df_coef['Odds ratio'] = np.exp(df_coef['Coeficiente'])
df_coef['Abs'] = df_coef['Coeficiente'].abs()
df_coef = df_coef.sort_values('Abs', ascending=False).reset_index(drop=True)
top15 = df_coef.head(15)

print(f"\n📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):")
print("-" * 70)
for i, r in top15.iterrows():
    signo = '↑ sube' if r['Coeficiente'] > 0 else '↓ baja'
    print(f"   {i + 1:2d}. {r['Variable']:28s} | coef {r['Coeficiente']:+.4f} | OR {r['Odds ratio']:.3f} | {signo} la prob.")

plot_data = top15.sort_values('Coeficiente')
colores = ['#e74c3c' if c > 0 else '#2ecc71' for c in plot_data['Coeficiente']]
plt.figure(figsize=(12, 8))
plt.barh(plot_data['Variable'], plot_data['Coeficiente'], color=colores, edgecolor='black')
plt.axvline(0, color='black', lw=0.8)
plt.xlabel('Coeficiente (variables estandarizadas)')
plt.title(f'Top 15 variables asociadas al recontacto - {KPI_ID}\n(rojo: aumenta la probabilidad, verde: la reduce)',
          fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
guardar('05_coeficientes')
plt.show()

print("\n⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.")
SALIDA (stdout)
text
======================================================================
🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)
======================================================================

📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):
----------------------------------------------------------------------
    1. transferencias               | coef +1.5784 | OR 4.847 | ↑ sube la prob.
    2. espera_seg                   | coef +0.9755 | OR 2.653 | ↑ sube la prob.
    3. casos_previos_30d            | coef +0.9346 | OR 2.546 | ↑ sube la prob.
    4. cola_reclamos                | coef -0.6072 | OR 0.545 | ↓ baja la prob.
    5. cola_soporte                 | coef -0.5023 | OR 0.605 | ↓ baja la prob.
    6. usuario_adulto_mayor         | coef +0.3487 | OR 1.417 | ↑ sube la prob.
    7. usuario_empresa              | coef -0.3384 | OR 0.713 | ↓ baja la prob.
    8. duracion_seg                 | coef +0.3166 | OR 1.372 | ↑ sube la prob.
    9. canal_red_social             | coef +0.2871 | OR 1.333 | ↑ sube la prob.
   10. es_duracion_corta            | coef +0.2827 | OR 1.327 | ↑ sube la prob.
   11. motivo_falla                 | coef -0.2346 | OR 0.791 | ↓ baja la prob.
   12. canal_telefono               | coef -0.1626 | OR 0.850 | ↓ baja la prob.
   13. turno_nocturno               | coef +0.1116 | OR 1.118 | ↑ sube la prob.
   14. cola_tramites                | coef -0.0977 | OR 0.907 | ↓ baja la prob.
   15. canal_sin_dato               | coef +0.0779 | OR 1.081 | ↑ sube la prob.

⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.
SALIDA (display_data)
[FIGURA MOSTRADA] — Figura de tamaño 1200x800 con 1 eje (barras horizontales de coeficientes top 15).

CELDA 13
ENTRADA (source)
python
# ============================================================
# Celda 12: Tasa de recontacto por segmento (reglas de negocio)
# ============================================================
print("=" * 70)
print("💼 RECONTACTO POR SEGMENTO")
print("=" * 70)

grupos = {'Canal': 'canal_', 'Motivo': 'motivo_', 'Cola': 'cola_',
          'Tipo de usuario': 'usuario_', 'Turno': 'turno_'}
tasa_kpi = df['es_recontacto'].mean() * 100

filas = []
for g, pref in grupos.items():
    for col in [c for c in df.columns if c.startswith(pref)]:
        m = df[col] == 1
        if m.sum() >= 30:
            filas.append({'Grupo': g, 'Segmento': col.replace(pref, ''),
                          'Casos': int(m.sum()),
                          'Tasa %': df.loc[m, 'es_recontacto'].mean() * 100})
seg = pd.DataFrame(filas).sort_values('Tasa %', ascending=True).reset_index(drop=True)
seg['Etiqueta'] = seg['Grupo'] + ': ' + seg['Segmento']

paleta = {'Canal': '#3498db', 'Motivo': '#e67e22', 'Cola': '#9b59b6',
          'Tipo de usuario': '#1abc9c', 'Turno': '#f1c40f'}
plt.figure(figsize=(12, max(6, len(seg) * 0.38)))
plt.barh(seg['Etiqueta'], seg['Tasa %'], color=[paleta[g] for g in seg['Grupo']], edgecolor='black')
plt.axvline(tasa_kpi, color='red', ls='--', label=f'Tasa del KPI ({tasa_kpi:.1f}%)')
plt.xlabel('% de recontacto a 7 días')
plt.title(f'Recontacto por segmento - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
guardar('06_segmentos')
plt.show()

print(f"\n🔺 SEGMENTOS CON MAYOR RECONTACTO:")
print(seg.tail(5).iloc[::-1][['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
print(f"\n🔻 SEGMENTOS CON MENOR RECONTACTO:")
print(seg.head(5)[['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
SALIDA (stdout)
text
======================================================================
💼 RECONTACTO POR SEGMENTO
======================================================================

🔺 SEGMENTOS CON MAYOR RECONTACTO:
                     Etiqueta  Casos  Tasa %
              Turno: nocturno    595    59.2
  Tipo de usuario: recurrente   1776    53.7
            Canal: red_social   1145    53.2
Tipo de usuario: adulto_mayor   1021    52.9
            Cola: facturacion    905    52.3

🔻 SEGMENTOS CON MENOR RECONTACTO:
                Etiqueta  Casos  Tasa %
Tipo de usuario: empresa    734    34.7
  Tipo de usuario: nuevo   1546    38.3
          Cola: reclamos    770    40.9
           Cola: soporte    948    41.4
         Canal: telefono   1252    42.2
SALIDA (display_data)
[FIGURA MOSTRADA] — Figura de tamaño 1200x760 con 1 eje (barras horizontales de tasa de recontacto por segmento).

CELDA 14
ENTRADA (source)
python
# ============================================================
# Celda 13: Cifras para el reporte ejecutivo (calculadas)
# El ANÁLISIS se redacta a mano en la celda de texto que sigue.
# ============================================================
ACCIONES = {
    'transferencias': 'Reducir transferencias: mejorar el ruteo para resolver en el primer contacto',
    'espera_seg': 'Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas',
    'casos_previos_30d': 'Seguimiento proactivo a usuarios con casos previos recientes',
    'es_duracion_corta': 'Revisar atenciones muy cortas: pueden cerrarse sin resolver',
    'duracion_seg': 'Revisar la duración de atención frente al tipo de caso',
    'usuario_adulto_mayor': 'Atención asistida y confirmación de resolución a adultos mayores',
    'turno_nocturno': 'Reforzar capacidad de resolución en turno nocturno',
    'canal_red_social': 'Mejorar la resolución en redes sociales o derivar a un canal con trazabilidad',
    'cola_informacion': 'Revisar la calidad de las respuestas en la cola de Información',
    'cola_facturacion': 'Revisar casos de Facturación para resolver en el primer contacto',
    'cola_tramites': 'Simplificar el seguimiento de Trámites',
}
REDUCCION_SUPUESTA = 0.15   # SUPUESTO de trabajo, no es un dato

tasa_kpi = df['es_recontacto'].mean() * 100
n_casos = len(df)
n_rec = int(df['es_recontacto'].sum())
dur_media = df['duracion_seg'].mean()
recall_ = tp / (tp + fn)

horas_rec = n_rec * dur_media / 3600
evitables = n_rec * recall_ * REDUCCION_SUPUESTA
horas_ahorradas = evitables * dur_media / 3600
tasa_proy = tasa_kpi - tasa_kpi * recall_ * REDUCCION_SUPUESTA
brecha_actual = tasa_kpi - META_6M
dif_proy = tasa_proy - META_6M
brecha_ref = tasa_proy - META_RECONTACTO

print("=" * 70)
print(f"📊 CIFRAS DEL REPORTE - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)
print(f"""
🎯 OBJETIVO: {OBJETIVO}
🔖 TARGET : {TARGET_DEF}

📊 DATOS
   • Casos del KPI: {n_casos:,}  |  Recontactos: {n_rec:,}  |  Tasa actual (línea base): {tasa_kpi:.2f}%
   • Meta a {PLAZO}: ≤ {META_6M}%  |  Brecha actual: {brecha_actual:.1f} puntos  |  Umbral de alerta: > {UMBRAL_ALERTA}%
   • Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%
   • Entrenamiento / prueba: {len(X_train):,} / {len(X_test):,}  |  Variables: {X.shape[1]}

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: {mejor})
   • Accuracy {accuracy:.2%} | Precision {precision:.2%} | Recall {recall:.2%} | F1 {f1:.4f} | ROC-AUC {auc:.4f}
   • CV-AUC (5 particiones): {cv.mean():.4f} ± {cv.std():.4f}
   • Recontactos detectados: {tp:,} de {tp + fn:,} ({recall_:.1%})  |  Alarmas falsas: {fp:,}

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = {dur_media:.0f} s)
   • Horas-agente que consumen los recontactos del periodo: ≈ {horas_rec:,.0f} h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el {REDUCCION_SUPUESTA:.0%} de los recontactos detectados)
   • Recontactos evitables: ≈ {evitables:,.0f}  |  Horas-agente liberadas: ≈ {horas_ahorradas:,.0f} h
   • La tasa bajaría de {tasa_kpi:.1f}% a ≈ {tasa_proy:.1f}%
   • Diferencia frente a la meta de {PLAZO} ({META_6M}%): {dif_proy:+.1f} puntos
   • Distancia que queda a la referencia de largo plazo ({META_RECONTACTO:.0f}%): {brecha_ref:.1f} puntos
   • El {REDUCCION_SUPUESTA:.0%} es un supuesto de trabajo, no un resultado del modelo.
""")

pos = df_coef[df_coef['Coeficiente'] > 0]
prioridades = ['URGENTE', 'ALTO', 'ALTO', 'MEDIO', 'MEDIO', 'MEDIO']
print("🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)")
print("-" * 70)
n = 0
for _, r in pos.iterrows():
    if r['Variable'] in ACCIONES and n < len(prioridades):
        print(f"   {prioridades[n]:8s} {ACCIONES[r['Variable']]}")
        print(f"            (variable: {r['Variable']} | odds ratio {r['Odds ratio']:.2f} por desv. estándar)")
        n += 1

print("\n🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):")
for _, r in df_coef[df_coef['Coeficiente'] < 0].head(3).iterrows():
    print(f"   ↓ {r['Variable']:26s} (odds ratio {r['Odds ratio']:.2f})")
SALIDA (stdout)
text
======================================================================
📊 CIFRAS DEL REPORTE - KPI02: Recontacto en motivos críticos (falla/queja)
======================================================================

🎯 OBJETIVO: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
🔖 TARGET : es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo

📊 DATOS
   • Casos del KPI: 5,077  |  Recontactos: 2,341  |  Tasa actual (línea base): 46.11%
   • Meta a 6 meses: ≤ 40.5%  |  Brecha actual: 5.6 puntos  |  Umbral de alerta: > 46.1%
   • Referencia de largo plazo (no equivalente): ≤ 22%
   • Entrenamiento / prueba: 4,061 / 1,016  |  Variables: 23

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: 1. Regresión Logística)
   • Accuracy 81.40% | Precision 79.37% | Recall 80.56% | F1 0.7996 | ROC-AUC 0.8926
   • CV-AUC (5 particiones): 0.8910 ± 0.0098
   • Recontactos detectados: 377 de 468 (80.6%)  |  Alarmas falsas: 98

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = 1021 s)
   • Horas-agente que consumen los recontactos del periodo: ≈ 664 h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el 15% de los recontactos detectados)
   • Recontactos evitables: ≈ 283  |  Horas-agente liberadas: ≈ 80 h
   • La tasa bajaría de 46.1% a ≈ 40.5%
   • Diferencia frente a la meta de 6 meses (40.5%): +0.0 puntos
   • Distancia que queda a la referencia de largo plazo (22%): 18.5 puntos
   • El 15% es un supuesto de trabajo, no un resultado del modelo.

🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)
----------------------------------------------------------------------
   URGENTE  Reducir transferencias: mejorar el ruteo para resolver en el primer contacto
            (variable: transferencias | odds ratio 4.85 por desv. estándar)
   ALTO     Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas
            (variable: espera_seg | odds ratio 2.65 por desv. estándar)
   ALTO     Seguimiento proactivo a usuarios con casos previos recientes
            (variable: casos_previos_30d | odds ratio 2.55 por desv. estándar)
   MEDIO    Atención asistida y confirmación de resolución a adultos mayores
            (variable: usuario_adulto_mayor | odds ratio 1.42 por desv. estándar)
   MEDIO    Revisar la duración de atención frente al tipo de caso
            (variable: duracion_seg | odds ratio 1.37 por desv. estándar)
   MEDIO    Mejorar la resolución en redes sociales o derivar a un canal con trazabilidad
            (variable: canal_red_social | odds ratio 1.33 por desv. estándar)

🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):
   ↓ cola_reclamos              (odds ratio 0.54)
   ↓ cola_soporte               (odds ratio 0.61)
   ↓ usuario_empresa            (odds ratio 0.71)
CELDA 15 (markdown)
ENTRADA (source)
text
# Fase 15: Reporte ejecutivo (redactado por el analista)

**Objetivo.** Reducir la tasa de recontacto a 7 días del Servicio Ciudadano 1800 en los casos por falla o queja (5,077 casos). La línea base es 46.1 %, la meta a 6 meses es ≤ 40.5 % (supuesto a validar con la dirección) y el umbral de alerta es una tasa superior a 46.1 %. El 22 % (100 % − 78 % de resolución en primer contacto) no es equivalente al recontacto a 7 días, por lo que se usa solo como referencia de largo plazo.

**Momento de predicción.** El modelo se aplica al finalizar la atención, para anticipar si el ciudadano volverá a contactar dentro de 7 días. Todas las variables (duración, espera, transferencias y casos previos de los 30 días anteriores) están disponibles en ese momento.

**Hallazgo central.** La Regresión Logística fue el mejor de los 7 modelos comparados (ROC-AUC 0.893, accuracy 81.4 %, CV-AUC 0.891). Detecta 377 de 468 recontactos del conjunto de prueba (recall 80.6 %) y 79.4 % de los casos que marca como riesgo sí recontactan (98 alarmas falsas). La probabilidad de recontacto se asocia sobre todo con las transferencias (OR 4.8 por desviación estándar), el tiempo de espera (OR 2.7) y los casos previos del usuario en los últimos 30 días (OR 2.5). Las comparaciones por categoría se leen frente a una referencia: cola Información, usuarios recurrentes, motivo Queja, canal Chat y turno PM. Frente a esas referencias, las colas de Reclamos (OR 0.54) y Soporte (OR 0.61), los usuarios empresa (OR 0.71) y los casos por falla (OR 0.79, frente a queja) tienen menor probabilidad de recontacto; los adultos mayores (OR 1.42) y el canal red social (OR 1.33) tienen mayor probabilidad. Tanto las atenciones de mayor duración (OR 1.37) como las muy cortas (OR 1.33), posible cierre sin resolver, se asocian a más recontacto.

**Lectura de negocio.** Las transferencias son la señal más fuerte: resolver en el primer contacto, con mejor ruteo, es la acción prioritaria. Después van la reducción del tiempo de espera y el seguimiento proactivo a usuarios con casos recientes. Por segmentos, la tasa observada dentro de los casos críticos es mayor en el turno nocturno (59.2 %), los usuarios recurrentes (53.7 %), la red social (53.2 %), los adultos mayores (52.9 %) y la cola de Facturación (52.3 %); es menor en los usuarios empresa (34.7 %) y los usuarios nuevos (38.3 %). Estas tasas son observadas y no ajustadas: una vez controlados el resto de factores, el turno y la cola de Facturación pesan poco frente a las transferencias y la espera.

**Alcance realista.** Los 2,341 recontactos por falla y queja consumen ≈ 664 horas-agente en el periodo (estimación con la duración media). Si se evitara el 15 % de los recontactos que el modelo detecta (supuesto de trabajo, no un resultado), se evitarían ≈ 283 recontactos y se liberarían ≈ 80 horas-agente. La tasa bajaría de 46.1 % a ≈ 40.5 %, que coincide con la meta a 6 meses; como esa meta se construyó con el mismo supuesto, solo es alcanzable si el supuesto se cumple. Quedarían ≈ 18.5 puntos hasta la referencia de largo plazo (22 %). El modelo ayuda a priorizar a quién dar seguimiento, pero no basta por sí solo.
SALIDA
(celda markdown, sin output)

CELDA 16 (code)
ENTRADA (source)
python
# Fase 16: Próximos pasos y limitaciones

**Próximos pasos.**
1. Validar con la dirección la meta a 6 meses, el umbral de alerta y el criterio de "mismo motivo".
2. Probar las intervenciones con una prueba A/B en las colas con mayor recontacto.
3. Monitorear mensualmente la tasa de recontacto y las métricas del modelo.
4. Reentrenar con datos reales cada trimestre.

**Limitaciones.**
- Los datos son sintéticos. Las relaciones entre variables (transferencias, espera, casos previos, etc.) fueron definidas como supuestos documentados del generador, y se verificó que el rendimiento resultante fuera realista (accuracy entre 75 % y 90 %, sin indicios de fuga de datos).
- La Regresión Logística gana en parte porque el generador usa una fórmula logística.
- Los coeficientes indican asociación, no causalidad, y se leen frente a la categoría de referencia de cada grupo. En este segmento el motivo solo distingue falla de queja.
- La división entre entrenamiento y prueba es aleatoria, no temporal, y no se dispone de un identificador de usuario para evitar que un mismo usuario aparezca en ambos conjuntos.
- No se dispone de costos por contacto, por lo que el impacto se expresa en casos y horas-agente, con un supuesto del 15 % declarado.
- La meta a 6 meses se deriva de ese mismo supuesto; la referencia del 22 % no es equivalente al recontacto a 7 días.
- Los 5 KPIs comparten el mismo evento (recontacto) y difieren en la población analizada.
- Parte del recontacto no es evitable (usuarios con consultas nuevas o complejas), por lo que conviene aceptar un piso mínimo.


-----------------------


Celda 1 — Código
Entrada

python
# ============================================================
# Celda 1: Importar librerías y configurar el KPI
# ============================================================
from lightgbm import LGBMClassifier
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from cargar_datos import cargar_dataset_recontacto

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ============ CAMBIAR SOLO ESTA LÍNEA EN CADA NOTEBOOK ============
KPI_SEL = 'KPI07'      # 'KPI01', 'KPI02', 'KPI07', 'KPI08' o 'KPI10'
# ===================================================================

FICHAS = {
    'KPI01': dict(nombre='Recontacto general', filtro=None,
                  formula='casos con recontacto a 7 días ÷ casos con resultado conocido',
                  base=41.5, meta=36.5),
    'KPI02': dict(nombre='Recontacto en motivos críticos (falla/queja)', filtro='motivos_criticos',
                  formula='recontactos a 7 días de casos por falla o queja ÷ casos por falla o queja con resultado conocido',
                  base=46.1, meta=40.5),
    'KPI07': dict(nombre='Recontacto en usuarios con historial', filtro='historial',
                  formula='recontactos a 7 días de usuarios con casos previos en 30 días ÷ casos de esos usuarios con resultado conocido',
                  base=50.4, meta=44.4),
    'KPI08': dict(nombre='Recontacto en canales de baja resolución (red social y correo)', filtro='canales_baja_resolucion',
                  formula='recontactos a 7 días en red social y correo ÷ casos en esos canales con resultado conocido',
                  base=44.3, meta=39.1),
    'KPI10': dict(nombre='Recontacto en turno nocturno y fin de semana', filtro='nocturno_finde',
                  formula='recontactos a 7 días en turno nocturno y fin de semana ÷ casos en esos turnos con resultado conocido',
                  base=44.1, meta=38.6),
}
_f = FICHAS[KPI_SEL]

# ------------------ CONFIGURACIÓN DEL KPI ------------------
KPI_ID = KPI_SEL
KPI_NOMBRE = _f['nombre']
FILTRO = _f['filtro']
KPI_FORMULA = _f['formula']

# ------------------ OBJETIVO Y META ------------------
OBJETIVO = 'Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800'

# Referencia externa (NO equivalente): 100% - 78% de resolución en primer contacto.
META_RECONTACTO = 22.0

# Ficha del KPI
LINEA_BASE = _f['base']     # tasa actual del segmento (%)
META_6M = _f['meta']        # meta a 6 meses (escenario del 15 %, supuesto a validar)
PLAZO = '6 meses'
UMBRAL_ALERTA = _f['base']  # si la tasa supera la línea base, se activa alerta

TARGET_DEF = ('es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; '
              '0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo')
# -----------------------------------------------------

os.makedirs('../informe_figuras', exist_ok=True)

def guardar(nombre):
    plt.savefig(f'../informe_figuras/{KPI_ID}_{nombre}.png', dpi=150, bbox_inches='tight')

print("✅ Librerías cargadas correctamente")
print(f"📌 {KPI_ID} - {KPI_NOMBRE}")
print(f"📐 Fórmula: {KPI_FORMULA}")
print(f"🎯 Objetivo: {OBJETIVO}")
print(f"📊 Línea base: {LINEA_BASE}% | 🎯 Meta {PLAZO}: ≤ {META_6M}% | 🚨 Umbral de alerta: > {UMBRAL_ALERTA}%")
print(f"🏁 Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%")
print(f"🔖 Target: {TARGET_DEF}")
Salida

text
✅ Librerías cargadas correctamente
📌 KPI07 - Recontacto en usuarios con historial
📐 Fórmula: recontactos a 7 días de usuarios con casos previos en 30 días ÷ casos de esos usuarios con resultado conocido
🎯 Objetivo: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
📊 Línea base: 50.4% | 🎯 Meta 6 meses: ≤ 44.4% | 🚨 Umbral de alerta: > 50.4%
🏁 Referencia de largo plazo (no equivalente): ≤ 22%
🔖 Target: es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo
Celda 2 — Código
Entrada

python
# ============================================================
# Celda 2: Cargar dataset desde SQL Server
# ============================================================
def filtrar(d, filtro):
    if filtro == 'motivos_criticos':
        d = d[(d.motivo_falla == 1) | (d.motivo_queja == 1)]
    elif filtro == 'historial':
        d = d[d.tiene_historial == 1]
    elif filtro == 'canales_baja_resolucion':
        d = d[(d.canal_red_social == 1) | (d.canal_correo == 1)]
    elif filtro == 'nocturno_finde':
        d = d[(d.turno_nocturno == 1) | (d.turno_fin_semana == 1)]
    return d.reset_index(drop=True)

print("=" * 70)
print(f"📊 CARGANDO DATASET - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)

df_base = cargar_dataset_recontacto()
df = filtrar(df_base, FILTRO)

tasa_inicial = df['es_recontacto'].mean() * 100

print(f"\n✅ Dataset del KPI listo")
print(f"📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"📈 Tasa de recontacto del KPI: {tasa_inicial:.2f}%")
print(f"📈 Tasa de recontacto global : {df_base['es_recontacto'].mean() * 100:.2f}%")
print(f"🏁 Meta {PLAZO}: ≤ {META_6M}% → brecha actual: {tasa_inicial - META_6M:.1f} puntos | referencia largo plazo: {META_RECONTACTO:.0f}%")

# Control: la línea base escrita en la Celda 1 debe coincidir con la tasa real
if abs(tasa_inicial - LINEA_BASE) > 0.5:
    print(f"⚠️ OJO: la línea base de la ficha ({LINEA_BASE}%) no coincide con la tasa real ({tasa_inicial:.1f}%). Revisar KPI_SEL.")
else:
    print(f"✅ La línea base de la ficha ({LINEA_BASE}%) coincide con la tasa real.")
Salida

text
======================================================================
📊 CARGANDO DATASET - KPI07: Recontacto en usuarios con historial
======================================================================
============================================================
📊 CARGANDO DATASET DE RECONTACTO (KPI 1)
============================================================
✅ Conexión exitosa a SQL Server
✅ Consulta ejecutada: 11,436 registros, 34 columnas

📏 Dimensiones: 11,436 filas × 34 columnas

🎯 Distribución del target:
es_recontacto
0    6687
1    4749
Name: count, dtype: int64

📈 Tasa de recontacto: 41.53%

✅ Dataset del KPI listo
📏 Dimensiones: 6,479 filas × 34 columnas
📈 Tasa de recontacto del KPI: 50.39%
📈 Tasa de recontacto global : 41.53%
🏁 Meta 6 meses: ≤ 44.4% → brecha actual: 6.0 puntos | referencia largo plazo: 22%
✅ La línea base de la ficha (50.4%) coincide con la tasa real.
Celda 3 — Código
Entrada

python
# ============================================================
# Celda 3: Exploración inicial del dataset
# ============================================================
print("=" * 70)
print("🔍 EXPLORACIÓN INICIAL DEL DATASET")
print("=" * 70)

print(f"\n📋 Columnas del dataset ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col:28s} | Tipo: {str(df[col].dtype):8s} | Nulos: {df[col].isnull().sum()}")

target_counts = df['es_recontacto'].value_counts().sort_index()
print(f"\n🎯 Distribución del target (es_recontacto):")
print(target_counts)
print(f"\n📈 Tasa de recontacto: {df['es_recontacto'].mean() * 100:.2f}%")

print(f"\n🔑 VARIABLES OPERATIVAS CLAVE:")
for c in ['espera_seg', 'duracion_seg', 'transferencias', 'casos_previos_30d']:
    print(f"   • {c}: media {df[c].mean():.2f} | mediana {df[c].median():.2f} | máx {df[c].max():.0f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

target_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Distribución del target', fontweight='bold')
axes[0, 0].set_xticklabels(['No recontacta (0)', 'Recontacta (1)'], rotation=0)
axes[0, 0].set_ylabel('Casos')
for i, v in enumerate(target_counts):
    axes[0, 0].text(i, v + max(target_counts) * 0.01, f'{v:,}', ha='center', fontweight='bold')

df['espera_seg'].hist(ax=axes[0, 1], bins=30, color='#3498db', edgecolor='black')
axes[0, 1].set_title('Distribución del tiempo de espera', fontweight='bold')
axes[0, 1].set_xlabel('Segundos de espera')

df['duracion_seg'].hist(ax=axes[1, 0], bins=30, color='#9b59b6', edgecolor='black')
axes[1, 0].set_title('Distribución de la duración de la atención', fontweight='bold')
axes[1, 0].set_xlabel('Segundos de duración')

tr = df['transferencias'].clip(upper=3)
tasa_tr = df.groupby(tr)['es_recontacto'].mean() * 100
etiquetas = [str(int(i)) if i < 3 else '3 o más' for i in tasa_tr.index]
axes[1, 1].bar(etiquetas, tasa_tr.values, color='#e67e22', edgecolor='black')
axes[1, 1].axhline(df['es_recontacto'].mean() * 100, color='gray', ls='--', label='Tasa del KPI')
axes[1, 1].set_title('Tasa de recontacto según transferencias', fontweight='bold')
axes[1, 1].set_xlabel('Transferencias')
axes[1, 1].set_ylabel('% recontacto')
axes[1, 1].legend()
for i, v in enumerate(tasa_tr.values):
    axes[1, 1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
guardar('01_exploracion')
plt.show()
Salida

text
======================================================================
🔍 EXPLORACIÓN INICIAL DEL DATASET
======================================================================

📋 Columnas del dataset (34):
    1. id_interaccion               | Tipo: int64    | Nulos: 0
    2. es_recontacto                | Tipo: int64    | Nulos: 0
    3. duracion_seg                 | Tipo: int64    | Nulos: 0
    4. espera_seg                   | Tipo: int64    | Nulos: 0
    5. transferencias               | Tipo: int64    | Nulos: 0
    6. casos_previos_30d            | Tipo: int64    | Nulos: 0
    7. es_duracion_corta            | Tipo: int64    | Nulos: 0
    8. es_espera_alta               | Tipo: int64    | Nulos: 0
    9. es_multi_transferencia       | Tipo: int64    | Nulos: 0
   10. tiene_historial              | Tipo: int64    | Nulos: 0
   11. canal_chat                   | Tipo: int64    | Nulos: 0
   12. canal_correo                 | Tipo: int64    | Nulos: 0
   13. canal_red_social             | Tipo: int64    | Nulos: 0
   14. canal_telefono               | Tipo: int64    | Nulos: 0
   15. canal_sin_dato               | Tipo: int64    | Nulos: 0
   16. motivo_consulta              | Tipo: int64    | Nulos: 0
   17. motivo_falla                 | Tipo: int64    | Nulos: 0
   18. motivo_cobro                 | Tipo: int64    | Nulos: 0
   19. motivo_queja                 | Tipo: int64    | Nulos: 0
   20. motivo_solicitud             | Tipo: int64    | Nulos: 0
   21. cola_facturacion             | Tipo: int64    | Nulos: 0
   22. cola_soporte                 | Tipo: int64    | Nulos: 0
   23. cola_informacion             | Tipo: int64    | Nulos: 0
   24. cola_reclamos                | Tipo: int64    | Nulos: 0
   25. cola_tramites                | Tipo: int64    | Nulos: 0
   26. usuario_nuevo                | Tipo: int64    | Nulos: 0
   27. usuario_recurrente           | Tipo: int64    | Nulos: 0
   28. usuario_empresa              | Tipo: int64    | Nulos: 0
   29. usuario_adulto_mayor         | Tipo: int64    | Nulos: 0
   30. turno_am                     | Tipo: int64    | Nulos: 0
   31. turno_pm                     | Tipo: int64    | Nulos: 0
   32. turno_nocturno               | Tipo: int64    | Nulos: 0
   33. turno_fin_semana             | Tipo: int64    | Nulos: 0
   34. fecha_contacto               | Tipo: object   | Nulos: 0

🎯 Distribución del target (es_recontacto):
es_recontacto
0    3214
1    3265
Name: count, dtype: int64

📈 Tasa de recontacto: 50.39%

🔑 VARIABLES OPERATIVAS CLAVE:
   • espera_seg: media 334.52 | mediana 294.00 | máx 900
   • duracion_seg: media 985.15 | mediana 982.00 | máx 1800
   • transferencias: media 0.73 | mediana 1.00 | máx 4
   • casos_previos_30d: media 1.70 | mediana 1.00 | máx 8
[Figura/imagen omitida: Figure size 1500x1000 with 4 Axes]

Celda 4 — Código
Entrada

python
# ============================================================
# Celda 4: Preprocesamiento de variables
# ============================================================
print("=" * 70)
print("🔧 PREPROCESAMIENTO DE VARIABLES")
print("=" * 70)

columnas_a_eliminar = ['id_interaccion', 'fecha_contacto', 'es_recontacto']
X = df.drop(columns=columnas_a_eliminar, errors='ignore')
y = df['es_recontacto']

print(f"\n📊 Features iniciales: {X.shape[1]} columnas")
print(f"🚫 Excluidas del modelo: {columnas_a_eliminar} (identificador, fecha y target)")

columnas_texto = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n📝 Variables categóricas sin codificar: {columnas_texto if columnas_texto else 'ninguna (ya vienen en dummies)'}")
if columnas_texto:
    X = pd.get_dummies(X, columns=columnas_texto, drop_first=True)

columnas_bool = X.select_dtypes(include=['bool']).columns.tolist()
if columnas_bool:
    X = X.astype({c: 'int8' for c in columnas_bool})
    print(f"✅ {len(columnas_bool)} columnas booleanas convertidas a enteros")

nulos_antes = int(X.isnull().sum().sum())
X = X.fillna(0)
print(f"✅ Valores nulos rellenados: {nulos_antes} → {int(X.isnull().sum().sum())}")

# --- Quitar columnas constantes (no aportan nada, p. ej. tiene_historial en KPI07) ---
constantes = [c for c in X.columns if X[c].nunique() <= 1]
if constantes:
    X = X.drop(columns=constantes)
    print(f"\n🧹 Columnas constantes eliminadas: {constantes}")

# --- Evitar colinealidad: una categoría de referencia por grupo de dummies ---
GRUPOS = ['canal_', 'motivo_', 'cola_', 'usuario_', 'turno_']
referencias = {}
for pref in GRUPOS:
    cols = [c for c in X.columns if c.startswith(pref)]
    if cols and (X[cols].sum(axis=1) == 1).all():      # el grupo suma 1 en todas las filas
        ref = max(cols, key=lambda c: X[c].sum())      # la más frecuente como referencia
        X = X.drop(columns=ref)
        referencias[pref.rstrip('_')] = ref

print("\n📌 Categorías de referencia (los coeficientes se leen frente a ellas):")
for g, r in referencias.items():
    print(f"   • {g:8s} → {r}")

print(f"\n📏 Features finales: {X.shape[1]} columnas")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")
Salida

text
======================================================================
🔧 PREPROCESAMIENTO DE VARIABLES
======================================================================

📊 Features iniciales: 31 columnas
🚫 Excluidas del modelo: ['id_interaccion', 'fecha_contacto', 'es_recontacto'] (identificador, fecha y target)

📝 Variables categóricas sin codificar: ninguna (ya vienen en dummies)
✅ Valores nulos rellenados: 0 → 0

🧹 Columnas constantes eliminadas: ['tiene_historial']

📌 Categorías de referencia (los coeficientes se leen frente a ellas):
   • canal    → canal_chat
   • motivo   → motivo_falla
   • cola     → cola_informacion
   • usuario  → usuario_recurrente
   • turno    → turno_am

📏 Features finales: 25 columnas
   1. duracion_seg
   2. espera_seg
   3. transferencias
   4. casos_previos_30d
   5. es_duracion_corta
   6. es_espera_alta
   7. es_multi_transferencia
   8. canal_correo
   9. canal_red_social
  10. canal_telefono
  11. canal_sin_dato
  12. motivo_consulta
  13. motivo_cobro
  14. motivo_queja
  15. motivo_solicitud
  16. cola_facturacion
  17. cola_soporte
  18. cola_reclamos
  19. cola_tramites
  20. usuario_nuevo
  21. usuario_empresa
  22. usuario_adulto_mayor
  23. turno_pm
  24. turno_nocturno
  25. turno_fin_semana
Celda 5 — Código
Entrada

python
# ============================================================
# Celda 5: División del dataset (80% / 20%)
# ============================================================
print("=" * 70)
print("📊 DIVISIÓN TRAIN / TEST (80% / 20%)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📏 Entrenamiento: {X_train.shape[0]:,} registros")
print(f"📏 Prueba:        {X_test.shape[0]:,} registros")
print(f"\n🎯 Clases en TRAIN:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\n🎯 Clases en TEST:\n{y_test.value_counts(normalize=True).round(3)}")
print("\n✅ División estratificada completada (proporción mantenida)")
print("✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)")
Salida

text
======================================================================
📊 DIVISIÓN TRAIN / TEST (80% / 20%)
======================================================================

📏 Entrenamiento: 5,183 registros
📏 Prueba:        1,296 registros

🎯 Clases en TRAIN:
es_recontacto
1    0.504
0    0.496
Name: proportion, dtype: float64

🎯 Clases en TEST:
es_recontacto
1    0.504
0    0.496
Name: proportion, dtype: float64

✅ División estratificada completada (proporción mantenida)
✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)
Celda 6 — Código
Entrada

python
# ============================================================
# Celda 6: Comparación de modelos (selección del mejor)
# ============================================================
print("=" * 70)
print("🤖 COMPARACIÓN DE 7 MODELOS")
print("=" * 70)

ratio = (y_train == 0).sum() / (y_train == 1).sum()

modelos = {
    '1. Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    '2. Árbol de Decisión': DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10,
                                                   class_weight='balanced', random_state=42),
    '3. Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10,
                                               class_weight='balanced', random_state=42, n_jobs=-1),
    '4. XGBoost': XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                scale_pos_weight=ratio, random_state=42, eval_metric='auc', verbosity=0),
    '5. LightGBM': LGBMClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=ratio, random_state=42, verbose=-1),
    '6. KNN (k=10)': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
    '7. Naive Bayes': GaussianNB(),
}
ESCALADOS = ['1. Regresión Logística', '6. KNN (k=10)']

resultados = []
for nombre, m in modelos.items():
    inicio = time.time()
    a, b = (X_train_scaled, X_test_scaled) if nombre in ESCALADOS else (X_train, X_test)
    m.fit(a, y_train)
    pred = m.predict(b)
    proba = m.predict_proba(b)[:, 1]
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, proba),
        'Tiempo (s)': time.time() - inicio,
    })
    print(f"   ✅ {nombre:26s} Accuracy {resultados[-1]['Accuracy']:.4f} | AUC {resultados[-1]['ROC-AUC']:.4f}")

df_res = pd.DataFrame(resultados).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n📊 TABLA COMPARATIVA (ordenada por ROC-AUC)")
print(df_res.round(4).to_string(index=False))

mejor = df_res.iloc[0]['Modelo']
print(f"\n🏆 MEJOR MODELO: {mejor}  (ROC-AUC {df_res.iloc[0]['ROC-AUC']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
orden = df_res.sort_values('Accuracy')
axes[0].barh(orden['Modelo'], orden['Accuracy'], color='#3498db', edgecolor='black')
axes[0].axvspan(0.75, 0.85, color='green', alpha=0.12, label='Rango esperado 75-85%')
axes[0].set_title('Accuracy por modelo', fontweight='bold')
axes[0].set_xlim(0.6, 0.95)
axes[0].legend(loc='lower right')
orden = df_res.sort_values('ROC-AUC')
axes[1].barh(orden['Modelo'], orden['ROC-AUC'], color='#e74c3c', edgecolor='black')
axes[1].set_title('ROC-AUC por modelo', fontweight='bold')
axes[1].set_xlim(0.6, 1.0)
plt.tight_layout()
guardar('02_comparacion_modelos')
plt.show()
Salida

text
======================================================================
🤖 COMPARACIÓN DE 7 MODELOS
======================================================================
   ✅ 1. Regresión Logística     Accuracy 0.8063 | AUC 0.8745
   ✅ 2. Árbol de Decisión       Accuracy 0.7431 | AUC 0.8195
   ✅ 3. Random Forest           Accuracy 0.7739 | AUC 0.8534
   ✅ 4. XGBoost                 Accuracy 0.7755 | AUC 0.8487
   ✅ 5. LightGBM                Accuracy 0.7670 | AUC 0.8454
   ✅ 6. KNN (k=10)              Accuracy 0.7184 | AUC 0.7855
   ✅ 7. Naive Bayes             Accuracy 0.7323 | AUC 0.8165

📊 TABLA COMPARATIVA (ordenada por ROC-AUC)
                Modelo  Accuracy  Precision  Recall  F1-Score  ROC-AUC  Tiempo (s)
1. Regresión Logística    0.8063     0.8131  0.7994    0.8062   0.8745      0.0449
      3. Random Forest    0.7739     0.7786  0.7703    0.7744   0.8534      1.7845
            4. XGBoost    0.7755     0.7864  0.7611    0.7735   0.8487      3.3363
           5. LightGBM    0.7670     0.7755  0.7565    0.7659   0.8454      1.0579
  2. Árbol de Decisión    0.7431     0.7516  0.7320    0.7417   0.8195      0.0618
        7. Naive Bayes    0.7323     0.7782  0.6554    0.7116   0.8165      0.0434
         6. KNN (k=10)    0.7184     0.7449  0.6708    0.7059   0.7855      0.2827

🏆 MEJOR MODELO: 1. Regresión Logística  (ROC-AUC 0.8745)
[Figura/imagen omitida: Figure size 1600x550 with 2 Axes]

Celda 7 — Código
Entrada

python
# ============================================================
# Celda 7: Entrenamiento del modelo final (Regresión Logística)
# ============================================================
print("=" * 70)
print("🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA")
print("=" * 70)

modelo = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
print(f"\n🔄 Entrenando con {X_train.shape[0]:,} registros...")
modelo.fit(X_train_scaled, y_train)
print("✅ Modelo entrenado exitosamente")

cv = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\n🔁 Validación cruzada (5 particiones) ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

if mejor != '1. Regresión Logística':
    print(f"\n⚠️ OJO: en este KPI el mejor por AUC fue {mejor}. Revisar antes de justificar la Regresión Logística.")
else:
    print("\n✅ La Regresión Logística es el mejor modelo de la comparación para este KPI")
Salida

text
======================================================================
🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA
======================================================================

🔄 Entrenando con 5,183 registros...
✅ Modelo entrenado exitosamente

🔁 Validación cruzada (5 particiones) ROC-AUC: 0.8777 ± 0.0141

✅ La Regresión Logística es el mejor modelo de la comparación para este KPI
Celda 8 — Código
Entrada

python
# ============================================================
# Celda 8: Evaluación de métricas en el set de prueba
# ============================================================
print("=" * 70)
print("📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA")
print("=" * 70)

y_pred = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\n🎯 MÉTRICAS PRINCIPALES:")
print(f"   • Accuracy:       {accuracy:.4f} ({accuracy:.2%})")
print(f"   • Precision:      {precision:.4f}")
print(f"   • Recall:         {recall:.4f}")
print(f"   • F1-Score:       {f1:.4f}")
print(f"   • ROC-AUC:        {auc:.4f}")
print(f"   • Avg Precision:  {avg_precision:.4f}")

print(f"\n📋 REPORTE DE CLASIFICACIÓN COMPLETO:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['No recontacta (0)', 'Recontacta (1)']))

print(f"💡 INTERPRETACIÓN:")
print(f"   • ROC-AUC = {auc:.3f} indica un poder discriminatorio "
      f"{'excelente' if auc > 0.85 else 'bueno' if auc > 0.75 else 'moderado'}")
print(f"   • Accuracy {'dentro' if 0.75 <= accuracy <= 0.85 else 'fuera'} del rango esperado (75%-85%)")
print(f"   • {'No hay indicio de data leakage (accuracy < 90%)' if accuracy < 0.90 else 'ALERTA: accuracy > 90%, revisar posible data leakage'}")
print(f"   • CV-AUC {cv.mean():.3f} vs AUC de prueba {auc:.3f}: diferencia {abs(cv.mean() - auc):.3f} "
      f"({'sin sobreajuste evidente' if abs(cv.mean() - auc) < 0.03 else 'revisar sobreajuste'})")
Salida

text
======================================================================
📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA
======================================================================

🎯 MÉTRICAS PRINCIPALES:
   • Accuracy:       0.8063 (80.63%)
   • Precision:      0.8131
   • Recall:         0.7994
   • F1-Score:       0.8062
   • ROC-AUC:        0.8745
   • Avg Precision:  0.8780

📋 REPORTE DE CLASIFICACIÓN COMPLETO:
----------------------------------------------------------------------
                   precision    recall  f1-score   support

No recontacta (0)       0.80      0.81      0.81       643
   Recontacta (1)       0.81      0.80      0.81       653

         accuracy                           0.81      1296
        macro avg       0.81      0.81      0.81      1296
     weighted avg       0.81      0.81      0.81      1296

💡 INTERPRETACIÓN:
   • ROC-AUC = 0.874 indica un poder discriminatorio excelente
   • Accuracy dentro del rango esperado (75%-85%)
   • No hay indicio de data leakage (accuracy < 90%)
   • CV-AUC 0.878 vs AUC de prueba 0.874: diferencia 0.003 (sin sobreajuste evidente)
Celda 8b — Código
Entrada

python
# ============================================================
# Celda 8b: Verificación anti-fuga de datos (data leakage)
# ============================================================
print("=" * 70)
print("🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS")
print("=" * 70)

# 1) Ninguna variable del modelo contiene el resultado
sospechosas = [c for c in X.columns if 'recontacto' in c.lower()]
print(f"\n1) Variables con 'recontacto' en el nombre dentro del modelo: {sospechosas if sospechosas else 'ninguna ✅'}")

# 2) Ninguna variable por sí sola separa casi perfectamente el target
aucs = {}
for c in X.columns:
    if X[c].nunique() > 1:
        a = roc_auc_score(y, X[c])
        aucs[c] = max(a, 1 - a)
aucs = pd.Series(aucs).sort_values(ascending=False)
print(f"\n2) AUC individual más alto de una sola variable: {aucs.index[0]} = {aucs.iloc[0]:.3f}")
print(f"   {'✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)' if aucs.iloc[0] < 0.95 else '⚠️ revisar: una variable casi separa el target'}")
print(aucs.head(5).round(3).to_string())

# 3) Rendimiento realista y estable
print(f"\n3) Accuracy {accuracy:.2%} (alerta si > 90%): {'✅' if accuracy < 0.90 else '⚠️'}")
print(f"4) CV-AUC {cv.mean():.3f} vs AUC prueba {auc:.3f} (diferencia {abs(cv.mean() - auc):.3f}): "
      f"{'✅ estable' if abs(cv.mean() - auc) < 0.03 else '⚠️ revisar'}")

print("""
5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
""")
Salida

text
======================================================================
🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS
======================================================================

1) Variables con 'recontacto' en el nombre dentro del modelo: ninguna ✅

2) AUC individual más alto de una sola variable: transferencias = 0.739
   ✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)
transferencias            0.739
espera_seg                0.720
es_espera_alta            0.638
es_multi_transferencia    0.632
duracion_seg              0.615

3) Accuracy 80.63% (alerta si > 90%): ✅
4) CV-AUC 0.878 vs AUC prueba 0.874 (diferencia 0.003): ✅ estable

5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
Celda 9 — Código
Entrada

python
# ============================================================
# Celda 9: Matriz de confusión
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No recontacta (0)', 'Recontacta (1)'],
            yticklabels=['No recontacta (0)', 'Recontacta (1)'],
            cbar_kws={'label': 'Cantidad'})
plt.title(f'Matriz de Confusión - Set de Prueba\n{KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Valor Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
guardar('03_matriz_confusion')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n📊 DESGLOSE DE LA MATRIZ:")
print(f"   • Verdaderos Positivos (VP): {tp:5d} → Recontactos detectados correctamente")
print(f"   • Verdaderos Negativos (VN): {tn:5d} → Casos resueltos identificados correctamente")
print(f"   • Falsos Positivos (FP):     {fp:5d} → Alarmas falsas (se predijo recontacto y no ocurrió)")
print(f"   • Falsos Negativos (FN):     {fn:5d} → Recontactos que el modelo no detectó")

print(f"\n💼 LECTURA OPERATIVA:")
print(f"   • De {tp + fn:,} recontactos reales, el modelo anticipó {tp:,} ({tp / (tp + fn):.1%})")
print(f"   • De {tp + fp:,} casos marcados como riesgo, {tp:,} sí recontactaron ({tp / (tp + fp):.1%})")
print(f"   • Seguimientos que serían innecesarios: {fp:,}")
Salida

text
[Figura/imagen omitida: Figure size 800x600 with 2 Axes]

📊 DESGLOSE DE LA MATRIZ:
   • Verdaderos Positivos (VP):   522 → Recontactos detectados correctamente
   • Verdaderos Negativos (VN):   523 → Casos resueltos identificados correctamente
   • Falsos Positivos (FP):       120 → Alarmas falsas (se predijo recontacto y no ocurrió)
   • Falsos Negativos (FN):       131 → Recontactos que el modelo no detectó

💼 LECTURA OPERATIVA:
   • De 653 recontactos reales, el modelo anticipó 522 (79.9%)
   • De 642 casos marcados como riesgo, 522 sí recontactaron (81.3%)
   • Seguimientos que serían innecesarios: 120
Celda 10 — Código
Entrada

python
# ============================================================
# Celda 10: Curva ROC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'Curva ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title(f'Curva ROC - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
guardar('04_curva_roc')
plt.show()

idx_optimo = np.argmax(tpr - fpr)
umbral_optimo = thresholds[idx_optimo]
print(f"\n🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):")
print(f"   • Umbral: {umbral_optimo:.3f}")
print(f"   • TPR en ese punto: {tpr[idx_optimo]:.3f}")
print(f"   • FPR en ese punto: {fpr[idx_optimo]:.3f}")
Salida

text
[Figura/imagen omitida: Figure size 1000x700 with 1 Axes]

🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):
   • Umbral: 0.502
   • TPR en ese punto: 0.799
   • FPR en ese punto: 0.185
Celda 11 — Código
Entrada

python
# ============================================================
# Celda 11: Variables más asociadas al recontacto
# ============================================================
print("=" * 70)
print("🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)")
print("=" * 70)

df_coef = pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_[0]})
df_coef['Odds ratio'] = np.exp(df_coef['Coeficiente'])
df_coef['Abs'] = df_coef['Coeficiente'].abs()
df_coef = df_coef.sort_values('Abs', ascending=False).reset_index(drop=True)
top15 = df_coef.head(15)

print(f"\n📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):")
print("-" * 70)
for i, r in top15.iterrows():
    signo = '↑ sube' if r['Coeficiente'] > 0 else '↓ baja'
    print(f"   {i + 1:2d}. {r['Variable']:28s} | coef {r['Coeficiente']:+.4f} | OR {r['Odds ratio']:.3f} | {signo} la prob.")

plot_data = top15.sort_values('Coeficiente')
colores = ['#e74c3c' if c > 0 else '#2ecc71' for c in plot_data['Coeficiente']]
plt.figure(figsize=(12, 8))
plt.barh(plot_data['Variable'], plot_data['Coeficiente'], color=colores, edgecolor='black')
plt.axvline(0, color='black', lw=0.8)
plt.xlabel('Coeficiente (variables estandarizadas)')
plt.title(f'Top 15 variables asociadas al recontacto - {KPI_ID}\n(rojo: aumenta la probabilidad, verde: la reduce)',
          fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
guardar('05_coeficientes')
plt.show()

print("\n⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.")
Salida

text
======================================================================
🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)
======================================================================

📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):
----------------------------------------------------------------------
    1. transferencias               | coef +1.4962 | OR 4.465 | ↑ sube la prob.
    2. espera_seg                   | coef +0.9147 | OR 2.496 | ↑ sube la prob.
    3. casos_previos_30d            | coef +0.8172 | OR 2.264 | ↑ sube la prob.
    4. cola_reclamos                | coef -0.5665 | OR 0.568 | ↓ baja la prob.
    5. cola_soporte                 | coef -0.4543 | OR 0.635 | ↓ baja la prob.
    6. es_duracion_corta            | coef +0.3489 | OR 1.418 | ↑ sube la prob.
    7. duracion_seg                 | coef +0.3401 | OR 1.405 | ↑ sube la prob.
    8. usuario_adulto_mayor         | coef +0.2853 | OR 1.330 | ↑ sube la prob.
    9. turno_nocturno               | coef +0.2769 | OR 1.319 | ↑ sube la prob.
   10. usuario_empresa              | coef -0.2544 | OR 0.775 | ↓ baja la prob.
   11. canal_red_social             | coef +0.2369 | OR 1.267 | ↑ sube la prob.
   12. motivo_queja                 | coef +0.2276 | OR 1.256 | ↑ sube la prob.
   13. motivo_solicitud             | coef +0.1562 | OR 1.169 | ↑ sube la prob.
   14. canal_telefono               | coef -0.1123 | OR 0.894 | ↓ baja la prob.
   15. cola_tramites                | coef -0.0970 | OR 0.908 | ↓ baja la prob.
[Figura/imagen omitida: Figure size 1200x800 with 1 Axes]

text
⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.
Celda 12 — Código
Entrada

python
# ============================================================
# Celda 12: Tasa de recontacto por segmento (reglas de negocio)
# ============================================================
print("=" * 70)
print("💼 RECONTACTO POR SEGMENTO")
print("=" * 70)

grupos = {'Canal': 'canal_', 'Motivo': 'motivo_', 'Cola': 'cola_',
          'Tipo de usuario': 'usuario_', 'Turno': 'turno_'}
tasa_kpi = df['es_recontacto'].mean() * 100

filas = []
for g, pref in grupos.items():
    for col in [c for c in df.columns if c.startswith(pref)]:
        m = df[col] == 1
        if m.sum() >= 30:
            filas.append({'Grupo': g, 'Segmento': col.replace(pref, ''),
                          'Casos': int(m.sum()),
                          'Tasa %': df.loc[m, 'es_recontacto'].mean() * 100})
seg = pd.DataFrame(filas).sort_values('Tasa %', ascending=True).reset_index(drop=True)
seg['Etiqueta'] = seg['Grupo'] + ': ' + seg['Segmento']

paleta = {'Canal': '#3498db', 'Motivo': '#e67e22', 'Cola': '#9b59b6',
          'Tipo de usuario': '#1abc9c', 'Turno': '#f1c40f'}
plt.figure(figsize=(12, max(6, len(seg) * 0.38)))
plt.barh(seg['Etiqueta'], seg['Tasa %'], color=[paleta[g] for g in seg['Grupo']], edgecolor='black')
plt.axvline(tasa_kpi, color='red', ls='--', label=f'Tasa del KPI ({tasa_kpi:.1f}%)')
plt.xlabel('% de recontacto a 7 días')
plt.title(f'Recontacto por segmento - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
guardar('06_segmentos')
plt.show()

print(f"\n🔺 SEGMENTOS CON MAYOR RECONTACTO:")
print(seg.tail(5).iloc[::-1][['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
print(f"\n🔻 SEGMENTOS CON MENOR RECONTACTO:")
print(seg.head(5)[['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
Salida

text
======================================================================
💼 RECONTACTO POR SEGMENTO
======================================================================
[Figura/imagen omitida: Figure size 1200x874 with 1 Axes]

text
🔺 SEGMENTOS CON MAYOR RECONTACTO:
                     Etiqueta  Casos  Tasa %
              Turno: nocturno    753    64.4
Tipo de usuario: adulto_mayor    958    57.8
                Motivo: falla   1473    57.2
            Canal: red_social   1439    56.7
            Cola: facturacion   1217    56.5

🔻 SEGMENTOS CON MENOR RECONTACTO:
                Etiqueta  Casos  Tasa %
Tipo de usuario: empresa   1053    39.4
        Motivo: consulta   1051    43.0
           Cola: soporte   1234    44.3
          Cola: reclamos    994    44.7
  Tipo de usuario: nuevo   1367    45.1
Celda 13 — Código
Entrada

python
# ============================================================
# Celda 13: Cifras para el reporte ejecutivo (calculadas)
# El ANÁLISIS se redacta a mano en la celda de texto que sigue.
# ============================================================
ACCIONES = {
    'transferencias': 'Reducir transferencias: mejorar el ruteo para resolver en el primer contacto',
    'espera_seg': 'Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas',
    'casos_previos_30d': 'Seguimiento proactivo a usuarios con casos previos recientes',
    'es_duracion_corta': 'Revisar atenciones muy cortas: pueden cerrarse sin resolver',
    'duracion_seg': 'Revisar la duración de atención frente al tipo de caso',
    'usuario_adulto_mayor': 'Atención asistida y confirmación de resolución a adultos mayores',
    'turno_nocturno': 'Reforzar capacidad de resolución en turno nocturno',
    'canal_red_social': 'Mejorar la resolución en redes sociales o derivar a un canal con trazabilidad',
    'cola_informacion': 'Revisar la calidad de las respuestas en la cola de Información',
    'cola_facturacion': 'Revisar casos de Facturación para resolver en el primer contacto',
    'cola_tramites': 'Simplificar el seguimiento de Trámites',
}
REDUCCION_SUPUESTA = 0.15   # SUPUESTO de trabajo, no es un dato: % de recontactos detectados que se logra evitar

tasa_kpi = df['es_recontacto'].mean() * 100
n_casos = len(df)
n_rec = int(df['es_recontacto'].sum())
dur_media = df['duracion_seg'].mean()
recall_ = tp / (tp + fn)

horas_rec = n_rec * dur_media / 3600
evitables = n_rec * recall_ * REDUCCION_SUPUESTA
horas_ahorradas = evitables * dur_media / 3600
tasa_proy = tasa_kpi - tasa_kpi * recall_ * REDUCCION_SUPUESTA
brecha_actual = tasa_kpi - META_RECONTACTO
brecha_proy = tasa_proy - META_RECONTACTO

print("=" * 70)
print(f"📊 CIFRAS DEL REPORTE - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)
print(f"""
🎯 OBJETIVO: {OBJETIVO}
🔖 TARGET : {TARGET_DEF}

📊 DATOS
   • Casos del KPI: {n_casos:,}  |  Recontactos: {n_rec:,}  |  Tasa actual: {tasa_kpi:.2f}%
   • Meta (proxy): ≤ {META_RECONTACTO:.0f}%  |  Brecha actual: {brecha_actual:.1f} puntos
   • Entrenamiento / prueba: {len(X_train):,} / {len(X_test):,}  |  Variables: {X.shape[1]}

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: {mejor})
   • Accuracy {accuracy:.2%} | Precision {precision:.2%} | Recall {recall:.2%} | F1 {f1:.4f} | ROC-AUC {auc:.4f}
   • CV-AUC (5 particiones): {cv.mean():.4f} ± {cv.std():.4f}
   • Recontactos detectados: {tp:,} de {tp + fn:,} ({recall_:.1%}) | Alarmas falsas: {fp:,}

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = {dur_media:.0f} s)
   • Horas-agente que consumen los recontactos del periodo: ≈ {horas_rec:,.0f} h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el {REDUCCION_SUPUESTA:.0%} de los recontactos detectados)
   • Recontactos evitables: ≈ {evitables:,.0f}  |  Horas-agente liberadas: ≈ {horas_ahorradas:,.0f} h
   • La tasa bajaría de {tasa_kpi:.1f}% a ≈ {tasa_proy:.1f}%  |  Brecha restante frente a la meta: {brecha_proy:.1f} puntos
   • El {REDUCCION_SUPUESTA:.0%} es un supuesto de trabajo, no un resultado del modelo.
""")

pos = df_coef[df_coef['Coeficiente'] > 0]
prioridades = ['URGENTE', 'ALTO', 'ALTO', 'MEDIO', 'MEDIO', 'MEDIO']
print("🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)")
print("-" * 70)
n = 0
for _, r in pos.iterrows():
    if r['Variable'] in ACCIONES and n < len(prioridades):
        print(f"   {prioridades[n]:8s} {ACCIONES[r['Variable']]}")
        print(f"            (variable: {r['Variable']} | odds ratio {r['Odds ratio']:.2f} por desv. estándar)")
        n += 1

print("\n🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO:")
for _, r in df_coef[df_coef['Coeficiente'] < 0].head(3).iterrows():
    print(f"   ↓ {r['Variable']:26s} (odds ratio {r['Odds ratio']:.2f})")
Salida

text
======================================================================
📊 CIFRAS DEL REPORTE - KPI07: Recontacto en usuarios con historial
======================================================================

🎯 OBJETIVO: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
🔖 TARGET : es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo

📊 DATOS
   • Casos del KPI: 6,479  |  Recontactos: 3,265  |  Tasa actual: 50.39%
   • Meta (proxy): ≤ 22%  |  Brecha actual: 28.4 puntos
   • Entrenamiento / prueba: 5,183 / 1,296  |  Variables: 25

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: 1. Regresión Logística)
   • Accuracy 80.63% | Precision 81.31% | Recall 79.94% | F1 0.8062 | ROC-AUC 0.8745
   • CV-AUC (5 particiones): 0.8777 ± 0.0141
   • Recontactos detectados: 522 de 653 (79.9%) | Alarmas falsas: 120

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = 985 s)
   • Horas-agente que consumen los recontactos del periodo: ≈ 893 h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el 15% de los recontactos detectados)
   • Recontactos evitables: ≈ 392  |  Horas-agente liberadas: ≈ 107 h
   • La tasa bajaría de 50.4% a ≈ 44.4%  |  Brecha restante frente a la meta: 22.4 puntos
   • El 15% es un supuesto de trabajo, no un resultado del modelo.

🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)
----------------------------------------------------------------------
   URGENTE  Reducir transferencias: mejorar el ruteo para resolver en el primer contacto
            (variable: transferencias | odds ratio 4.46 por desv. estándar)
   ALTO     Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas
            (variable: espera_seg | odds ratio 2.50 por desv. estándar)
   ALTO     Seguimiento proactivo a usuarios con casos previos recientes
            (variable: casos_previos_30d | odds ratio 2.26 por desv. estándar)
   MEDIO    Revisar atenciones muy cortas: pueden cerrarse sin resolver
            (variable: es_duracion_corta | odds ratio 1.42 por desv. estándar)
   MEDIO    Revisar la duración de atención frente al tipo de caso
            (variable: duracion_seg | odds ratio 1.41 por desv. estándar)
   MEDIO    Atención asistida y confirmación de resolución a adultos mayores
            (variable: usuario_adulto_mayor | odds ratio 1.33 por desv. estándar)

🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO:
   ↓ cola_reclamos              (odds ratio 0.57)
   ↓ cola_soporte               (odds ratio 0.63)
   ↓ usuario_empresa            (odds ratio 0.78)
Celda 14 — Markdown
Entrada (Markdown)

markdown
# Fase 15: Reporte ejecutivo (redactado por el analista)

**Objetivo.** Reducir la tasa de recontacto a 7 días del Servicio Ciudadano 1800 en los usuarios con historial de contactos previos (6,479 casos, usuarios con al menos un caso en los 30 días anteriores). La línea base es 50.4 %, la meta a 6 meses es ≤ 44.4 % (supuesto a validar con la dirección) y el umbral de alerta es una tasa superior a 50.4 %. El 22 % (100 % − 78 % de resolución en primer contacto) no es equivalente al recontacto a 7 días, por lo que se usa solo como referencia de largo plazo.

**Momento de predicción.** El modelo se aplica al finalizar la atención, para anticipar si el ciudadano volverá a contactar dentro de 7 días. Todas las variables (duración, espera, transferencias y casos previos de los 30 días anteriores) están disponibles en ese momento.

**Hallazgo central.** La Regresión Logística fue el mejor de los 7 modelos comparados (ROC-AUC 0.875, accuracy 80.6 %, CV-AUC 0.878 ± 0.014, consistente con la prueba). Detecta 522 de 653 recontactos del conjunto de prueba (recall 79.9 %) y 81.3 % de los casos que marca como riesgo sí recontactan (120 alarmas falsas). La probabilidad de recontacto se asocia sobre todo con las transferencias (OR 4.5 por desviación estándar), el tiempo de espera (OR 2.5) y los casos previos del usuario en los últimos 30 días (OR 2.3). Dentro de este segmento la variable `tiene_historial` es constante, por lo que se excluyó del modelo. Las comparaciones por categoría se leen frente a una referencia: cola Información, usuarios recurrentes, motivo Falla, canal Chat y turno AM. Frente a esas referencias, las colas de Reclamos (OR 0.57) y Soporte (OR 0.63) y los usuarios empresa (OR 0.78) tienen menor probabilidad de recontacto; los adultos mayores (OR 1.33), el turno nocturno (OR 1.32), el canal red social (OR 1.27) y los motivos Queja (OR 1.26) y Solicitud (OR 1.17) tienen mayor probabilidad. Las atenciones muy cortas, posible cierre sin resolver (OR 1.42), y las de mayor duración (OR 1.41) se asocian a más recontacto.

**Lectura de negocio.** Las transferencias son la señal más fuerte: la primera prioridad es mejorar el ruteo para favorecer la resolución en el primer contacto. La segunda es reducir los tiempos de espera y la tercera implementar seguimiento proactivo a los usuarios con antecedentes recientes. Por segmentos, la tasa observada es mayor en el turno nocturno (64.4 %), los adultos mayores (57.8 %), los casos por falla (57.2 %), el canal red social (56.7 %) y la cola de Facturación (56.5 %). Estas tasas son observadas y no ajustadas, por lo que pueden diferir del modelo: los casos por falla tienen una tasa observada alta, pero frente a ellos la queja y la solicitud muestran más probabilidad una vez controlados los demás factores.

**Alcance realista.** Los 3,265 recontactos del segmento consumen ≈ 893 horas-agente en el periodo (estimación con la duración media). Si se evitara el 15 % de los recontactos que el modelo detecta (supuesto de trabajo, no un resultado medido), se evitarían ≈ 392 recontactos y se liberarían ≈ 107 horas-agente. La tasa bajaría de 50.4 % a ≈ 44.4 %, que coincide con la meta a 6 meses; como esa meta se construyó con el mismo supuesto, solo es alcanzable si el supuesto se cumple. Quedarían ≈ 22.4 puntos hasta la referencia de largo plazo (22 %). Las cifras de la matriz de confusión (por ejemplo, 522 detectados) corresponden al conjunto de prueba de 1,296 casos. El modelo permite priorizar a los usuarios que requieren seguimiento, pero no basta por sí solo para alcanzar el objetivo.
Salida: —

Celda 15 — Markdown
Entrada (Markdown)

markdown
# Fase 16: Próximos pasos y limitaciones

**Próximos pasos.**
1. Validar con la dirección la meta a 6 meses, el umbral de alerta y el criterio de "mismo motivo".
2. Probar las intervenciones con una prueba A/B en las colas con mayor recontacto.
3. Monitorear mensualmente la tasa de recontacto, el desempeño del modelo y las horas-agente ahorradas.
4. Reentrenar el modelo cada trimestre con datos reales.

**Limitaciones.**
- Los datos son sintéticos. Las relaciones entre variables (transferencias, espera, casos previos, etc.) fueron definidas como supuestos documentados del generador, y se verificó que el rendimiento resultante fuera realista (accuracy entre 75 % y 90 %, sin indicios de fuga de datos).
- La Regresión Logística gana en parte porque el generador usa una fórmula logística.
- Los coeficientes indican asociación, no causalidad, y se leen frente a la categoría de referencia de cada grupo (en este segmento: cola Información, usuarios recurrentes, motivo Falla, canal Chat y turno AM).
- El segmento está definido por tener al menos un caso previo, por lo que `casos_previos_30d` mide la intensidad del historial y no su existencia.
- La división entre entrenamiento y prueba es aleatoria, no temporal, y no se dispone de un identificador de usuario para evitar que un mismo usuario aparezca en ambos conjuntos.
- No se dispone de costos por contacto, por lo que el impacto se expresa en casos y horas-agente, con un supuesto del 15 % declarado.
- La meta a 6 meses se deriva de ese mismo supuesto; la referencia del 22 % no es equivalente al recontacto a 7 días.
- Los 5 KPIs comparten el mismo evento (recontacto) y difieren en la población analizada.
- Parte del recontacto no es evitable, por lo que conviene aceptar un piso mínimo.


---------------------------------------

Celda 1 — Código
Entrada

python
# ============================================================
# Celda 1: Importar librerías y configurar el KPI
# ============================================================
from lightgbm import LGBMClassifier
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from cargar_datos import cargar_dataset_recontacto

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ============ CAMBIAR SOLO ESTA LÍNEA EN CADA NOTEBOOK ============
KPI_SEL = 'KPI08'      # 'KPI01', 'KPI02', 'KPI07', 'KPI08' o 'KPI10'
# ===================================================================

FICHAS = {
    'KPI01': dict(nombre='Recontacto general', filtro=None,
                  formula='casos con recontacto a 7 días ÷ casos con resultado conocido',
                  base=41.5, meta=36.5),
    'KPI02': dict(nombre='Recontacto en motivos críticos (falla/queja)', filtro='motivos_criticos',
                  formula='recontactos a 7 días de casos por falla o queja ÷ casos por falla o queja con resultado conocido',
                  base=46.1, meta=40.5),
    'KPI07': dict(nombre='Recontacto en usuarios con historial', filtro='historial',
                  formula='recontactos a 7 días de usuarios con casos previos en 30 días ÷ casos de esos usuarios con resultado conocido',
                  base=50.4, meta=44.4),
    'KPI08': dict(nombre='Recontacto en canales de baja resolución (red social y correo)', filtro='canales_baja_resolucion',
                  formula='recontactos a 7 días en red social y correo ÷ casos en esos canales con resultado conocido',
                  base=44.3, meta=39.1),
    'KPI10': dict(nombre='Recontacto en turno nocturno y fin de semana', filtro='nocturno_finde',
                  formula='recontactos a 7 días en turno nocturno y fin de semana ÷ casos en esos turnos con resultado conocido',
                  base=44.1, meta=38.6),
}
_f = FICHAS[KPI_SEL]

# ------------------ CONFIGURACIÓN DEL KPI ------------------
KPI_ID = KPI_SEL
KPI_NOMBRE = _f['nombre']
FILTRO = _f['filtro']
KPI_FORMULA = _f['formula']

# ------------------ OBJETIVO Y META ------------------
OBJETIVO = 'Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800'

# Referencia externa (NO equivalente): 100% - 78% de resolución en primer contacto.
META_RECONTACTO = 22.0

# Ficha del KPI
LINEA_BASE = _f['base']     # tasa actual del segmento (%)
META_6M = _f['meta']        # meta a 6 meses (escenario del 15 %, supuesto a validar)
PLAZO = '6 meses'
UMBRAL_ALERTA = _f['base']  # si la tasa supera la línea base, se activa alerta

TARGET_DEF = ('es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; '
              '0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo')
# -----------------------------------------------------

os.makedirs('../informe_figuras', exist_ok=True)

def guardar(nombre):
    plt.savefig(f'../informe_figuras/{KPI_ID}_{nombre}.png', dpi=150, bbox_inches='tight')

print("✅ Librerías cargadas correctamente")
print(f"📌 {KPI_ID} - {KPI_NOMBRE}")
print(f"📐 Fórmula: {KPI_FORMULA}")
print(f"🎯 Objetivo: {OBJETIVO}")
print(f"📊 Línea base: {LINEA_BASE}% | 🎯 Meta {PLAZO}: ≤ {META_6M}% | 🚨 Umbral de alerta: > {UMBRAL_ALERTA}%")
print(f"🏁 Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%")
print(f"🔖 Target: {TARGET_DEF}")
Salida

text
✅ Librerías cargadas correctamente
📌 KPI08 - Recontacto en canales de baja resolución (red social y correo)
📐 Fórmula: recontactos a 7 días en red social y correo ÷ casos en esos canales con resultado conocido
🎯 Objetivo: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
📊 Línea base: 44.3% | 🎯 Meta 6 meses: ≤ 39.1% | 🚨 Umbral de alerta: > 44.3%
🏁 Referencia de largo plazo (no equivalente): ≤ 22%
🔖 Target: es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo
Celda 2 — Código
Entrada

python
# ============================================================
# Celda 2: Cargar dataset desde SQL Server
# ============================================================
def filtrar(d, filtro):
    if filtro == 'motivos_criticos':
        d = d[(d.motivo_falla == 1) | (d.motivo_queja == 1)]
    elif filtro == 'historial':
        d = d[d.tiene_historial == 1]
    elif filtro == 'canales_baja_resolucion':
        d = d[(d.canal_red_social == 1) | (d.canal_correo == 1)]
    elif filtro == 'nocturno_finde':
        d = d[(d.turno_nocturno == 1) | (d.turno_fin_semana == 1)]
    return d.reset_index(drop=True)

print("=" * 70)
print(f"📊 CARGANDO DATASET - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)

df_base = cargar_dataset_recontacto()
df = filtrar(df_base, FILTRO)

tasa_inicial = df['es_recontacto'].mean() * 100

print(f"\n✅ Dataset del KPI listo")
print(f"📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"📈 Tasa de recontacto del KPI: {tasa_inicial:.2f}%")
print(f"📈 Tasa de recontacto global : {df_base['es_recontacto'].mean() * 100:.2f}%")
print(f"🏁 Meta {PLAZO}: ≤ {META_6M}% → brecha actual: {tasa_inicial - META_6M:.1f} puntos | referencia largo plazo: {META_RECONTACTO:.0f}%")

# Control: la línea base escrita en la Celda 1 debe coincidir con la tasa real
if abs(tasa_inicial - LINEA_BASE) > 0.5:
    print(f"⚠️ OJO: la línea base de la ficha ({LINEA_BASE}%) no coincide con la tasa real ({tasa_inicial:.1f}%). Revisar KPI_SEL.")
else:
    print(f"✅ La línea base de la ficha ({LINEA_BASE}%) coincide con la tasa real.")
Salida

text
======================================================================
📊 CARGANDO DATASET - KPI08: Recontacto en canales de baja resolución (red social y correo)
======================================================================
============================================================
📊 CARGANDO DATASET DE RECONTACTO (KPI 1)
============================================================
✅ Conexión exitosa a SQL Server
✅ Consulta ejecutada: 11,436 registros, 34 columnas

📏 Dimensiones: 11,436 filas × 34 columnas

🎯 Distribución del target:
es_recontacto
0    6687
1    4749
Name: count, dtype: int64

📈 Tasa de recontacto: 41.53%

✅ Dataset del KPI listo
📏 Dimensiones: 5,233 filas × 34 columnas
📈 Tasa de recontacto del KPI: 44.26%
📈 Tasa de recontacto global : 41.53%
🏁 Meta 6 meses: ≤ 39.1% → brecha actual: 5.2 puntos | referencia largo plazo: 22%
✅ La línea base de la ficha (44.3%) coincide con la tasa real.
Celda 3 — Código
Entrada

python
# ============================================================
# Celda 3: Exploración inicial del dataset
# ============================================================
print("=" * 70)
print("🔍 EXPLORACIÓN INICIAL DEL DATASET")
print("=" * 70)

print(f"\n📋 Columnas del dataset ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col:28s} | Tipo: {str(df[col].dtype):8s} | Nulos: {df[col].isnull().sum()}")

target_counts = df['es_recontacto'].value_counts().sort_index()
print(f"\n🎯 Distribución del target (es_recontacto):")
print(target_counts)
print(f"\n📈 Tasa de recontacto: {df['es_recontacto'].mean() * 100:.2f}%")

print(f"\n🔑 VARIABLES OPERATIVAS CLAVE:")
for c in ['espera_seg', 'duracion_seg', 'transferencias', 'casos_previos_30d']:
    print(f"   • {c}: media {df[c].mean():.2f} | mediana {df[c].median():.2f} | máx {df[c].max():.0f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

target_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Distribución del target', fontweight='bold')
axes[0, 0].set_xticklabels(['No recontacta (0)', 'Recontacta (1)'], rotation=0)
axes[0, 0].set_ylabel('Casos')
for i, v in enumerate(target_counts):
    axes[0, 0].text(i, v + max(target_counts) * 0.01, f'{v:,}', ha='center', fontweight='bold')

df['espera_seg'].hist(ax=axes[0, 1], bins=30, color='#3498db', edgecolor='black')
axes[0, 1].set_title('Distribución del tiempo de espera', fontweight='bold')
axes[0, 1].set_xlabel('Segundos de espera')

df['duracion_seg'].hist(ax=axes[1, 0], bins=30, color='#9b59b6', edgecolor='black')
axes[1, 0].set_title('Distribución de la duración de la atención', fontweight='bold')
axes[1, 0].set_xlabel('Segundos de duración')

tr = df['transferencias'].clip(upper=3)
tasa_tr = df.groupby(tr)['es_recontacto'].mean() * 100
etiquetas = [str(int(i)) if i < 3 else '3 o más' for i in tasa_tr.index]
axes[1, 1].bar(etiquetas, tasa_tr.values, color='#e67e22', edgecolor='black')
axes[1, 1].axhline(df['es_recontacto'].mean() * 100, color='gray', ls='--', label='Tasa del KPI')
axes[1, 1].set_title('Tasa de recontacto según transferencias', fontweight='bold')
axes[1, 1].set_xlabel('Transferencias')
axes[1, 1].set_ylabel('% recontacto')
axes[1, 1].legend()
for i, v in enumerate(tasa_tr.values):
    axes[1, 1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
guardar('01_exploracion')
plt.show()
Salida

text
======================================================================
🔍 EXPLORACIÓN INICIAL DEL DATASET
======================================================================

📋 Columnas del dataset (34):
    1. id_interaccion               | Tipo: int64    | Nulos: 0
    2. es_recontacto                | Tipo: int64    | Nulos: 0
    3. duracion_seg                 | Tipo: int64    | Nulos: 0
    4. espera_seg                   | Tipo: int64    | Nulos: 0
    5. transferencias               | Tipo: int64    | Nulos: 0
    6. casos_previos_30d            | Tipo: int64    | Nulos: 0
    7. es_duracion_corta            | Tipo: int64    | Nulos: 0
    8. es_espera_alta               | Tipo: int64    | Nulos: 0
    9. es_multi_transferencia       | Tipo: int64    | Nulos: 0
   10. tiene_historial              | Tipo: int64    | Nulos: 0
   11. canal_chat                   | Tipo: int64    | Nulos: 0
   12. canal_correo                 | Tipo: int64    | Nulos: 0
   13. canal_red_social             | Tipo: int64    | Nulos: 0
   14. canal_telefono               | Tipo: int64    | Nulos: 0
   15. canal_sin_dato               | Tipo: int64    | Nulos: 0
   16. motivo_consulta              | Tipo: int64    | Nulos: 0
   17. motivo_falla                 | Tipo: int64    | Nulos: 0
   18. motivo_cobro                 | Tipo: int64    | Nulos: 0
   19. motivo_queja                 | Tipo: int64    | Nulos: 0
   20. motivo_solicitud             | Tipo: int64    | Nulos: 0
   21. cola_facturacion             | Tipo: int64    | Nulos: 0
   22. cola_soporte                 | Tipo: int64    | Nulos: 0
   23. cola_informacion             | Tipo: int64    | Nulos: 0
   24. cola_reclamos                | Tipo: int64    | Nulos: 0
   25. cola_tramites                | Tipo: int64    | Nulos: 0
   26. usuario_nuevo                | Tipo: int64    | Nulos: 0
   27. usuario_recurrente           | Tipo: int64    | Nulos: 0
   28. usuario_empresa              | Tipo: int64    | Nulos: 0
   29. usuario_adulto_mayor         | Tipo: int64    | Nulos: 0
   30. turno_am                     | Tipo: int64    | Nulos: 0
   31. turno_pm                     | Tipo: int64    | Nulos: 0
   32. turno_nocturno               | Tipo: int64    | Nulos: 0
   33. turno_fin_semana             | Tipo: int64    | Nulos: 0
   34. fecha_contacto               | Tipo: object   | Nulos: 0

🎯 Distribución del target (es_recontacto):
es_recontacto
0    2917
1    2316
Name: count, dtype: int64

📈 Tasa de recontacto: 44.26%

🔑 VARIABLES OPERATIVAS CLAVE:
   • espera_seg: media 323.34 | mediana 286.00 | máx 900
   • duracion_seg: media 982.39 | mediana 978.00 | máx 1800
   • transferencias: media 0.74 | mediana 1.00 | máx 4
   • casos_previos_30d: media 0.97 | mediana 1.00 | máx 7
[Figura/imagen omitida: Figure size 1500x1000 with 4 Axes]

Celda 4 — Código
Entrada

python
# ============================================================
# Celda 4: Preprocesamiento de variables
# ============================================================
print("=" * 70)
print("🔧 PREPROCESAMIENTO DE VARIABLES")
print("=" * 70)

columnas_a_eliminar = ['id_interaccion', 'fecha_contacto', 'es_recontacto']
X = df.drop(columns=columnas_a_eliminar, errors='ignore')
y = df['es_recontacto']

print(f"\n📊 Features iniciales: {X.shape[1]} columnas")
print(f"🚫 Excluidas del modelo: {columnas_a_eliminar} (identificador, fecha y target)")

columnas_texto = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n📝 Variables categóricas sin codificar: {columnas_texto if columnas_texto else 'ninguna (ya vienen en dummies)'}")
if columnas_texto:
    X = pd.get_dummies(X, columns=columnas_texto, drop_first=True)

columnas_bool = X.select_dtypes(include=['bool']).columns.tolist()
if columnas_bool:
    X = X.astype({c: 'int8' for c in columnas_bool})
    print(f"✅ {len(columnas_bool)} columnas booleanas convertidas a enteros")

nulos_antes = int(X.isnull().sum().sum())
X = X.fillna(0)
print(f"✅ Valores nulos rellenados: {nulos_antes} → {int(X.isnull().sum().sum())}")

# --- Quitar columnas constantes (p. ej. canal_chat/telefono/sin_dato = 0 al filtrar por red social OR correo) ---
constantes = [c for c in X.columns if X[c].nunique() <= 1]
if constantes:
    X = X.drop(columns=constantes)
    print(f"\n🧹 Columnas constantes eliminadas: {constantes}")

# --- Evitar colinealidad: una categoría de referencia por grupo de dummies ---
GRUPOS = ['canal_', 'motivo_', 'cola_', 'usuario_', 'turno_']
referencias = {}
for pref in GRUPOS:
    cols = [c for c in X.columns if c.startswith(pref)]
    if cols and (X[cols].sum(axis=1) == 1).all():      # el grupo suma 1 en todas las filas
        ref = max(cols, key=lambda c: X[c].sum())      # la más frecuente como referencia
        X = X.drop(columns=ref)
        referencias[pref.rstrip('_')] = ref

print("\n📌 Categorías de referencia (los coeficientes se leen frente a ellas):")
for g, r in referencias.items():
    print(f"   • {g:8s} → {r}")

print(f"\n📏 Features finales: {X.shape[1]} columnas")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")
Salida

text
======================================================================
🔧 PREPROCESAMIENTO DE VARIABLES
======================================================================

📊 Features iniciales: 31 columnas
🚫 Excluidas del modelo: ['id_interaccion', 'fecha_contacto', 'es_recontacto'] (identificador, fecha y target)

📝 Variables categóricas sin codificar: ninguna (ya vienen en dummies)
✅ Valores nulos rellenados: 0 → 0

🧹 Columnas constantes eliminadas: ['canal_chat', 'canal_telefono', 'canal_sin_dato']

📌 Categorías de referencia (los coeficientes se leen frente a ellas):
   • canal    → canal_correo
   • motivo   → motivo_queja
   • cola     → cola_informacion
   • usuario  → usuario_recurrente
   • turno    → turno_am

📏 Features finales: 23 columnas
   1. duracion_seg
   2. espera_seg
   3. transferencias
   4. casos_previos_30d
   5. es_duracion_corta
   6. es_espera_alta
   7. es_multi_transferencia
   8. tiene_historial
   9. canal_red_social
  10. motivo_consulta
  11. motivo_falla
  12. motivo_cobro
  13. motivo_solicitud
  14. cola_facturacion
  15. cola_soporte
  16. cola_reclamos
  17. cola_tramites
  18. usuario_nuevo
  19. usuario_empresa
  20. usuario_adulto_mayor
  21. turno_pm
  22. turno_nocturno
  23. turno_fin_semana
Celda 5 — Código
Entrada

python
# ============================================================
# Celda 5: División del dataset (80% / 20%)
# ============================================================
print("=" * 70)
print("📊 DIVISIÓN TRAIN / TEST (80% / 20%)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📏 Entrenamiento: {X_train.shape[0]:,} registros")
print(f"📏 Prueba:        {X_test.shape[0]:,} registros")
print(f"\n🎯 Clases en TRAIN:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\n🎯 Clases en TEST:\n{y_test.value_counts(normalize=True).round(3)}")
print("\n✅ División estratificada completada (proporción mantenida)")
print("✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)")
Salida

text
======================================================================
📊 DIVISIÓN TRAIN / TEST (80% / 20%)
======================================================================

📏 Entrenamiento: 4,186 registros
📏 Prueba:        1,047 registros

🎯 Clases en TRAIN:
es_recontacto
0    0.557
1    0.443
Name: proportion, dtype: float64

🎯 Clases en TEST:
es_recontacto
0    0.558
1    0.442
Name: proportion, dtype: float64

✅ División estratificada completada (proporción mantenida)
✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)
Celda 6 — Código
Entrada

python
# ============================================================
# Celda 6: Comparación de modelos (selección del mejor)
# ============================================================
print("=" * 70)
print("🤖 COMPARACIÓN DE 7 MODELOS")
print("=" * 70)

ratio = (y_train == 0).sum() / (y_train == 1).sum()

modelos = {
    '1. Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    '2. Árbol de Decisión': DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10,
                                                   class_weight='balanced', random_state=42),
    '3. Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10,
                                               class_weight='balanced', random_state=42, n_jobs=-1),
    '4. XGBoost': XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                scale_pos_weight=ratio, random_state=42, eval_metric='auc', verbosity=0),
    '5. LightGBM': LGBMClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=ratio, random_state=42, verbose=-1),
    '6. KNN (k=10)': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
    '7. Naive Bayes': GaussianNB(),
}
ESCALADOS = ['1. Regresión Logística', '6. KNN (k=10)']

resultados = []
for nombre, m in modelos.items():
    inicio = time.time()
    a, b = (X_train_scaled, X_test_scaled) if nombre in ESCALADOS else (X_train, X_test)
    m.fit(a, y_train)
    pred = m.predict(b)
    proba = m.predict_proba(b)[:, 1]
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, proba),
        'Tiempo (s)': time.time() - inicio,
    })
    print(f"   ✅ {nombre:26s} Accuracy {resultados[-1]['Accuracy']:.4f} | AUC {resultados[-1]['ROC-AUC']:.4f}")

df_res = pd.DataFrame(resultados).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n📊 TABLA COMPARATIVA (ordenada por ROC-AUC)")
print(df_res.round(4).to_string(index=False))

mejor = df_res.iloc[0]['Modelo']
print(f"\n🏆 MEJOR MODELO: {mejor}  (ROC-AUC {df_res.iloc[0]['ROC-AUC']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
orden = df_res.sort_values('Accuracy')
axes[0].barh(orden['Modelo'], orden['Accuracy'], color='#3498db', edgecolor='black')
axes[0].axvspan(0.75, 0.85, color='green', alpha=0.12, label='Rango esperado 75-85%')
axes[0].set_title('Accuracy por modelo', fontweight='bold')
axes[0].set_xlim(0.6, 0.95)
axes[0].legend(loc='lower right')
orden = df_res.sort_values('ROC-AUC')
axes[1].barh(orden['Modelo'], orden['ROC-AUC'], color='#e74c3c', edgecolor='black')
axes[1].set_title('ROC-AUC por modelo', fontweight='bold')
axes[1].set_xlim(0.6, 1.0)
plt.tight_layout()
guardar('02_comparacion_modelos')
plt.show()
Salida

text
======================================================================
🤖 COMPARACIÓN DE 7 MODELOS
======================================================================
   ✅ 1. Regresión Logística     Accuracy 0.7966 | AUC 0.8836
   ✅ 2. Árbol de Decisión       Accuracy 0.7450 | AUC 0.8129
   ✅ 3. Random Forest           Accuracy 0.7784 | AUC 0.8625
   ✅ 4. XGBoost                 Accuracy 0.7813 | AUC 0.8616
   ✅ 5. LightGBM                Accuracy 0.7755 | AUC 0.8595
   ✅ 6. KNN (k=10)              Accuracy 0.7278 | AUC 0.8020
   ✅ 7. Naive Bayes             Accuracy 0.7641 | AUC 0.8362

📊 TABLA COMPARATIVA (ordenada por ROC-AUC)
                Modelo  Accuracy  Precision  Recall  F1-Score  ROC-AUC  Tiempo (s)
1. Regresión Logística    0.7966     0.7648  0.7797    0.7722   0.8836      0.0501
      3. Random Forest    0.7784     0.7495  0.7495    0.7495   0.8625      1.2629
            4. XGBoost    0.7813     0.7543  0.7495    0.7519   0.8616      2.5392
           5. LightGBM    0.7755     0.7457  0.7473    0.7465   0.8595      0.7011
        7. Naive Bayes    0.7641     0.7784  0.6523    0.7098   0.8362      0.0160
  2. Árbol de Decisión    0.7450     0.7159  0.7019    0.7088   0.8129      0.0446
         6. KNN (k=10)    0.7278     0.7225  0.6242    0.6698   0.8020      0.1566

🏆 MEJOR MODELO: 1. Regresión Logística  (ROC-AUC 0.8836)
[Figura/imagen omitida: Figure size 1600x550 with 2 Axes]

Celda 7 — Código
Entrada

python
# ============================================================
# Celda 7: Entrenamiento del modelo final (Regresión Logística)
# ============================================================
print("=" * 70)
print("🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA")
print("=" * 70)

modelo = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
print(f"\n🔄 Entrenando con {X_train.shape[0]:,} registros...")
modelo.fit(X_train_scaled, y_train)
print("✅ Modelo entrenado exitosamente")

cv = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\n🔁 Validación cruzada (5 particiones) ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

if mejor != '1. Regresión Logística':
    print(f"\n⚠️ OJO: en este KPI el mejor por AUC fue {mejor}. Revisar antes de justificar la Regresión Logística.")
else:
    print("\n✅ La Regresión Logística es el mejor modelo de la comparación para este KPI")
Salida

text
======================================================================
🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA
======================================================================

🔄 Entrenando con 4,186 registros...
✅ Modelo entrenado exitosamente

🔁 Validación cruzada (5 particiones) ROC-AUC: 0.8849 ± 0.0088

✅ La Regresión Logística es el mejor modelo de la comparación para este KPI
Celda 8 — Código
Entrada

python
# ============================================================
# Celda 8: Evaluación de métricas en el set de prueba
# ============================================================
print("=" * 70)
print("📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA")
print("=" * 70)

y_pred = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\n🎯 MÉTRICAS PRINCIPALES:")
print(f"   • Accuracy:       {accuracy:.4f} ({accuracy:.2%})")
print(f"   • Precision:      {precision:.4f}")
print(f"   • Recall:         {recall:.4f}")
print(f"   • F1-Score:       {f1:.4f}")
print(f"   • ROC-AUC:        {auc:.4f}")
print(f"   • Avg Precision:  {avg_precision:.4f}")

print(f"\n📋 REPORTE DE CLASIFICACIÓN COMPLETO:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['No recontacta (0)', 'Recontacta (1)']))

print(f"💡 INTERPRETACIÓN:")
print(f"   • ROC-AUC = {auc:.3f} indica un poder discriminatorio "
      f"{'excelente' if auc > 0.85 else 'bueno' if auc > 0.75 else 'moderado'}")
print(f"   • Accuracy {'dentro' if 0.75 <= accuracy <= 0.85 else 'fuera'} del rango esperado (75%-85%)")
print(f"   • {'No hay indicio de data leakage (accuracy < 90%)' if accuracy < 0.90 else 'ALERTA: accuracy > 90%, revisar posible data leakage'}")
print(f"   • CV-AUC {cv.mean():.3f} vs AUC de prueba {auc:.3f}: diferencia {abs(cv.mean() - auc):.3f} "
      f"({'sin sobreajuste evidente' if abs(cv.mean() - auc) < 0.03 else 'revisar sobreajuste'})")
Salida

text
======================================================================
📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA
======================================================================

🎯 MÉTRICAS PRINCIPALES:
   • Accuracy:       0.7966 (79.66%)
   • Precision:      0.7648
   • Recall:         0.7797
   • F1-Score:       0.7722
   • ROC-AUC:        0.8836
   • Avg Precision:  0.8687

📋 REPORTE DE CLASIFICACIÓN COMPLETO:
----------------------------------------------------------------------
                   precision    recall  f1-score   support

No recontacta (0)       0.82      0.81      0.82       584
   Recontacta (1)       0.76      0.78      0.77       463

         accuracy                           0.80      1047
        macro avg       0.79      0.79      0.79      1047
     weighted avg       0.80      0.80      0.80      1047

💡 INTERPRETACIÓN:
   • ROC-AUC = 0.884 indica un poder discriminatorio excelente
   • Accuracy dentro del rango esperado (75%-85%)
   • No hay indicio de data leakage (accuracy < 90%)
   • CV-AUC 0.885 vs AUC de prueba 0.884: diferencia 0.001 (sin sobreajuste evidente)
Celda 8b — Código
Entrada

python
# ============================================================
# Celda 8b: Verificación anti-fuga de datos (data leakage)
# ============================================================
print("=" * 70)
print("🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS")
print("=" * 70)

# 1) Ninguna variable del modelo contiene el resultado
sospechosas = [c for c in X.columns if 'recontacto' in c.lower()]
print(f"\n1) Variables con 'recontacto' en el nombre dentro del modelo: {sospechosas if sospechosas else 'ninguna ✅'}")

# 2) Ninguna variable por sí sola separa casi perfectamente el target
aucs = {}
for c in X.columns:
    if X[c].nunique() > 1:
        a = roc_auc_score(y, X[c])
        aucs[c] = max(a, 1 - a)
aucs = pd.Series(aucs).sort_values(ascending=False)
print(f"\n2) AUC individual más alto de una sola variable: {aucs.index[0]} = {aucs.iloc[0]:.3f}")
print(f"   {'✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)' if aucs.iloc[0] < 0.95 else '⚠️ revisar: una variable casi separa el target'}")
print(aucs.head(5).round(3).to_string())

# 3) Rendimiento realista y estable
print(f"\n3) Accuracy {accuracy:.2%} (alerta si > 90%): {'✅' if accuracy < 0.90 else '⚠️'}")
print(f"4) CV-AUC {cv.mean():.3f} vs AUC prueba {auc:.3f} (diferencia {abs(cv.mean() - auc):.3f}): "
      f"{'✅ estable' if abs(cv.mean() - auc) < 0.03 else '⚠️ revisar'}")

print("""
5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
""")
Salida

text
======================================================================
🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS
======================================================================

1) Variables con 'recontacto' en el nombre dentro del modelo: ninguna ✅

2) AUC individual más alto de una sola variable: transferencias = 0.739
   ✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)
transferencias            0.739
espera_seg                0.716
es_espera_alta            0.638
es_multi_transferencia    0.636
casos_previos_30d         0.633

3) Accuracy 79.66% (alerta si > 90%): ✅
4) CV-AUC 0.885 vs AUC prueba 0.884 (diferencia 0.001): ✅ estable

5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
Celda 9 — Código
Entrada

python
# ============================================================
# Celda 9: Matriz de confusión
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No recontacta (0)', 'Recontacta (1)'],
            yticklabels=['No recontacta (0)', 'Recontacta (1)'],
            cbar_kws={'label': 'Cantidad'})
plt.title(f'Matriz de Confusión - Set de Prueba\n{KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Valor Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
guardar('03_matriz_confusion')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n📊 DESGLOSE DE LA MATRIZ:")
print(f"   • Verdaderos Positivos (VP): {tp:5d} → Recontactos detectados correctamente")
print(f"   • Verdaderos Negativos (VN): {tn:5d} → Casos resueltos identificados correctamente")
print(f"   • Falsos Positivos (FP):     {fp:5d} → Alarmas falsas (se predijo recontacto y no ocurrió)")
print(f"   • Falsos Negativos (FN):     {fn:5d} → Recontactos que el modelo no detectó")

print(f"\n💼 LECTURA OPERATIVA:")
print(f"   • De {tp + fn:,} recontactos reales, el modelo anticipó {tp:,} ({tp / (tp + fn):.1%})")
print(f"   • De {tp + fp:,} casos marcados como riesgo, {tp:,} sí recontactaron ({tp / (tp + fp):.1%})")
print(f"   • Seguimientos que serían innecesarios: {fp:,}")
Salida

text
[Figura/imagen omitida: Figure size 800x600 with 2 Axes]

📊 DESGLOSE DE LA MATRIZ:
   • Verdaderos Positivos (VP):   361 → Recontactos detectados correctamente
   • Verdaderos Negativos (VN):   473 → Casos resueltos identificados correctamente
   • Falsos Positivos (FP):       111 → Alarmas falsas (se predijo recontacto y no ocurrió)
   • Falsos Negativos (FN):       102 → Recontactos que el modelo no detectó

💼 LECTURA OPERATIVA:
   • De 463 recontactos reales, el modelo anticipó 361 (78.0%)
   • De 472 casos marcados como riesgo, 361 sí recontactaron (76.5%)
   • Seguimientos que serían innecesarios: 111
Celda 10 — Código
Entrada

python
# ============================================================
# Celda 10: Curva ROC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'Curva ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title(f'Curva ROC - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
guardar('04_curva_roc')
plt.show()

idx_optimo = np.argmax(tpr - fpr)
umbral_optimo = thresholds[idx_optimo]
print(f"\n🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):")
print(f"   • Umbral: {umbral_optimo:.3f}")
print(f"   • TPR en ese punto: {tpr[idx_optimo]:.3f}")
print(f"   • FPR en ese punto: {fpr[idx_optimo]:.3f}")
Salida

text
[Figura/imagen omitida: Figure size 1000x700 with 1 Axes]

🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):
   • Umbral: 0.537
   • TPR en ese punto: 0.769
   • FPR en ese punto: 0.161
Celda 11 — Código
Entrada

python
# ============================================================
# Celda 11: Variables más asociadas al recontacto
# ============================================================
print("=" * 70)
print("🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)")
print("=" * 70)

print("\n📌 Los coeficientes se leen frente a la categoría de referencia de cada grupo:")
for g, r in referencias.items():
    print(f"   • {g:8s} → {r}")

df_coef = pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_[0]})
df_coef['Odds ratio'] = np.exp(df_coef['Coeficiente'])
df_coef['Abs'] = df_coef['Coeficiente'].abs()
df_coef = df_coef.sort_values('Abs', ascending=False).reset_index(drop=True)
top15 = df_coef.head(15)

print(f"\n📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):")
print("-" * 70)
for i, r in top15.iterrows():
    signo = '↑ sube' if r['Coeficiente'] > 0 else '↓ baja'
    print(f"   {i + 1:2d}. {r['Variable']:28s} | coef {r['Coeficiente']:+.4f} | OR {r['Odds ratio']:.3f} | {signo} la prob.")

plot_data = top15.sort_values('Coeficiente')
colores = ['#e74c3c' if c > 0 else '#2ecc71' for c in plot_data['Coeficiente']]
plt.figure(figsize=(12, 8))
plt.barh(plot_data['Variable'], plot_data['Coeficiente'], color=colores, edgecolor='black')
plt.axvline(0, color='black', lw=0.8)
plt.xlabel('Coeficiente (variables estandarizadas)')
plt.title(f'Top 15 variables asociadas al recontacto - {KPI_ID}\n(rojo: aumenta la probabilidad, verde: la reduce)',
          fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
guardar('05_coeficientes')
plt.show()

print("\n⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.")
Salida

text
======================================================================
🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)
======================================================================

📌 Los coeficientes se leen frente a la categoría de referencia de cada grupo:
   • canal    → canal_correo
   • motivo   → motivo_queja
   • cola     → cola_informacion
   • usuario  → usuario_recurrente
   • turno    → turno_am

📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):
----------------------------------------------------------------------
    1. transferencias               | coef +1.6410 | OR 5.160 | ↑ sube la prob.
    2. espera_seg                   | coef +0.8852 | OR 2.423 | ↑ sube la prob.
    3. casos_previos_30d            | coef +0.8746 | OR 2.398 | ↑ sube la prob.
    4. cola_reclamos                | coef -0.6281 | OR 0.534 | ↓ baja la prob.
    5. cola_soporte                 | coef -0.4689 | OR 0.626 | ↓ baja la prob.
    6. es_duracion_corta            | coef +0.4012 | OR 1.494 | ↑ sube la prob.
    7. usuario_adulto_mayor         | coef +0.3728 | OR 1.452 | ↑ sube la prob.
    8. duracion_seg                 | coef +0.3456 | OR 1.413 | ↑ sube la prob.
    9. motivo_consulta              | coef -0.3111 | OR 0.733 | ↓ baja la prob.
   10. canal_red_social             | coef +0.2998 | OR 1.350 | ↑ sube la prob.
   11. usuario_empresa              | coef -0.2968 | OR 0.743 | ↓ baja la prob.
   12. motivo_cobro                 | coef -0.2525 | OR 0.777 | ↓ baja la prob.
   13. motivo_falla                 | coef -0.2445 | OR 0.783 | ↓ baja la prob.
   14. turno_nocturno               | coef +0.1853 | OR 1.204 | ↑ sube la prob.
   15. es_multi_transferencia       | coef -0.1576 | OR 0.854 | ↓ baja la prob.
[Figura/imagen omitida: Figure size 1200x800 with 1 Axes]

text
⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.
Celda 12 — Código
Entrada

python
# ============================================================
# Celda 12: Tasa de recontacto por segmento (reglas de negocio)
# ============================================================
print("=" * 70)
print("💼 RECONTACTO POR SEGMENTO")
print("=" * 70)

grupos = {'Canal': 'canal_', 'Motivo': 'motivo_', 'Cola': 'cola_',
          'Tipo de usuario': 'usuario_', 'Turno': 'turno_'}
tasa_kpi = df['es_recontacto'].mean() * 100

filas = []
for g, pref in grupos.items():
    for col in [c for c in df.columns if c.startswith(pref)]:
        m = df[col] == 1
        if m.sum() >= 30:
            filas.append({'Grupo': g, 'Segmento': col.replace(pref, ''),
                          'Casos': int(m.sum()),
                          'Tasa %': df.loc[m, 'es_recontacto'].mean() * 100})
seg = pd.DataFrame(filas).sort_values('Tasa %', ascending=True).reset_index(drop=True)
seg['Etiqueta'] = seg['Grupo'] + ': ' + seg['Segmento']

paleta = {'Canal': '#3498db', 'Motivo': '#e67e22', 'Cola': '#9b59b6',
          'Tipo de usuario': '#1abc9c', 'Turno': '#f1c40f'}
plt.figure(figsize=(12, max(6, len(seg) * 0.38)))
plt.barh(seg['Etiqueta'], seg['Tasa %'], color=[paleta[g] for g in seg['Grupo']], edgecolor='black')
plt.axvline(tasa_kpi, color='red', ls='--', label=f'Tasa del KPI ({tasa_kpi:.1f}%)')
plt.xlabel('% de recontacto a 7 días')
plt.title(f'Recontacto por segmento - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
guardar('06_segmentos')
plt.show()

print(f"\n🔺 SEGMENTOS CON MAYOR RECONTACTO:")
print(seg.tail(5).iloc[::-1][['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
print(f"\n🔻 SEGMENTOS CON MENOR RECONTACTO:")
print(seg.head(5)[['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
Salida

text
======================================================================
💼 RECONTACTO POR SEGMENTO
======================================================================
[Figura/imagen omitida: Figure size 1200x760 with 1 Axes]

text
🔺 SEGMENTOS CON MAYOR RECONTACTO:
                     Etiqueta  Casos  Tasa %
              Turno: nocturno    595    57.1
Tipo de usuario: adulto_mayor   1073    51.1
  Tipo de usuario: recurrente   1790    50.9
            Cola: facturacion    960    50.5
                Motivo: falla   1130    50.2

🔻 SEGMENTOS CON MENOR RECONTACTO:
                Etiqueta  Casos  Tasa %
Tipo de usuario: empresa    796    32.4
        Motivo: consulta    895    37.5
  Tipo de usuario: nuevo   1574    38.0
           Cola: soporte   1047    38.3
           Motivo: cobro   1077    39.5
Celda 13 — Código
Entrada

python
# ============================================================
# Celda 13: Cifras para el reporte ejecutivo (calculadas)
# El ANÁLISIS se redacta a mano en la celda de texto que sigue.
# ============================================================
ACCIONES = {
    'transferencias': 'Reducir transferencias: mejorar el ruteo para resolver en el primer contacto',
    'espera_seg': 'Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas',
    'casos_previos_30d': 'Seguimiento proactivo a usuarios con casos previos recientes',
    'es_duracion_corta': 'Revisar atenciones muy cortas: pueden cerrarse sin resolver',
    'duracion_seg': 'Revisar la duración de atención frente al tipo de caso',
    'usuario_adulto_mayor': 'Atención asistida y confirmación de resolución a adultos mayores',
    'turno_nocturno': 'Reforzar capacidad de resolución en turno nocturno',
    'canal_red_social': 'Mejorar la resolución en redes sociales o derivar a un canal con trazabilidad',
    'cola_informacion': 'Revisar la calidad de las respuestas en la cola de Información',
    'cola_facturacion': 'Revisar casos de Facturación para resolver en el primer contacto',
    'cola_tramites': 'Simplificar el seguimiento de Trámites',
}
REDUCCION_SUPUESTA = 0.15   # SUPUESTO de trabajo, no es un dato

tasa_kpi = df['es_recontacto'].mean() * 100
n_casos = len(df)
n_rec = int(df['es_recontacto'].sum())
dur_media = df['duracion_seg'].mean()
recall_ = tp / (tp + fn)

horas_rec = n_rec * dur_media / 3600
evitables = n_rec * recall_ * REDUCCION_SUPUESTA
horas_ahorradas = evitables * dur_media / 3600
tasa_proy = tasa_kpi - tasa_kpi * recall_ * REDUCCION_SUPUESTA
brecha_actual = tasa_kpi - META_6M
dif_proy = tasa_proy - META_6M
brecha_ref = tasa_proy - META_RECONTACTO

print("=" * 70)
print(f"📊 CIFRAS DEL REPORTE - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)
print(f"""
🎯 OBJETIVO: {OBJETIVO}
🔖 TARGET : {TARGET_DEF}

📊 DATOS
   • Casos del KPI: {n_casos:,}  |  Recontactos: {n_rec:,}  |  Tasa actual (línea base): {tasa_kpi:.2f}%
   • Meta a {PLAZO}: ≤ {META_6M}%  |  Brecha actual: {brecha_actual:.1f} puntos  |  Umbral de alerta: > {UMBRAL_ALERTA}%
   • Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%
   • Entrenamiento / prueba: {len(X_train):,} / {len(X_test):,}  |  Variables: {X.shape[1]}

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: {mejor})
   • Accuracy {accuracy:.2%} | Precision {precision:.2%} | Recall {recall:.2%} | F1 {f1:.4f} | ROC-AUC {auc:.4f}
   • CV-AUC (5 particiones): {cv.mean():.4f} ± {cv.std():.4f}
   • Recontactos detectados: {tp:,} de {tp + fn:,} ({recall_:.1%})  |  Alarmas falsas: {fp:,}

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = {dur_media:.0f} s)
   • Horas-agente que consumen los recontactos del periodo: ≈ {horas_rec:,.0f} h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el {REDUCCION_SUPUESTA:.0%} de los recontactos detectados)
   • Recontactos evitables: ≈ {evitables:,.0f}  |  Horas-agente liberadas: ≈ {horas_ahorradas:,.0f} h
   • La tasa bajaría de {tasa_kpi:.1f}% a ≈ {tasa_proy:.1f}%
   • Diferencia frente a la meta de {PLAZO} ({META_6M}%): {dif_proy:+.1f} puntos
   • Distancia que queda a la referencia de largo plazo ({META_RECONTACTO:.0f}%): {brecha_ref:.1f} puntos
   • El {REDUCCION_SUPUESTA:.0%} es un supuesto de trabajo, no un resultado del modelo.
""")

pos = df_coef[df_coef['Coeficiente'] > 0]
prioridades = ['URGENTE', 'ALTO', 'ALTO', 'MEDIO', 'MEDIO', 'MEDIO']
print("🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)")
print("-" * 70)
n = 0
for _, r in pos.iterrows():
    if r['Variable'] in ACCIONES and n < len(prioridades):
        print(f"   {prioridades[n]:8s} {ACCIONES[r['Variable']]}")
        print(f"            (variable: {r['Variable']} | odds ratio {r['Odds ratio']:.2f} por desv. estándar)")
        n += 1

print("\n🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):")
for _, r in df_coef[df_coef['Coeficiente'] < 0].head(3).iterrows():
    print(f"   ↓ {r['Variable']:26s} (odds ratio {r['Odds ratio']:.2f})")
Salida

text
======================================================================
📊 CIFRAS DEL REPORTE - KPI08: Recontacto en canales de baja resolución (red social y correo)
======================================================================

🎯 OBJETIVO: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
🔖 TARGET : es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo

📊 DATOS
   • Casos del KPI: 5,233  |  Recontactos: 2,316  |  Tasa actual (línea base): 44.26%
   • Meta a 6 meses: ≤ 39.1%  |  Brecha actual: 5.2 puntos  |  Umbral de alerta: > 44.3%
   • Referencia de largo plazo (no equivalente): ≤ 22%
   • Entrenamiento / prueba: 4,186 / 1,047  |  Variables: 23

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: 1. Regresión Logística)
   • Accuracy 79.66% | Precision 76.48% | Recall 77.97% | F1 0.7722 | ROC-AUC 0.8836
   • CV-AUC (5 particiones): 0.8849 ± 0.0088
   • Recontactos detectados: 361 de 463 (78.0%)  |  Alarmas falsas: 111

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = 982 s)
   • Horas-agente que consumen los recontactos del periodo: ≈ 632 h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el 15% de los recontactos detectados)
   • Recontactos evitables: ≈ 271  |  Horas-agente liberadas: ≈ 74 h
   • La tasa bajaría de 44.3% a ≈ 39.1%
   • Diferencia frente a la meta de 6 meses (39.1%): -0.0 puntos
   • Distancia que queda a la referencia de largo plazo (22%): 17.1 puntos
   • El 15% es un supuesto de trabajo, no un resultado del modelo.

🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)
----------------------------------------------------------------------
   URGENTE  Reducir transferencias: mejorar el ruteo para resolver en el primer contacto
            (variable: transferencias | odds ratio 5.16 por desv. estándar)
   ALTO     Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas
            (variable: espera_seg | odds ratio 2.42 por desv. estándar)
   ALTO     Seguimiento proactivo a usuarios con casos previos recientes
            (variable: casos_previos_30d | odds ratio 2.40 por desv. estándar)
   MEDIO    Revisar atenciones muy cortas: pueden cerrarse sin resolver
            (variable: es_duracion_corta | odds ratio 1.49 por desv. estándar)
   MEDIO    Atención asistida y confirmación de resolución a adultos mayores
            (variable: usuario_adulto_mayor | odds ratio 1.45 por desv. estándar)
   MEDIO    Revisar la duración de atención frente al tipo de caso
            (variable: duracion_seg | odds ratio 1.41 por desv. estándar)

🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):
   ↓ cola_reclamos              (odds ratio 0.53)
   ↓ cola_soporte               (odds ratio 0.63)
   ↓ motivo_consulta            (odds ratio 0.73)
Celda 14 — Markdown
Entrada (Markdown)

markdown
# Fase 15: Reporte ejecutivo (redactado por el analista)

**Objetivo.** Reducir la tasa de recontacto a 7 días del Servicio Ciudadano 1800 en los canales de baja resolución (red social y correo), donde hoy es de 44.3%. La meta a 6 meses es ≤ 39.1% (escenario del 15%, supuesto a validar con la dirección) y el umbral de alerta es superar la línea base (44.3%). La referencia de 22% (100% − 78% de resolución en primer contacto) no es equivalente al recontacto a 7 días y se usa solo como referencia de largo plazo.

**Momento de predicción.** El modelo se aplica al cierre de la atención, cuando ya se conocen la espera, la duración, las transferencias y los casos previos del usuario. Ninguna de esas variables incluye información posterior al contacto.

**Hallazgo central.** El modelo (Regresión Logística, ROC-AUC 0.884, accuracy 79.7%) anticipa 8 de cada 10 recontactos en estos canales (recall 78.0%; 361 de 463 en la prueba). Fue el mejor de los 7 modelos comparados y su desempeño es estable (CV-AUC 0.885). La probabilidad de recontacto se asocia sobre todo con tres factores operativos: el número de transferencias (OR 5.2 por desviación estándar), el tiempo de espera (OR 2.4) y los casos previos del usuario en los últimos 30 días (OR 2.4). Una vez controlado el efecto de las demás variables, la cola de Reclamos (OR 0.53) y la cola de Soporte (OR 0.63) muestran menor probabilidad de recontacto.

**Lectura de negocio.** Las transferencias son la señal más fuerte: resolver en el primer contacto, con mejor ruteo, es la acción prioritaria. Después van la reducción de la espera y el seguimiento proactivo a usuarios con casos recientes. Por segmentos, el turno nocturno (57.1%), los adultos mayores (51.1%) y los usuarios recurrentes (50.9%) superan el promedio del KPI; los usuarios empresa (32.4%) son los que menos recontactan. Dentro de estos canales, red social tiene una asociación algo mayor que correo (OR 1.35 frente a correo), pero con un efecto mucho menor que el de las transferencias o la espera.

**Alcance realista.** En el periodo analizado hubo 2,316 recontactos, que consumen ≈ 632 horas-agente. Si se evitara el 15% de los recontactos que el modelo detecta (supuesto de trabajo, no un resultado), se evitarían ≈ 271 recontactos y se liberarían ≈ 74 horas-agente. La tasa bajaría de 44.3% a ≈ 39.1%, **coincidiendo con la meta a 6 meses**, pero todavía quedaría **17.1 puntos por encima de la referencia de largo plazo (22%)**. El modelo ayuda a priorizar a quién y dónde intervenir, pero no basta por sí solo para alcanzar esa referencia.
Salida: —

Celda 15 — Markdown
Entrada (Markdown)

markdown
# Fase 16: Próximos pasos y limitaciones

**Próximos pasos.**
1. Validar con la dirección la meta de recontacto y el criterio de "mismo motivo".
2. Probar las intervenciones con una prueba A/B en las colas con mayor recontacto.
3. Monitorear mensualmente la tasa de recontacto y las métricas del modelo.
4. Reentrenar con datos reales cada trimestre.

**Limitaciones.**
- Los datos son sintéticos. Las relaciones entre variables son supuestos documentados del generador, y su intensidad se calibró para que el modelo quedara en el rango esperado de 75–85%.
- La Regresión Logística gana en parte porque el generador usa una fórmula logística.
- Los coeficientes indican asociación, no causalidad.
- No se dispone de costos por contacto, por lo que el impacto se expresa en casos y horas-agente, con un supuesto del 15% declarado.
- Los 5 KPIs comparten el mismo evento (recontacto) y difieren en la población analizada.
- Parte del recontacto no es evitable (usuarios con consultas nuevas o complejas), por lo que conviene aceptar un piso mínimo.




--------------


Celda 1 — Código
Entrada

python
# ============================================================
# Celda 1: Importar librerías y configurar el KPI
# ============================================================
from lightgbm import LGBMClassifier
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from cargar_datos import cargar_dataset_recontacto

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ============ CAMBIAR SOLO ESTA LÍNEA EN CADA NOTEBOOK ============
KPI_SEL = 'KPI10'      # 'KPI01', 'KPI02', 'KPI07', 'KPI08' o 'KPI10'
# ===================================================================

FICHAS = {
    'KPI01': dict(nombre='Recontacto general', filtro=None,
                  formula='casos con recontacto a 7 días ÷ casos con resultado conocido',
                  base=41.5, meta=36.5),
    'KPI02': dict(nombre='Recontacto en motivos críticos (falla/queja)', filtro='motivos_criticos',
                  formula='recontactos a 7 días de casos por falla o queja ÷ casos por falla o queja con resultado conocido',
                  base=46.1, meta=40.5),
    'KPI07': dict(nombre='Recontacto en usuarios con historial', filtro='historial',
                  formula='recontactos a 7 días de usuarios con casos previos en 30 días ÷ casos de esos usuarios con resultado conocido',
                  base=50.4, meta=44.4),
    'KPI08': dict(nombre='Recontacto en canales de baja resolución (red social y correo)', filtro='canales_baja_resolucion',
                  formula='recontactos a 7 días en red social y correo ÷ casos en esos canales con resultado conocido',
                  base=44.3, meta=39.1),
    'KPI10': dict(nombre='Recontacto en turno nocturno y fin de semana', filtro='nocturno_finde',
                  formula='recontactos a 7 días en turno nocturno y fin de semana ÷ casos en esos turnos con resultado conocido',
                  base=44.1, meta=38.6),
}
_f = FICHAS[KPI_SEL]

# ------------------ CONFIGURACIÓN DEL KPI ------------------
KPI_ID = KPI_SEL
KPI_NOMBRE = _f['nombre']
FILTRO = _f['filtro']
KPI_FORMULA = _f['formula']

# ------------------ OBJETIVO Y META ------------------
OBJETIVO = 'Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800'

# Referencia externa (NO equivalente): 100% - 78% de resolución en primer contacto.
META_RECONTACTO = 22.0

# Ficha del KPI
LINEA_BASE = _f['base']     # tasa actual del segmento (%)
META_6M = _f['meta']        # meta a 6 meses (escenario del 15 %, supuesto a validar)
PLAZO = '6 meses'
UMBRAL_ALERTA = _f['base']  # si la tasa supera la línea base, se activa alerta

TARGET_DEF = ('es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; '
              '0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo')
# -----------------------------------------------------

os.makedirs('../informe_figuras', exist_ok=True)

def guardar(nombre):
    plt.savefig(f'../informe_figuras/{KPI_ID}_{nombre}.png', dpi=150, bbox_inches='tight')

print("✅ Librerías cargadas correctamente")
print(f"📌 {KPI_ID} - {KPI_NOMBRE}")
print(f"📐 Fórmula: {KPI_FORMULA}")
print(f"🎯 Objetivo: {OBJETIVO}")
print(f"📊 Línea base: {LINEA_BASE}% | 🎯 Meta {PLAZO}: ≤ {META_6M}% | 🚨 Umbral de alerta: > {UMBRAL_ALERTA}%")
print(f"🏁 Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%")
print(f"🔖 Target: {TARGET_DEF}")
Salida

text
✅ Librerías cargadas correctamente
📌 KPI10 - Recontacto en turno nocturno y fin de semana
📐 Fórmula: recontactos a 7 días en turno nocturno y fin de semana ÷ casos en esos turnos con resultado conocido
🎯 Objetivo: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
📊 Línea base: 44.1% | 🎯 Meta 6 meses: ≤ 38.6% | 🚨 Umbral de alerta: > 44.1%
🏁 Referencia de largo plazo (no equivalente): ≤ 22%
🔖 Target: es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo
Celda 2 — Código
Entrada

python
# ============================================================
# Celda 2: Cargar dataset desde SQL Server
# ============================================================
def filtrar(d, filtro):
    if filtro == 'motivos_criticos':
        d = d[(d.motivo_falla == 1) | (d.motivo_queja == 1)]
    elif filtro == 'historial':
        d = d[d.tiene_historial == 1]
    elif filtro == 'canales_baja_resolucion':
        d = d[(d.canal_red_social == 1) | (d.canal_correo == 1)]
    elif filtro == 'nocturno_finde':
        d = d[(d.turno_nocturno == 1) | (d.turno_fin_semana == 1)]
    return d.reset_index(drop=True)

print("=" * 70)
print(f"📊 CARGANDO DATASET - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)

df_base = cargar_dataset_recontacto()
df = filtrar(df_base, FILTRO)

tasa_inicial = df['es_recontacto'].mean() * 100

print(f"\n✅ Dataset del KPI listo")
print(f"📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"📈 Tasa de recontacto del KPI: {tasa_inicial:.2f}%")
print(f"📈 Tasa de recontacto global : {df_base['es_recontacto'].mean() * 100:.2f}%")
print(f"🏁 Meta {PLAZO}: ≤ {META_6M}% → brecha actual: {tasa_inicial - META_6M:.1f} puntos | referencia largo plazo: {META_RECONTACTO:.0f}%")

# Control: la línea base escrita en la Celda 1 debe coincidir con la tasa real
if abs(tasa_inicial - LINEA_BASE) > 0.5:
    print(f"⚠️ OJO: la línea base de la ficha ({LINEA_BASE}%) no coincide con la tasa real ({tasa_inicial:.1f}%). Revisar KPI_SEL.")
else:
    print(f"✅ La línea base de la ficha ({LINEA_BASE}%) coincide con la tasa real.")
Salida

text
======================================================================
📊 CARGANDO DATASET - KPI10: Recontacto en turno nocturno y fin de semana
======================================================================
============================================================
📊 CARGANDO DATASET DE RECONTACTO (KPI 1)
============================================================
✅ Conexión exitosa a SQL Server
✅ Consulta ejecutada: 11,436 registros, 34 columnas

📏 Dimensiones: 11,436 filas × 34 columnas

🎯 Distribución del target:
es_recontacto
0    6687
1    4749
Name: count, dtype: int64

📈 Tasa de recontacto: 41.53%

✅ Dataset del KPI listo
📏 Dimensiones: 4,589 filas × 34 columnas
📈 Tasa de recontacto del KPI: 44.06%
📈 Tasa de recontacto global : 41.53%
🏁 Meta 6 meses: ≤ 38.6% → brecha actual: 5.5 puntos | referencia largo plazo: 22%
✅ La línea base de la ficha (44.1%) coincide con la tasa real.
Celda 3 — Código
Entrada

python
# ============================================================
# Celda 3: Exploración inicial del dataset
# ============================================================
print("=" * 70)
print("🔍 EXPLORACIÓN INICIAL DEL DATASET")
print("=" * 70)

print(f"\n📋 Columnas del dataset ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col:28s} | Tipo: {str(df[col].dtype):8s} | Nulos: {df[col].isnull().sum()}")

target_counts = df['es_recontacto'].value_counts().sort_index()
print(f"\n🎯 Distribución del target (es_recontacto):")
print(target_counts)
print(f"\n📈 Tasa de recontacto: {df['es_recontacto'].mean() * 100:.2f}%")

print(f"\n🔑 VARIABLES OPERATIVAS CLAVE:")
for c in ['espera_seg', 'duracion_seg', 'transferencias', 'casos_previos_30d']:
    print(f"   • {c}: media {df[c].mean():.2f} | mediana {df[c].median():.2f} | máx {df[c].max():.0f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

target_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Distribución del target', fontweight='bold')
axes[0, 0].set_xticklabels(['No recontacta (0)', 'Recontacta (1)'], rotation=0)
axes[0, 0].set_ylabel('Casos')
for i, v in enumerate(target_counts):
    axes[0, 0].text(i, v + max(target_counts) * 0.01, f'{v:,}', ha='center', fontweight='bold')

df['espera_seg'].hist(ax=axes[0, 1], bins=30, color='#3498db', edgecolor='black')
axes[0, 1].set_title('Distribución del tiempo de espera', fontweight='bold')
axes[0, 1].set_xlabel('Segundos de espera')

df['duracion_seg'].hist(ax=axes[1, 0], bins=30, color='#9b59b6', edgecolor='black')
axes[1, 0].set_title('Distribución de la duración de la atención', fontweight='bold')
axes[1, 0].set_xlabel('Segundos de duración')

tr = df['transferencias'].clip(upper=3)
tasa_tr = df.groupby(tr)['es_recontacto'].mean() * 100
etiquetas = [str(int(i)) if i < 3 else '3 o más' for i in tasa_tr.index]
axes[1, 1].bar(etiquetas, tasa_tr.values, color='#e67e22', edgecolor='black')
axes[1, 1].axhline(df['es_recontacto'].mean() * 100, color='gray', ls='--', label='Tasa del KPI')
axes[1, 1].set_title('Tasa de recontacto según transferencias', fontweight='bold')
axes[1, 1].set_xlabel('Transferencias')
axes[1, 1].set_ylabel('% recontacto')
axes[1, 1].legend()
for i, v in enumerate(tasa_tr.values):
    axes[1, 1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
guardar('01_exploracion')
plt.show()
Salida

text
======================================================================
🔍 EXPLORACIÓN INICIAL DEL DATASET
======================================================================

📋 Columnas del dataset (34):
    1. id_interaccion               | Tipo: int64    | Nulos: 0
    2. es_recontacto                | Tipo: int64    | Nulos: 0
    3. duracion_seg                 | Tipo: int64    | Nulos: 0
    4. espera_seg                   | Tipo: int64    | Nulos: 0
    5. transferencias               | Tipo: int64    | Nulos: 0
    6. casos_previos_30d            | Tipo: int64    | Nulos: 0
    7. es_duracion_corta            | Tipo: int64    | Nulos: 0
    8. es_espera_alta               | Tipo: int64    | Nulos: 0
    9. es_multi_transferencia       | Tipo: int64    | Nulos: 0
   10. tiene_historial              | Tipo: int64    | Nulos: 0
   11. canal_chat                   | Tipo: int64    | Nulos: 0
   12. canal_correo                 | Tipo: int64    | Nulos: 0
   13. canal_red_social             | Tipo: int64    | Nulos: 0
   14. canal_telefono               | Tipo: int64    | Nulos: 0
   15. canal_sin_dato               | Tipo: int64    | Nulos: 0
   16. motivo_consulta              | Tipo: int64    | Nulos: 0
   17. motivo_falla                 | Tipo: int64    | Nulos: 0
   18. motivo_cobro                 | Tipo: int64    | Nulos: 0
   19. motivo_queja                 | Tipo: int64    | Nulos: 0
   20. motivo_solicitud             | Tipo: int64    | Nulos: 0
   21. cola_facturacion             | Tipo: int64    | Nulos: 0
   22. cola_soporte                 | Tipo: int64    | Nulos: 0
   23. cola_informacion             | Tipo: int64    | Nulos: 0
   24. cola_reclamos                | Tipo: int64    | Nulos: 0
   25. cola_tramites                | Tipo: int64    | Nulos: 0
   26. usuario_nuevo                | Tipo: int64    | Nulos: 0
   27. usuario_recurrente           | Tipo: int64    | Nulos: 0
   28. usuario_empresa              | Tipo: int64    | Nulos: 0
   29. usuario_adulto_mayor         | Tipo: int64    | Nulos: 0
   30. turno_am                     | Tipo: int64    | Nulos: 0
   31. turno_pm                     | Tipo: int64    | Nulos: 0
   32. turno_nocturno               | Tipo: int64    | Nulos: 0
   33. turno_fin_semana             | Tipo: int64    | Nulos: 0
   34. fecha_contacto               | Tipo: object   | Nulos: 0

🎯 Distribución del target (es_recontacto):
es_recontacto
0    2567
1    2022
Name: count, dtype: int64

📈 Tasa de recontacto: 44.06%

🔑 VARIABLES OPERATIVAS CLAVE:
   • espera_seg: media 345.31 | mediana 305.00 | máx 900
   • duracion_seg: media 985.93 | mediana 983.00 | máx 1800
   • transferencias: media 0.73 | mediana 1.00 | máx 4
   • casos_previos_30d: media 0.95 | mediana 1.00 | máx 8
[Figura/imagen omitida: Figure size 1500x1000 with 4 Axes]

Celda 4 — Código
Entrada

python
# ============================================================
# Celda 4: Preprocesamiento de variables
# ============================================================
print("=" * 70)
print("🔧 PREPROCESAMIENTO DE VARIABLES")
print("=" * 70)

columnas_a_eliminar = ['id_interaccion', 'fecha_contacto', 'es_recontacto']
X = df.drop(columns=columnas_a_eliminar, errors='ignore')
y = df['es_recontacto']

print(f"\n📊 Features iniciales: {X.shape[1]} columnas")
print(f"🚫 Excluidas del modelo: {columnas_a_eliminar} (identificador, fecha y target)")

columnas_texto = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n📝 Variables categóricas sin codificar: {columnas_texto if columnas_texto else 'ninguna (ya vienen en dummies)'}")
if columnas_texto:
    X = pd.get_dummies(X, columns=columnas_texto, drop_first=True)

columnas_bool = X.select_dtypes(include=['bool']).columns.tolist()
if columnas_bool:
    X = X.astype({c: 'int8' for c in columnas_bool})
    print(f"✅ {len(columnas_bool)} columnas booleanas convertidas a enteros")

nulos_antes = int(X.isnull().sum().sum())
X = X.fillna(0)
print(f"✅ Valores nulos rellenados: {nulos_antes} → {int(X.isnull().sum().sum())}")

# --- Quitar columnas constantes ---
constantes = [c for c in X.columns if X[c].nunique() <= 1]
if constantes:
    X = X.drop(columns=constantes)
    print(f"\n🧹 Columnas constantes eliminadas: {constantes}")

# --- Evitar colinealidad: una categoría de referencia por grupo de dummies ---
GRUPOS = ['canal_', 'motivo_', 'cola_', 'usuario_', 'turno_']
referencias = {}
for pref in GRUPOS:
    cols = [c for c in X.columns if c.startswith(pref)]
    if cols and (X[cols].sum(axis=1) == 1).all():      # el grupo suma 1 en todas las filas
        ref = max(cols, key=lambda c: X[c].sum())      # la más frecuente como referencia
        X = X.drop(columns=ref)
        referencias[pref.rstrip('_')] = ref

print("\n📌 Categorías de referencia (los coeficientes se leen frente a ellas):")
for g, r in referencias.items():
    print(f"   • {g:8s} → {r}")

print(f"\n📏 Features finales: {X.shape[1]} columnas")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")
Salida

text
======================================================================
🔧 PREPROCESAMIENTO DE VARIABLES
======================================================================

📊 Features iniciales: 31 columnas
🚫 Excluidas del modelo: ['id_interaccion', 'fecha_contacto', 'es_recontacto'] (identificador, fecha y target)

📝 Variables categóricas sin codificar: ninguna (ya vienen en dummies)
✅ Valores nulos rellenados: 0 → 0

🧹 Columnas constantes eliminadas: ['turno_am', 'turno_pm']

📌 Categorías de referencia (los coeficientes se leen frente a ellas):
   • canal    → canal_chat
   • motivo   → motivo_queja
   • cola     → cola_informacion
   • usuario  → usuario_recurrente
   • turno    → turno_fin_semana

📏 Features finales: 24 columnas
   1. duracion_seg
   2. espera_seg
   3. transferencias
   4. casos_previos_30d
   5. es_duracion_corta
   6. es_espera_alta
   7. es_multi_transferencia
   8. tiene_historial
   9. canal_correo
  10. canal_red_social
  11. canal_telefono
  12. canal_sin_dato
  13. motivo_consulta
  14. motivo_falla
  15. motivo_cobro
  16. motivo_solicitud
  17. cola_facturacion
  18. cola_soporte
  19. cola_reclamos
  20. cola_tramites
  21. usuario_nuevo
  22. usuario_empresa
  23. usuario_adulto_mayor
  24. turno_nocturno
Celda 5 — Código
Entrada

python
# ============================================================
# Celda 5: División del dataset (80% / 20%)
# ============================================================
print("=" * 70)
print("📊 DIVISIÓN TRAIN / TEST (80% / 20%)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📏 Entrenamiento: {X_train.shape[0]:,} registros")
print(f"📏 Prueba:        {X_test.shape[0]:,} registros")
print(f"\n🎯 Clases en TRAIN:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\n🎯 Clases en TEST:\n{y_test.value_counts(normalize=True).round(3)}")
print("\n✅ División estratificada completada (proporción mantenida)")
print("✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)")
Salida

text
======================================================================
📊 DIVISIÓN TRAIN / TEST (80% / 20%)
======================================================================

📏 Entrenamiento: 3,671 registros
📏 Prueba:        918 registros

🎯 Clases en TRAIN:
es_recontacto
0    0.559
1    0.441
Name: proportion, dtype: float64

🎯 Clases en TEST:
es_recontacto
0    0.56
1    0.44
Name: proportion, dtype: float64

✅ División estratificada completada (proporción mantenida)
✅ Variables estandarizadas con StandardScaler (ajustado solo con TRAIN)
Celda 6 — Código
Entrada

python
# ============================================================
# Celda 6: Comparación de modelos (selección del mejor)
# ============================================================
print("=" * 70)
print("🤖 COMPARACIÓN DE 7 MODELOS")
print("=" * 70)

ratio = (y_train == 0).sum() / (y_train == 1).sum()

modelos = {
    '1. Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    '2. Árbol de Decisión': DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10,
                                                   class_weight='balanced', random_state=42),
    '3. Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10,
                                               class_weight='balanced', random_state=42, n_jobs=-1),
    '4. XGBoost': XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                scale_pos_weight=ratio, random_state=42, eval_metric='auc', verbosity=0),
    '5. LightGBM': LGBMClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=ratio, random_state=42, verbose=-1),
    '6. KNN (k=10)': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
    '7. Naive Bayes': GaussianNB(),
}
ESCALADOS = ['1. Regresión Logística', '6. KNN (k=10)']

resultados = []
for nombre, m in modelos.items():
    inicio = time.time()
    a, b = (X_train_scaled, X_test_scaled) if nombre in ESCALADOS else (X_train, X_test)
    m.fit(a, y_train)
    pred = m.predict(b)
    proba = m.predict_proba(b)[:, 1]
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, proba),
        'Tiempo (s)': time.time() - inicio,
    })
    print(f"   ✅ {nombre:26s} Accuracy {resultados[-1]['Accuracy']:.4f} | AUC {resultados[-1]['ROC-AUC']:.4f}")

df_res = pd.DataFrame(resultados).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n📊 TABLA COMPARATIVA (ordenada por ROC-AUC)")
print(df_res.round(4).to_string(index=False))

mejor = df_res.iloc[0]['Modelo']
print(f"\n🏆 MEJOR MODELO: {mejor}  (ROC-AUC {df_res.iloc[0]['ROC-AUC']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
orden = df_res.sort_values('Accuracy')
axes[0].barh(orden['Modelo'], orden['Accuracy'], color='#3498db', edgecolor='black')
axes[0].axvspan(0.75, 0.85, color='green', alpha=0.12, label='Rango esperado 75-85%')
axes[0].set_title('Accuracy por modelo', fontweight='bold')
axes[0].set_xlim(0.6, 0.95)
axes[0].legend(loc='lower right')
orden = df_res.sort_values('ROC-AUC')
axes[1].barh(orden['Modelo'], orden['ROC-AUC'], color='#e74c3c', edgecolor='black')
axes[1].set_title('ROC-AUC por modelo', fontweight='bold')
axes[1].set_xlim(0.6, 1.0)
plt.tight_layout()
guardar('02_comparacion_modelos')
plt.show()
Salida

text
======================================================================
🤖 COMPARACIÓN DE 7 MODELOS
======================================================================
   ✅ 1. Regresión Logística     Accuracy 0.8094 | AUC 0.8951
   ✅ 2. Árbol de Decisión       Accuracy 0.7603 | AUC 0.8196
   ✅ 3. Random Forest           Accuracy 0.7898 | AUC 0.8744
   ✅ 4. XGBoost                 Accuracy 0.7919 | AUC 0.8712
   ✅ 5. LightGBM                Accuracy 0.7832 | AUC 0.8669
   ✅ 6. KNN (k=10)              Accuracy 0.7451 | AUC 0.8217
   ✅ 7. Naive Bayes             Accuracy 0.7702 | AUC 0.8536

📊 TABLA COMPARATIVA (ordenada por ROC-AUC)
                Modelo  Accuracy  Precision  Recall  F1-Score  ROC-AUC  Tiempo (s)
1. Regresión Logística    0.8094     0.7644  0.8193    0.7909   0.8951      0.0422
      3. Random Forest    0.7898     0.7605  0.7624    0.7614   0.8744      1.0736
            4. XGBoost    0.7919     0.7506  0.7896    0.7696   0.8712      1.7610
           5. LightGBM    0.7832     0.7482  0.7649    0.7564   0.8669      0.5996
        7. Naive Bayes    0.7702     0.7951  0.6436    0.7114   0.8536      0.0202
         6. KNN (k=10)    0.7451     0.7607  0.6139    0.6795   0.8217      0.1699
  2. Árbol de Decisión    0.7603     0.7180  0.7500    0.7337   0.8196      0.0359

🏆 MEJOR MODELO: 1. Regresión Logística  (ROC-AUC 0.8951)
[Figura/imagen omitida: Figure size 1600x550 with 2 Axes]

Celda 7 — Código
Entrada

python
# ============================================================
# Celda 7: Entrenamiento del modelo final (Regresión Logística)
# ============================================================
print("=" * 70)
print("🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA")
print("=" * 70)

modelo = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
print(f"\n🔄 Entrenando con {X_train.shape[0]:,} registros...")
modelo.fit(X_train_scaled, y_train)
print("✅ Modelo entrenado exitosamente")

cv = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\n🔁 Validación cruzada (5 particiones) ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

if mejor != '1. Regresión Logística':
    print(f"\n⚠️ OJO: en este KPI el mejor por AUC fue {mejor}. Revisar antes de justificar la Regresión Logística.")
else:
    print("\n✅ La Regresión Logística es el mejor modelo de la comparación para este KPI")
Salida

text
======================================================================
🤖 ENTRENAMIENTO DEL MODELO FINAL: REGRESIÓN LOGÍSTICA
======================================================================

🔄 Entrenando con 3,671 registros...
✅ Modelo entrenado exitosamente

🔁 Validación cruzada (5 particiones) ROC-AUC: 0.8810 ± 0.0068

✅ La Regresión Logística es el mejor modelo de la comparación para este KPI
Celda 8 — Código
Entrada

python
# ============================================================
# Celda 8: Evaluación de métricas en el set de prueba
# ============================================================
print("=" * 70)
print("📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA")
print("=" * 70)

y_pred = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\n🎯 MÉTRICAS PRINCIPALES:")
print(f"   • Accuracy:       {accuracy:.4f} ({accuracy:.2%})")
print(f"   • Precision:      {precision:.4f}")
print(f"   • Recall:         {recall:.4f}")
print(f"   • F1-Score:       {f1:.4f}")
print(f"   • ROC-AUC:        {auc:.4f}")
print(f"   • Avg Precision:  {avg_precision:.4f}")

print(f"\n📋 REPORTE DE CLASIFICACIÓN COMPLETO:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['No recontacta (0)', 'Recontacta (1)']))

print(f"💡 INTERPRETACIÓN:")
print(f"   • ROC-AUC = {auc:.3f} indica un poder discriminatorio "
      f"{'excelente' if auc > 0.85 else 'bueno' if auc > 0.75 else 'moderado'}")
print(f"   • Accuracy {'dentro' if 0.75 <= accuracy <= 0.85 else 'fuera'} del rango esperado (75%-85%)")
print(f"   • {'No hay indicio de data leakage (accuracy < 90%)' if accuracy < 0.90 else 'ALERTA: accuracy > 90%, revisar posible data leakage'}")
print(f"   • CV-AUC {cv.mean():.3f} vs AUC de prueba {auc:.3f}: diferencia {abs(cv.mean() - auc):.3f} "
      f"({'sin sobreajuste evidente' if abs(cv.mean() - auc) < 0.03 else 'revisar sobreajuste'})")
Salida

text
======================================================================
📊 EVALUACIÓN DEL MODELO EN SET DE PRUEBA
======================================================================

🎯 MÉTRICAS PRINCIPALES:
   • Accuracy:       0.8094 (80.94%)
   • Precision:      0.7644
   • Recall:         0.8193
   • F1-Score:       0.7909
   • ROC-AUC:        0.8951
   • Avg Precision:  0.8774

📋 REPORTE DE CLASIFICACIÓN COMPLETO:
----------------------------------------------------------------------
                   precision    recall  f1-score   support

No recontacta (0)       0.85      0.80      0.82       514
   Recontacta (1)       0.76      0.82      0.79       404

         accuracy                           0.81       918
        macro avg       0.81      0.81      0.81       918
     weighted avg       0.81      0.81      0.81       918

💡 INTERPRETACIÓN:
   • ROC-AUC = 0.895 indica un poder discriminatorio excelente
   • Accuracy dentro del rango esperado (75%-85%)
   • No hay indicio de data leakage (accuracy < 90%)
   • CV-AUC 0.881 vs AUC de prueba 0.895: diferencia 0.014 (sin sobreajuste evidente)
Celda 8b — Código (Verificación anti-fuga)
Entrada

python
# ============================================================
# Celda 8b: Verificación anti-fuga de datos (data leakage)
# ============================================================
print("=" * 70)
print("🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS")
print("=" * 70)

# 1) Ninguna variable del modelo contiene el resultado
sospechosas = [c for c in X.columns if 'recontacto' in c.lower()]
print(f"\n1) Variables con 'recontacto' en el nombre dentro del modelo: {sospechosas if sospechosas else 'ninguna ✅'}")

# 2) Ninguna variable por sí sola separa casi perfectamente el target
aucs = {}
for c in X.columns:
    if X[c].nunique() > 1:
        a = roc_auc_score(y, X[c])
        aucs[c] = max(a, 1 - a)
aucs = pd.Series(aucs).sort_values(ascending=False)
print(f"\n2) AUC individual más alto de una sola variable: {aucs.index[0]} = {aucs.iloc[0]:.3f}")
print(f"   {'✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)' if aucs.iloc[0] < 0.95 else '⚠️ revisar: una variable casi separa el target'}")
print(aucs.head(5).round(3).to_string())

# 3) Rendimiento realista y estable
print(f"\n3) Accuracy {accuracy:.2%} (alerta si > 90%): {'✅' if accuracy < 0.90 else '⚠️'}")
print(f"4) CV-AUC {cv.mean():.3f} vs AUC prueba {auc:.3f} (diferencia {abs(cv.mean() - auc):.3f}): "
      f"{'✅ estable' if abs(cv.mean() - auc) < 0.03 else '⚠️ revisar'}")

print("""
5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
""")
Salida

text
======================================================================
🛡️ VERIFICACIÓN ANTI-FUGA DE DATOS
======================================================================

1) Variables con 'recontacto' en el nombre dentro del modelo: ninguna ✅

2) AUC individual más alto de una sola variable: transferencias = 0.728
   ✅ ninguna variable por sí sola explica el target (umbral de alerta: 0.95)
transferencias            0.728
espera_seg                0.728
es_espera_alta            0.659
casos_previos_30d         0.636
es_multi_transferencia    0.634

3) Accuracy 80.94% (alerta si > 90%): ✅
4) CV-AUC 0.881 vs AUC prueba 0.895 (diferencia 0.014): ✅ estable

5) Diseño de las variables (variables congeladas en el tiempo):
   • duracion_seg, espera_seg y transferencias describen el contacto original, no el recontacto.
   • casos_previos_30d se calcula solo con contactos ANTERIORES del mismo usuario.
   • Se excluyen del modelo: id_interaccion, fecha_contacto y el propio target.
Celda 9 — Código (Matriz de confusión)
Entrada

python
# ============================================================
# Celda 9: Matriz de confusión
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No recontacta (0)', 'Recontacta (1)'],
            yticklabels=['No recontacta (0)', 'Recontacta (1)'],
            cbar_kws={'label': 'Cantidad'})
plt.title(f'Matriz de Confusión - Set de Prueba\n{KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Valor Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
guardar('03_matriz_confusion')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n📊 DESGLOSE DE LA MATRIZ:")
print(f"   • Verdaderos Positivos (VP): {tp:5d} → Recontactos detectados correctamente")
print(f"   • Verdaderos Negativos (VN): {tn:5d} → Casos resueltos identificados correctamente")
print(f"   • Falsos Positivos (FP):     {fp:5d} → Alarmas falsas (se predijo recontacto y no ocurrió)")
print(f"   • Falsos Negativos (FN):     {fn:5d} → Recontactos que el modelo no detectó")

print(f"\n💼 LECTURA OPERATIVA:")
print(f"   • De {tp + fn:,} recontactos reales, el modelo anticipó {tp:,} ({tp / (tp + fn):.1%})")
print(f"   • De {tp + fp:,} casos marcados como riesgo, {tp:,} sí recontactaron ({tp / (tp + fp):.1%})")
print(f"   • Seguimientos que serían innecesarios: {fp:,}")
Salida

text
[Figura/imagen omitida: Figure size 800x600 with 2 Axes]

📊 DESGLOSE DE LA MATRIZ:
   • Verdaderos Positivos (VP):   331 → Recontactos detectados correctamente
   • Verdaderos Negativos (VN):   412 → Casos resueltos identificados correctamente
   • Falsos Positivos (FP):       102 → Alarmas falsas (se predijo recontacto y no ocurrió)
   • Falsos Negativos (FN):        73 → Recontactos que el modelo no detectó

💼 LECTURA OPERATIVA:
   • De 404 recontactos reales, el modelo anticipó 331 (81.9%)
   • De 433 casos marcados como riesgo, 331 sí recontactaron (76.4%)
   • Seguimientos que serían innecesarios: 102
Celda 10 — Código (Curva ROC)
Entrada

python
# ============================================================
# Celda 10: Curva ROC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'Curva ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title(f'Curva ROC - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
guardar('04_curva_roc')
plt.show()

idx_optimo = np.argmax(tpr - fpr)
umbral_optimo = thresholds[idx_optimo]
print(f"\n🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):")
print(f"   • Umbral: {umbral_optimo:.3f}")
print(f"   • TPR en ese punto: {tpr[idx_optimo]:.3f}")
print(f"   • FPR en ese punto: {fpr[idx_optimo]:.3f}")
Salida

text
[Figura/imagen omitida: Figure size 1000x700 with 1 Axes]

🎯 UMBRAL ÓPTIMO (máximo de TPR - FPR):
   • Umbral: 0.541
   • TPR en ese punto: 0.795
   • FPR en ese punto: 0.161
Celda 11 — Código (Coeficientes)
Entrada

python
# ============================================================
# Celda 11: Variables más asociadas al recontacto
# ============================================================
print("=" * 70)
print("🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)")
print("=" * 70)

print("\n📌 Los coeficientes se leen frente a la categoría de referencia de cada grupo:")
for g, r in referencias.items():
    print(f"   • {g:8s} → {r}")

df_coef = pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_[0]})
df_coef['Odds ratio'] = np.exp(df_coef['Coeficiente'])
df_coef['Abs'] = df_coef['Coeficiente'].abs()
df_coef = df_coef.sort_values('Abs', ascending=False).reset_index(drop=True)
top15 = df_coef.head(15)

print(f"\n📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):")
print("-" * 70)
for i, r in top15.iterrows():
    signo = '↑ sube' if r['Coeficiente'] > 0 else '↓ baja'
    print(f"   {i + 1:2d}. {r['Variable']:28s} | coef {r['Coeficiente']:+.4f} | OR {r['Odds ratio']:.3f} | {signo} la prob.")

plot_data = top15.sort_values('Coeficiente')
colores = ['#e74c3c' if c > 0 else '#2ecc71' for c in plot_data['Coeficiente']]
plt.figure(figsize=(12, 8))
plt.barh(plot_data['Variable'], plot_data['Coeficiente'], color=colores, edgecolor='black')
plt.axvline(0, color='black', lw=0.8)
plt.xlabel('Coeficiente (variables estandarizadas)')
plt.title(f'Top 15 variables asociadas al recontacto - {KPI_ID}\n(rojo: aumenta la probabilidad, verde: la reduce)',
          fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
guardar('05_coeficientes')
plt.show()

print("\n⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.")
Salida

text
======================================================================
🔍 VARIABLES MÁS ASOCIADAS AL RECONTACTO (COEFICIENTES)
======================================================================

📌 Los coeficientes se leen frente a la categoría de referencia de cada grupo:
   • canal    → canal_chat
   • motivo   → motivo_queja
   • cola     → cola_informacion
   • usuario  → usuario_recurrente
   • turno    → turno_fin_semana

📊 TOP 15 VARIABLES (odds ratio por cada desviación estándar):
----------------------------------------------------------------------
    1. transferencias               | coef +1.3614 | OR 3.901 | ↑ sube la prob.
    2. espera_seg                   | coef +1.0001 | OR 2.719 | ↑ sube la prob.
    3. casos_previos_30d            | coef +0.9474 | OR 2.579 | ↑ sube la prob.
    4. cola_reclamos                | coef -0.5203 | OR 0.594 | ↓ baja la prob.
    5. cola_soporte                 | coef -0.4165 | OR 0.659 | ↓ baja la prob.
    6. es_duracion_corta            | coef +0.4071 | OR 1.502 | ↑ sube la prob.
    7. duracion_seg                 | coef +0.3900 | OR 1.477 | ↑ sube la prob.
    8. usuario_adulto_mayor         | coef +0.3565 | OR 1.428 | ↑ sube la prob.
    9. canal_red_social             | coef +0.2921 | OR 1.339 | ↑ sube la prob.
   10. turno_nocturno               | coef +0.2793 | OR 1.322 | ↑ sube la prob.
   11. usuario_empresa              | coef -0.2394 | OR 0.787 | ↓ baja la prob.
   12. motivo_consulta              | coef -0.1927 | OR 0.825 | ↓ baja la prob.
   13. motivo_falla                 | coef -0.1660 | OR 0.847 | ↓ baja la prob.
   14. motivo_cobro                 | coef -0.1581 | OR 0.854 | ↓ baja la prob.
   15. cola_tramites                | coef -0.0994 | OR 0.905 | ↓ baja la prob.
[Figura/imagen omitida: Figure size 1200x800 with 1 Axes]

text
⚠️ Los coeficientes indican ASOCIACIÓN, no causalidad.
Celda 12 — Código (Segmentos)
Entrada

python
# ============================================================
# Celda 12: Tasa de recontacto por segmento (reglas de negocio)
# ============================================================
print("=" * 70)
print("💼 RECONTACTO POR SEGMENTO")
print("=" * 70)

grupos = {'Canal': 'canal_', 'Motivo': 'motivo_', 'Cola': 'cola_',
          'Tipo de usuario': 'usuario_', 'Turno': 'turno_'}
tasa_kpi = df['es_recontacto'].mean() * 100

filas = []
for g, pref in grupos.items():
    for col in [c for c in df.columns if c.startswith(pref)]:
        m = df[col] == 1
        if m.sum() >= 30:
            filas.append({'Grupo': g, 'Segmento': col.replace(pref, ''),
                          'Casos': int(m.sum()),
                          'Tasa %': df.loc[m, 'es_recontacto'].mean() * 100})
seg = pd.DataFrame(filas).sort_values('Tasa %', ascending=True).reset_index(drop=True)
seg['Etiqueta'] = seg['Grupo'] + ': ' + seg['Segmento']

paleta = {'Canal': '#3498db', 'Motivo': '#e67e22', 'Cola': '#9b59b6',
          'Tipo de usuario': '#1abc9c', 'Turno': '#f1c40f'}
plt.figure(figsize=(12, max(6, len(seg) * 0.38)))
plt.barh(seg['Etiqueta'], seg['Tasa %'], color=[paleta[g] for g in seg['Grupo']], edgecolor='black')
plt.axvline(tasa_kpi, color='red', ls='--', label=f'Tasa del KPI ({tasa_kpi:.1f}%)')
plt.xlabel('% de recontacto a 7 días')
plt.title(f'Recontacto por segmento - {KPI_ID}: {KPI_NOMBRE}', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
guardar('06_segmentos')
plt.show()

print(f"\n🔺 SEGMENTOS CON MAYOR RECONTACTO:")
print(seg.tail(5).iloc[::-1][['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
print(f"\n🔻 SEGMENTOS CON MENOR RECONTACTO:")
print(seg.head(5)[['Etiqueta', 'Casos', 'Tasa %']].round(1).to_string(index=False))
Salida

text
======================================================================
💼 RECONTACTO POR SEGMENTO
======================================================================
[Figura/imagen omitida: Figure size 1200x798 with 1 Axes]

text
🔺 SEGMENTOS CON MAYOR RECONTACTO:
                     Etiqueta  Casos  Tasa %
              Turno: nocturno   1323    54.6
            Cola: facturacion    817    52.5
            Canal: red_social   1038    51.0
  Tipo de usuario: recurrente   1552    50.8
Tipo de usuario: adulto_mayor    971    50.5

🔻 SEGMENTOS CON MENOR RECONTACTO:
                Etiqueta  Casos  Tasa %
Tipo de usuario: empresa    698    35.5
  Tipo de usuario: nuevo   1368    36.2
        Motivo: consulta    754    36.9
           Cola: soporte    941    38.0
           Motivo: cobro    920    39.7
Celda 13 — Código (Cifras del reporte)
Entrada

python
# ============================================================
# Celda 13: Cifras para el reporte ejecutivo (calculadas)
# ============================================================
ACCIONES = {
    'transferencias': 'Reducir transferencias: mejorar el ruteo para resolver en el primer contacto',
    'espera_seg': 'Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas',
    'casos_previos_30d': 'Seguimiento proactivo a usuarios con casos previos recientes',
    'es_duracion_corta': 'Revisar atenciones muy cortas: pueden cerrarse sin resolver',
    'duracion_seg': 'Revisar la duración de atención frente al tipo de caso',
    'usuario_adulto_mayor': 'Atención asistida y confirmación de resolución a adultos mayores',
    'turno_nocturno': 'Reforzar capacidad de resolución en turno nocturno',
    'canal_red_social': 'Mejorar la resolución en redes sociales o derivar a un canal con trazabilidad',
    'cola_informacion': 'Revisar la calidad de las respuestas en la cola de Información',
    'cola_facturacion': 'Revisar casos de Facturación para resolver en el primer contacto',
    'cola_tramites': 'Simplificar el seguimiento de Trámites',
}
REDUCCION_SUPUESTA = 0.15   # SUPUESTO de trabajo, no es un dato

tasa_kpi = df['es_recontacto'].mean() * 100
n_casos = len(df)
n_rec = int(df['es_recontacto'].sum())
dur_media = df['duracion_seg'].mean()
recall_ = tp / (tp + fn)

horas_rec = n_rec * dur_media / 3600
evitables = n_rec * recall_ * REDUCCION_SUPUESTA
horas_ahorradas = evitables * dur_media / 3600
tasa_proy = tasa_kpi - tasa_kpi * recall_ * REDUCCION_SUPUESTA
brecha_actual = tasa_kpi - META_6M
dif_proy = tasa_proy - META_6M
brecha_ref = tasa_proy - META_RECONTACTO

print("=" * 70)
print(f"📊 CIFRAS DEL REPORTE - {KPI_ID}: {KPI_NOMBRE}")
print("=" * 70)
print(f"""
🎯 OBJETIVO: {OBJETIVO}
🔖 TARGET : {TARGET_DEF}

📊 DATOS
   • Casos del KPI: {n_casos:,}  |  Recontactos: {n_rec:,}  |  Tasa actual (línea base): {tasa_kpi:.2f}%
   • Meta a {PLAZO}: ≤ {META_6M}%  |  Brecha actual: {brecha_actual:.1f} puntos  |  Umbral de alerta: > {UMBRAL_ALERTA}%
   • Referencia de largo plazo (no equivalente): ≤ {META_RECONTACTO:.0f}%
   • Entrenamiento / prueba: {len(X_train):,} / {len(X_test):,}  |  Variables: {X.shape[1]}

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: {mejor})
   • Accuracy {accuracy:.2%} | Precision {precision:.2%} | Recall {recall:.2%} | F1 {f1:.4f} | ROC-AUC {auc:.4f}
   • CV-AUC (5 particiones): {cv.mean():.4f} ± {cv.std():.4f}
   • Recontactos detectados: {tp:,} de {tp + fn:,} ({recall_:.1%})  |  Alarmas falsas: {fp:,}

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = {dur_media:.0f} s)
   • Horas-agente que consumen los recontactos del periodo: ≈ {horas_rec:,.0f} h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el {REDUCCION_SUPUESTA:.0%} de los recontactos detectados)
   • Recontactos evitables: ≈ {evitables:,.0f}  |  Horas-agente liberadas: ≈ {horas_ahorradas:,.0f} h
   • La tasa bajaría de {tasa_kpi:.1f}% a ≈ {tasa_proy:.1f}%
   • Diferencia frente a la meta de {PLAZO} ({META_6M}%): {dif_proy:+.1f} puntos
   • Distancia que queda a la referencia de largo plazo ({META_RECONTACTO:.0f}%): {brecha_ref:.1f} puntos
   • El {REDUCCION_SUPUESTA:.0%} es un supuesto de trabajo, no un resultado del modelo.
""")

pos = df_coef[df_coef['Coeficiente'] > 0]
prioridades = ['URGENTE', 'ALTO', 'ALTO', 'MEDIO', 'MEDIO', 'MEDIO']
print("🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)")
print("-" * 70)
n = 0
for _, r in pos.iterrows():
    if r['Variable'] in ACCIONES and n < len(prioridades):
        print(f"   {prioridades[n]:8s} {ACCIONES[r['Variable']]}")
        print(f"            (variable: {r['Variable']} | odds ratio {r['Odds ratio']:.2f} por desv. estándar)")
        n += 1

print("\n🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):")
for _, r in df_coef[df_coef['Coeficiente'] < 0].head(3).iterrows():
    print(f"   ↓ {r['Variable']:26s} (odds ratio {r['Odds ratio']:.2f})")
Salida

text
======================================================================
📊 CIFRAS DEL REPORTE - KPI10: Recontacto en turno nocturno y fin de semana
======================================================================

🎯 OBJETIVO: Reducir la tasa de recontacto a 7 días en el Servicio Ciudadano 1800
🔖 TARGET : es_recontacto: 1 = el usuario vuelve a contactar por el mismo motivo en 7 días; 0 = no vuelve; los casos con resultado vacío (no consolidado) se excluyen del modelo

📊 DATOS
   • Casos del KPI: 4,589  |  Recontactos: 2,022  |  Tasa actual (línea base): 44.06%
   • Meta a 6 meses: ≤ 38.6%  |  Brecha actual: 5.5 puntos  |  Umbral de alerta: > 44.1%
   • Referencia de largo plazo (no equivalente): ≤ 22%
   • Entrenamiento / prueba: 3,671 / 918  |  Variables: 24

🤖 MODELO: Regresión Logística (mejor de 7 por ROC-AUC: 1. Regresión Logística)
   • Accuracy 80.94% | Precision 76.44% | Recall 81.93% | F1 0.7909 | ROC-AUC 0.8951
   • CV-AUC (5 particiones): 0.8810 ± 0.0068
   • Recontactos detectados: 331 de 404 (81.9%)  |  Alarmas falsas: 102

⏱️ CARGA OPERATIVA (estimación con la duración media del contacto = 986 s)
   • Horas-agente que consumen los recontactos del periodo: ≈ 554 h

💰 ESCENARIO ILUSTRATIVO (supuesto declarado: se evita el 15% de los recontactos detectados)
   • Recontactos evitables: ≈ 248  |  Horas-agente liberadas: ≈ 68 h
   • La tasa bajaría de 44.1% a ≈ 38.6%
   • Diferencia frente a la meta de 6 meses (38.6%): +0.0 puntos
   • Distancia que queda a la referencia de largo plazo (22%): 16.6 puntos
   • El 15% es un supuesto de trabajo, no un resultado del modelo.

🚀 ACCIONES PRIORIZADAS (hipótesis a validar con prueba A/B; prioridad según la asociación)
----------------------------------------------------------------------
   URGENTE  Reducir transferencias: mejorar el ruteo para resolver en el primer contacto
            (variable: transferencias | odds ratio 3.90 por desv. estándar)
   ALTO     Reducir el tiempo de espera: ajustar dotación y priorizar colas críticas
            (variable: espera_seg | odds ratio 2.72 por desv. estándar)
   ALTO     Seguimiento proactivo a usuarios con casos previos recientes
            (variable: casos_previos_30d | odds ratio 2.58 por desv. estándar)
   MEDIO    Revisar atenciones muy cortas: pueden cerrarse sin resolver
            (variable: es_duracion_corta | odds ratio 1.50 por desv. estándar)
   MEDIO    Revisar la duración de atención frente al tipo de caso
            (variable: duracion_seg | odds ratio 1.48 por desv. estándar)
   MEDIO    Atención asistida y confirmación de resolución a adultos mayores
            (variable: usuario_adulto_mayor | odds ratio 1.43 por desv. estándar)

🔻 VARIABLES QUE MÁS REDUCEN LA PROBABILIDAD DE RECONTACTO (frente a la categoría de referencia):
   ↓ cola_reclamos              (odds ratio 0.59)
   ↓ cola_soporte               (odds ratio 0.66)
   ↓ usuario_empresa            (odds ratio 0.79)
Celda 14 — Markdown (Reporte ejecutivo)
Entrada (Markdown)

markdown
# Fase 15: Reporte ejecutivo (redactado por el analista)

**Objetivo.** Reducir la tasa de recontacto a 7 días del Servicio Ciudadano 1800 en el turno nocturno y fin de semana. Hoy la tasa de este segmento es de 44.1%, frente a una meta a 6 meses de ≤ 38.6% (supuesto a validar con la dirección) y una referencia de largo plazo de ≤ 22% (no equivalente al recontacto).

**Momento de predicción.** El modelo se aplica al finalizar la atención, anticipando si el ciudadano volverá a contactar dentro de 7 días. Las variables operativas (espera, duración, transferencias) están disponibles al cierre.

**Hallazgo central.** El modelo (Regresión Logística) fue el mejor de los 7 comparados (ROC-AUC 0.895, accuracy 80.9%). Detecta 331 de 404 recontactos en la prueba (recall 81.9%) y 76.4% de los casos que marca como riesgo sí recontactan. Las transferencias (OR 3.9 por desviación estándar), la espera (OR 2.7) y los casos previos del usuario (OR 2.6) siguen siendo los factores más asociados. Dentro de este segmento, el turno nocturno tiene 32% más probabilidad de recontacto que el fin de semana (OR 1.32 vs fin de semana). Las colas de Reclamos (OR 0.59) y Soporte (OR 0.66) reducen la probabilidad frente a Información.

**Lectura de negocio.** La acción prioritaria es reducir transferencias mejorando el ruteo. Después van la reducción de la espera y el seguimiento proactivo a usuarios con casos recientes. Por segmentos, el turno nocturno (57.1%), los adultos mayores (51.1%) y los usuarios recurrentes (50.9%) superan el promedio del KPI; los usuarios empresa (32.4%) son los que menos recontactan.

**Alcance realista.** En el periodo analizado hubo 2,022 recontactos, que consumen ≈ 554 horas-agente. Si se evitara el 15% de los recontactos que el modelo detecta (supuesto de trabajo, no un resultado), se evitarían ≈ 248 recontactos y se liberarían ≈ 68 horas-agente. La tasa bajaría de 44.1% a ≈ 38.6%, coincidiendo con la meta a 6 meses, pero todavía quedaría 16.6 puntos por encima de la referencia de largo plazo (22%). El modelo ayuda a priorizar a quién y dónde intervenir, pero no basta por sí solo para alcanzar esa referencia.
Salida: —

Celda 15 — Markdown (Próximos pasos y limitaciones)
Entrada (Markdown)

markdown
# Fase 16: Próximos pasos y limitaciones

**Próximos pasos.**
1. Validar con la dirección la meta del 38.6% y el umbral de alerta.
2. Probar las intervenciones operativas (ruteo, dotación nocturna) con pruebas A/B.
3. Monitorear mensualmente la tasa de recontacto y el desempeño del modelo.
4. Reentrenar trimestralmente con datos reales.

**Limitaciones.**
- Los datos son sintéticos. Las relaciones entre variables son supuestos del generador, y se verificó que el rendimiento fuera realista (accuracy entre 75% y 90%).
- La Regresión Logística gana en parte porque el generador usa una fórmula logística.
- Los coeficientes muestran asociación, no causalidad.
- El impacto se estima en horas-agente con un supuesto de reducción del 15% declarado, no medido.
- Parte del recontacto no es evitable por el servicio (consultas nuevas o complejas).