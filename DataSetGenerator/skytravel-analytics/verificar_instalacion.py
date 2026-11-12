# ============================================
# verificar_instalacion.py
# Verifica que TODAS las librerías estén instaladas
# ============================================

def verificar():
    print("=" * 60)
    print("🔍 VERIFICACIÓN DE INSTALACIÓN")
    print("=" * 60)

    errores = []

    # 1. Python
    import sys
    print(f"\n🐍 Python: {sys.version}")

    # 2. Pandas
    try:
        import pandas as pd
        print(f"✅ pandas: {pd.__version__}")
    except ImportError:
        errores.append("pandas")

    # 3. NumPy
    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")
    except ImportError:
        errores.append("numpy")

    # 4. Scikit-learn
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        errores.append("scikit-learn")

    # 5. XGBoost
    try:
        import xgboost as xgb
        print(f"✅ xgboost: {xgb.__version__}")
    except ImportError:
        errores.append("xgboost")

    # 6. Matplotlib
    try:
        import matplotlib
        print(f"✅ matplotlib: {matplotlib.__version__}")
    except ImportError:
        errores.append("matplotlib")

    # 7. Seaborn
    try:
        import seaborn as sns
        print(f"✅ seaborn: {sns.__version__}")
    except ImportError:
        errores.append("seaborn")

    # 8. pyodbc (CRÍTICO para SQL Server)
    try:
        import pyodbc
        print(f"✅ pyodbc: {pyodbc.version}")
        print(f"\n📋 Drivers ODBC disponibles:")
        drivers = pyodbc.drivers()
        if drivers:
            for d in drivers:
                print(f"   • {d}")
        else:
            print(f"   ⚠️ No se encontraron drivers ODBC!")
            print(f"   💡 Descarga desde: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")
            errores.append("ODBC Driver")
    except ImportError:
        errores.append("pyodbc")

    # 9. SQLAlchemy
    try:
        import sqlalchemy
        print(f"✅ sqlalchemy: {sqlalchemy.__version__}")
    except ImportError:
        errores.append("sqlalchemy")

    # 10. dotenv
    try:
        import dotenv
        print(f"✅ python-dotenv: instalado")
    except ImportError:
        errores.append("python-dotenv")

    # Resumen
    print("\n" + "=" * 60)
    if errores:
        print(f"❌ Faltan {len(errores)} dependencias: {', '.join(errores)}")
        print(f"💡 Ejecuta: pip install {' '.join(errores)}")
    else:
        print("🎉 ¡TODAS las dependencias están instaladas!")
    print("=" * 60)

import pyodbc
print("Drivers ODBC instalados en tu sistema:")
for driver in pyodbc.drivers():
    print(f"  ✅ {driver}")

if __name__ == "__main__":
    verificar()