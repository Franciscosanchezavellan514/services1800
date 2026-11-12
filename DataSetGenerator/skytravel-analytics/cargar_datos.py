# ============================================
# src/cargar_datos.py
# Carga de datasets desde SQL Server a pandas
# VERSIÓN SIMPLIFICADA - Sin exportación a CSV
# ============================================

import sys
import os
import pandas as pd

# Agregar la carpeta src al path para importar database.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.database import execute_query


def cargar_dataset_abandono():
    """
    Carga el dataset de abandono de reserva desde SQL Server.
    
    Retorna:
        pd.DataFrame con ~79,200 registros y ~25 variables
    """
    print("=" * 60)
    print("📊 CARGANDO DATASET DE ABANDONO")
    print("=" * 60)

    query = """
    SELECT *
    FROM dbo.dm_dataset_abandono
    WHERE es_abandono IS NOT NULL
    ORDER BY fecha_inicio_reserva;
    """

    df = execute_query(query)

    if df is not None:
        print(f"\n📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"\n🎯 Distribución de la variable target:")
        print(df['es_abandono'].value_counts())
        print(f"\n📈 Tasa de abandono: {df['es_abandono'].mean() * 100:.2f}%")
        
        return df

    return None

def cargar_dataset_abandono_correcto():
    """
    Carga el dataset de abandono de reserva desde SQL Server.
    
    Retorna:
        pd.DataFrame con ~79,200 registros y ~25 variables
    """
    print("=" * 60)
    print("📊 CARGANDO DATASET DE ABANDONO")
    print("=" * 60)

    query = """
    SELECT *
    FROM dbo.dm_dataset_abandono_correcto
    WHERE es_abandono IS NOT NULL
    ORDER BY fecha_inicio_reserva;
    """

    df = execute_query(query)

    if df is not None:
        print(f"\n📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"\n🎯 Distribución de la variable target:")
        print(df['es_abandono'].value_counts())
        print(f"\n📈 Tasa de abandono: {df['es_abandono'].mean() * 100:.2f}%")
        
        return df

    return None

def cargar_dataset_recompra():
    """
    Carga el dataset de recompra a 12 meses desde SQL Server.
    
    Retorna:
        pd.DataFrame con ~47,250 registros
    """
    print("=" * 60)
    print("📊 CARGANDO DATASET DE RECOMPRA")
    print("=" * 60)

    query = """
    SELECT *
    FROM dbo.dm_dataset_recompra
    WHERE recompra_12m IS NOT NULL;
    """

    df = execute_query(query)

    if df is not None:
        print(f"\n📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"\n🎯 Distribución de la variable target:")
        print(df['recompra_12m'].value_counts())
        print(f"\n📈 Tasa de recompra: {df['recompra_12m'].mean() * 100:.2f}%")
        
        return df

    return None


# ============================================
# CASO 15 - SERVICIO CIUDADANO 1800
# KPI 1: Recontacto a 7 días
# ============================================
def cargar_dataset_recontacto():
    """
    Carga el dataset del KPI 1 (recontacto a 7 días) desde SQL Server.
    
    Retorna:
        pd.DataFrame con ~11,436 registros
    """
    print("=" * 60)
    print("📊 CARGANDO DATASET DE RECONTACTO (KPI 1)")
    print("=" * 60)

    query = """
    SELECT *
    FROM dbo.dm_dataset_recontacto
    WHERE es_recontacto IS NOT NULL
    ORDER BY fecha_contacto;
    """

    df = execute_query(query)

    if df is not None:
        print(f"\n📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"\n🎯 Distribución del target:")
        print(df['es_recontacto'].value_counts())
        print(f"\n📈 Tasa de recontacto: {df['es_recontacto'].mean() * 100:.2f}%")
        
        return df

    return None


# ============================================
# EJECUCIÓN DIRECTA
# ============================================
if __name__ == "__main__":
    df_recontacto = cargar_dataset_recontacto()

    print("\n" + "=" * 60)
    print("✅ Datos cargados en memoria, listos para modelado")
    print("=" * 60)