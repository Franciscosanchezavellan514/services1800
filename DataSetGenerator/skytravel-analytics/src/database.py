# ============================================
# src/database.py
# SKYTRAVEL NICARAGUA - CONEXIÓN SQL SERVER 2022
# Reemplaza la versión anterior de PostgreSQL
# ============================================

import os
import pandas as pd
import pyodbc
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()


def get_connection_string():
    """
    Construye la cadena de conexión para SQL Server.
    Soporta autenticación SQL y autenticación de Windows.
    """
    server   = os.getenv('DB_SERVER', 'localhost')
    database = os.getenv('DB_NAME', 'SkyTravelDW')
    driver   = os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
    user     = os.getenv('DB_USER', '')
    password = os.getenv('DB_PASSWORD', '')
    trusted  = os.getenv('DB_TRUSTED', 'no')

    # Autenticación de Windows (sin usuario/password)
    if trusted.lower() == 'yes':
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
            f"TrustServerCertificate=yes;"
        )
    # Autenticación SQL Server (usuario + password)
    else:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )

    return conn_str


def get_connection():
    """
    Establece y retorna una conexión activa a SQL Server.
    """
    try:
        conn_str = get_connection_string()
        conn = pyodbc.connect(conn_str)
        print("✅ Conexión exitosa a SQL Server")
        return conn
    except pyodbc.Error as e:
        print(f"❌ Error de conexión a SQL Server:")
        print(f"   {e}")
        print(f"\n💡 Posibles soluciones:")
        print(f"   1. Verifica que SQL Server esté corriendo")
        print(f"   2. Revisa las credenciales en el archivo .env")
        print(f"   3. Asegúrate de que la base '{os.getenv('DB_NAME')}' exista")
        print(f"   4. Verifica el driver ODBC instalado")
        return None


def execute_query(query, params=None):
    """
    Ejecuta una consulta SELECT y retorna un DataFrame de pandas.

    Parámetros:
        query  (str): Consulta SQL
        params (tuple): Parámetros opcionales para evitar SQL injection

    Retorna:
        pd.DataFrame o None
    """
    conn = get_connection()
    if conn is None:
        return None

    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)

        print(f"✅ Consulta ejecutada: {len(df):,} registros, {len(df.columns)} columnas")
        return df

    except Exception as e:
        print(f"❌ Error en consulta SQL:")
        print(f"   {e}")
        return None

    finally:
        conn.close()


def execute_non_query(sql):
    """
    Ejecuta una sentencia que NO retorna datos
    (CREATE TABLE, INSERT, UPDATE, DELETE, DROP).

    Parámetros:
        sql (str): Sentencia SQL

    Retorna:
        bool: True si fue exitoso, False si hubo error
    """
    conn = get_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        print(f"✅ Sentencia ejecutada exitosamente")
        cursor.close()
        return True

    except Exception as e:
        print(f"❌ Error ejecutando sentencia:")
        print(f"   {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


def execute_script_file(script_path):
    """
    Ejecuta un archivo .sql completo.
    Maneja múltiples sentencias separadas por punto y coma.
    NO usa GO (porque pyodbc no lo reconoce).

    Parámetros:
        script_path (str): Ruta al archivo .sql

    Retorna:
        bool: True si fue exitoso
    """
    conn = get_connection()
    if conn is None:
        return False

    try:
        with open(script_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()

        cursor = conn.cursor()

        # Separar por punto y coma (las sentencias individuales)
        # Filtrar líneas vacías y comentarios
        statements = []
        for stmt in sql_content.split(';'):
            stmt = stmt.strip()
            # Ignorar líneas vacías, comentarios y la palabra GO
            if stmt and not stmt.startswith('--') and stmt.upper() != 'GO':
                statements.append(stmt)

        total = len(statements)
        for i, stmt in enumerate(statements, 1):
            try:
                cursor.execute(stmt)
                conn.commit()
                print(f"  ✅ Sentencia {i}/{total} ejecutada")
            except Exception as e:
                # Algunas sentencias pueden fallar (ej: DROP TABLE si no existe)
                # pero no detenemos todo el script
                print(f"  ⚠️ Sentencia {i}/{total} con advertencia: {str(e)[:80]}")
                conn.rollback()

        cursor.close()
        print(f"\n✅ Script completado: {script_path}")
        return True

    except Exception as e:
        print(f"❌ Error ejecutando script: {e}")
        return False

    finally:
        conn.close()


def test_connection():
    """
    Prueba completa de conexión a SQL Server.
    Muestra versión del servidor y bases de datos disponibles.
    """
    print("=" * 60)
    print("🔍 PRUEBA DE CONEXIÓN A SQL SERVER")
    print("=" * 60)

    conn = get_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()

        # 1. Versión de SQL Server
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        print(f"\n📊 Versión SQL Server:")
        print(f"   {version[:80]}...")

        # 2. Base de datos actual
        cursor.execute("SELECT DB_NAME()")
        db_name = cursor.fetchone()[0]
        print(f"\n📁 Base de datos activa: {db_name}")

        # 3. Tablas disponibles
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n📋 Tablas encontradas ({len(tables)}):")
        for table in tables:
            print(f"   • {table}")

        cursor.close()
        print(f"\n🎉 ¡Conexión verificada exitosamente!")
        return True

    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        return False

    finally:
        conn.close()


# ============================================
# EJECUCIÓN DIRECTA (para probar desde terminal)
# ============================================
if __name__ == "__main__":
    test_connection()