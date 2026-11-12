# ============================================
# test_conexion.py
# Prueba rápida de conexión a SQL Server
# ============================================

from src.database import test_connection, execute_query

if __name__ == "__main__":
    # Prueba 1: Conexión básica
    print("\n🔌 PRUEBA 1: Conexión al servidor")
    test_connection()

    # Prueba 2: Consulta a las tablas del DW
    print("\n\n📊 PRUEBA 2: Conteo de registros en tablas del DW")
    tablas = ['hecho_reserva', 'hecho_pago', 'dim_cliente', 'dim_tiempo']

    for tabla in tablas:
        query = f"SELECT COUNT(*) AS total FROM dbo.{tabla};"
        df = execute_query(query)
        if df is not None:
            print(f"   📋 {tabla}: {df['total'][0]:,} registros")

    # Prueba 3: Verificar datasets de minería
    print("\n\n🤖 PRUEBA 3: Datasets de minería de datos")
    datasets = ['dm_dataset_abandono', 'dm_dataset_recompra']

    for ds in datasets:
        query = f"""
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END) AS con_datos
            FROM dbo.{ds};
        """
        # Consulta simplificada
        query_simple = f"SELECT COUNT(*) AS total FROM dbo.{ds};"
        df = execute_query(query_simple)
        if df is not None:
            print(f"   🤖 {ds}: {df['total'][0]:,} registros")
        else:
            print(f"   ⚠️ {ds}: No existe aún. Ejecuta los scripts SQL primero.")