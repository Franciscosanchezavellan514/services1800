// ============================================================
// SKYTRAVEL NICARAGUA - GENERADOR DE DATOS PARA DATA WAREHOUSE
// Genera datos respetando los hallazgos del caso de estudio
// Autor: Equipo de Analítica
// Fecha: Septiembre 2026
// ============================================================

using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;

namespace SkyTravelDataGenerator
{
    class Program
    {
        // ============================================================
        // CONFIGURACIÓN PRINCIPAL - MODIFICAR AQUÍ EL VOLUMEN DE DATOS
        // ============================================================
        
        // ⚙️ NÚMERO DE RESERVAS A GENERAR (escalable)
        // Ejemplos:
        //   seeds = 100     → Prueba rápida (~100 reservas)
        //   seeds = 10000   → Dataset mediano
        //   seeds = 120000  → Dataset completo (réplica del caso real)
        static int seeds = 1000;  // ← CAMBIAR ESTE VALOR SEGÚN NECESIDAD
        
        // Cadena de conexión (ajustar según tu servidor)
        static string connectionString = 
            @"Server=localhost;Database=SkyTravelDW;Integrated Security=True;TrustServerCertificate=True";
        
        // Semilla para reproducibilidad de datos
        static Random rnd = new Random(42);
        
        // ============================================================
        // CONSTANTES BASADAS EN LOS HALLAZGOS DEL CASO DE ESTUDIO
        // ============================================================
        
        // Distribución de canales (Informe 2)
        static readonly Dictionary<string, double> DistribucionCanales = new Dictionary<string, double>
        {
            { "web",      0.433 },  // 52,000 / 120,000
            { "app",      0.400 },  // 48,000 / 120,000
            { "tablet",   0.067 },  //  8,000 / 120,000
            { "sucursal", 0.100 }   // 12,000 / 120,000
        };
        
        // Tasas de abandono por canal (Informe 2)
        static readonly Dictionary<string, double> TasaAbandonoPorCanal = new Dictionary<string, double>
        {
            { "web",      0.47 },
            { "app",      0.74 },  // ← PEOR canal
            { "tablet",   0.56 },
            { "sucursal", 0.28 }   // ← MEJOR canal
        };
        
        // Tasas de abandono por tipo de vuelo (Informe 5)
        static readonly Dictionary<string, double> TasaAbandonoPorTipoVuelo = new Dictionary<string, double>
        {
            { "nacional",      0.49 },
            { "internacional", 0.71 }
        };
        
        // Distribución de tipo de vuelo (Informe 5)
        static readonly Dictionary<string, double> DistribucionTipoVuelo = new Dictionary<string, double>
        {
            { "nacional",      0.567 },  // 68,000 / 120,000
            { "internacional", 0.433 }   // 52,000 / 120,000
        };
        
        // Tasas de abandono por ruta (Informe 8)
        static readonly Dictionary<string, double> TasaAbandonoPorRuta = new Dictionary<string, double>
        {
            { "MGA-MIA", 0.78 },  // ← Ruta más crítica
            { "MGA-GUA", 0.71 },
            { "MGA-SJO", 0.65 },
            { "MGA-MEX", 0.58 },
            { "MGA-PTY", 0.45 },
            { "nacional",0.49 }   // Promedio rutas nacionales
        };
        
        // Distribución de rutas internacionales (Informe 8)
        static readonly Dictionary<string, double> DistribucionRutasIntl = new Dictionary<string, double>
        {
            { "MGA-MIA", 0.356 },  // 18,500 / 52,000
            { "MGA-GUA", 0.237 },
            { "MGA-SJO", 0.188 },
            { "MGA-MEX", 0.158 },
            { "MGA-PTY", 0.061 }
        };
        
        // Segmentos de clientes y su distribución (Informe 3)
        static readonly Dictionary<string, double> DistribucionSegmentos = new Dictionary<string, double>
        {
            { "Viajeros de Negocios", 0.091 },   //  4,280 / 47,250
            { "Familias Premium",     0.045 },   //  2,150 / 47,250
            { "Millennials Digitales",0.082 },   //  3,890 / 47,250
            { "Estudiantes",          0.036 },   //  1,720 / 47,250
            { "Ocasional Promo",      0.115 },   //  5,420 / 47,250
            { "Otros",                0.631 }    // 29,790 / 47,250
        };
        
        // Tasas de recompra por segmento (Informe 3)
        static readonly Dictionary<string, double> TasaRecompraPorSegmento = new Dictionary<string, double>
        {
            { "Viajeros de Negocios",  0.68 },
            { "Familias Premium",      0.52 },
            { "Millennials Digitales", 0.38 },
            { "Estudiantes",           0.24 },
            { "Ocasional Promo",       0.11 },  // ← PEOR segmento
            { "Otros",                 0.16 }
        };
        
        // Métodos de pago
        static readonly string[] MetodosPago = { "tarjeta", "transferencia", "efectivo" };
        
        // Nombres para generar clientes
        static readonly string[] Nombres = { "Juan", "María", "Carlos", "Ana", "Luis", "Sofía", 
            "Pedro", "Laura", "Miguel", "Carmen", "José", "Rosa", "Daniel", "Elena", "Francisco" };
        static readonly string[] Apellidos = { "García", "Rodríguez", "Martínez", "López", 
            "González", "Hernández", "Pérez", "Sánchez", "Ramírez", "Torres", "Flores", "Mendoza" };
        static readonly string[] Ciudades = { "Managua", "León", "Granada", "Masaya", "Estelí", 
            "Chinandega", "Tipitapa", "Matagalpa" };
        
        // ============================================================
        // PUNTO DE ENTRADA PRINCIPAL
        // ============================================================
        static void Main(string[] args)
        {
            Console.WriteLine("╔════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║   SKYTRAVEL NICARAGUA - GENERADOR DE DATOS PARA DW        ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════╝");
            Console.WriteLine($"\n📊 Configuración:");
            Console.WriteLine($"   • Reservas a generar: {seeds:N0}");
            Console.WriteLine($"   • Clientes estimados: {EstimarClientes():N0}");
            Console.WriteLine($"   • Periodo: Enero 2024 - Septiembre 2026");
            Console.WriteLine();
            
            try
            {
                // 1. Generar calendario
                Console.WriteLine("📅 Generando dim_tiempo...");
                GenerarDimTiempo();
                
                // 2. Generar clientes
                int totalClientes = EstimarClientes();
                Console.WriteLine($"👥 Generando {totalClientes:N0} clientes en dim_cliente...");
                var clientes = GenerarClientes(totalClientes);
                InsertarClientes(clientes);
                
                // 3. Generar reservas y pagos
                Console.WriteLine($"✈️  Generando {seeds:N0} reservas en hecho_reserva...");
                GenerarReservas(clientes);
                
                // 4. Actualizar métricas de clientes (recompra, LTV)
                Console.WriteLine("🔄 Actualizando métricas de clientes...");
                ActualizarMetricasClientes();
                
                Console.WriteLine("\n✅ ¡Generación completada exitosamente!");
                Console.WriteLine($"   Total reservas: {seeds:N0}");
                Console.WriteLine($"   Total clientes: {totalClientes:N0}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ ERROR: {ex.Message}");
                Console.WriteLine($"   Detalle: {ex.StackTrace}");
            }
            
            Console.WriteLine("\nPresione cualquier tecla para salir...");
            Console.ReadKey();
        }
        
        // ============================================================
        // MÉTODO: Estimar número de clientes según seeds
        // ============================================================
        static int EstimarClientes()
        {
            // En el caso real: 120,000 reservas → 47,250 clientes (ratio ~2.54)
            return Math.Max(50, (int)(seeds / 2.54));
        }
        
        // ============================================================
        // MÉTODO: Generar dim_tiempo (calendario)
        // ============================================================
        static void GenerarDimTiempo()
        {
            using (var conn = new SqlConnection(connectionString))
            {
                conn.Open();
                
                // Limpiar tabla primero
                using (var cmd = new SqlCommand("DELETE FROM dim_tiempo", conn))
                    cmd.ExecuteNonQuery();
                
                DateTime inicio = new DateTime(2024, 1, 1);
                DateTime fin = new DateTime(2026, 9, 30);
                
                using (var transaction = conn.BeginTransaction())
                {
                    using (var cmd = new SqlCommand())
                    {
                        cmd.Connection = conn;
                        cmd.Transaction = transaction;
                        cmd.CommandText = @"
                            INSERT INTO dim_tiempo 
                            (fecha, anio, mes, nombre_mes, dia, dia_semana, nombre_dia_semana, 
                             trimestre, es_fin_semana, es_temporada_alta)
                            VALUES 
                            (@fecha, @anio, @mes, @nombreMes, @dia, @diaSemana, @nombreDiaSemana,
                             @trimestre, @esFinSemana, @esTemporadaAlta)";
                        
                        cmd.Parameters.Add("@fecha", SqlDbType.Date);
                        cmd.Parameters.Add("@anio", SqlDbType.Int);
                        cmd.Parameters.Add("@mes", SqlDbType.Int);
                        cmd.Parameters.Add("@nombreMes", SqlDbType.NVarChar, 20);
                        cmd.Parameters.Add("@dia", SqlDbType.Int);
                        cmd.Parameters.Add("@diaSemana", SqlDbType.Int);
                        cmd.Parameters.Add("@nombreDiaSemana", SqlDbType.NVarChar, 20);
                        cmd.Parameters.Add("@trimestre", SqlDbType.Int);
                        cmd.Parameters.Add("@esFinSemana", SqlDbType.Bit);
                        cmd.Parameters.Add("@esTemporadaAlta", SqlDbType.Bit);
                        
                        int contador = 0;
                        for (DateTime fecha = inicio; fecha <= fin; fecha = fecha.AddDays(1))
                        {
                            cmd.Parameters["@fecha"].Value = fecha;
                            cmd.Parameters["@anio"].Value = fecha.Year;
                            cmd.Parameters["@mes"].Value = fecha.Month;
                            cmd.Parameters["@nombreMes"].Value = fecha.ToString("MMMM");
                            cmd.Parameters["@dia"].Value = fecha.Day;
                            cmd.Parameters["@diaSemana"].Value = (int)fecha.DayOfWeek;
                            cmd.Parameters["@nombreDiaSemana"].Value = fecha.ToString("dddd");
                            cmd.Parameters["@trimestre"].Value = (fecha.Month - 1) / 3 + 1;
                            cmd.Parameters["@esFinSemana"].Value = 
                                (fecha.DayOfWeek == DayOfWeek.Saturday || fecha.DayOfWeek == DayOfWeek.Sunday);
                            
                            // Temporada alta: Junio-Agosto y Diciembre
                            bool temporadaAlta = (fecha.Month >= 6 && fecha.Month <= 8) || fecha.Month == 12;
                            cmd.Parameters["@esTemporadaAlta"].Value = temporadaAlta;
                            
                            cmd.ExecuteNonQuery();
                            contador++;
                            
                            if (contador % 100 == 0)
                                Console.Write($"\r   Procesados {contador} días...");
                        }
                    }
                    transaction.Commit();
                }
            }
            Console.WriteLine($"\r   ✓ dim_tiempo poblada con ~{((DateTime)new DateTime(2026,9,30) - new DateTime(2024,1,1)).TotalDays:N0} días");
        }
        
        // ============================================================
        // MÉTODO: Generar lista de clientes
        // ============================================================
        static List<Cliente> GenerarClientes(int total)
        {
            var clientes = new List<Cliente>();
            int id = 1;
            
            foreach (var kvp in DistribucionSegmentos)
            {
                string segmento = kvp.Key;
                int cantidadSegmento = (int)(total * kvp.Value);
                
                for (int i = 0; i < cantidadSegmento; i++)
                {
                    var cliente = new Cliente
                    {
                        Id = id++,
                        Nombre = $"{Nombres[rnd.Next(Nombres.Length)]} {Apellidos[rnd.Next(Apellidos.Length)]} {Apellidos[rnd.Next(Apellidos.Length)]}",
                        Email = $"cliente{id}@email.com",
                        Telefono = $"+505 {rnd.Next(2000,9999)}-{rnd.Next(1000,9999)}",
                        Ciudad = Ciudades[rnd.Next(Ciudades.Length)],
                        // Antigüedad: más clientes nuevos que viejos (según Informe 4)
                        FechaRegistro = GenerarFechaRegistro(),
                        Segmento = segmento
                    };
                    clientes.Add(cliente);
                }
            }
            
            return clientes;
        }
        
        static DateTime GenerarFechaRegistro()
        {
            // Distribución de antigüedad según Informe 4:
            // <1 año: 38.5%, 1-3 años: 32.6%, 3-5 años: 18.8%, >5 años: 10.1%
            double r = rnd.NextDouble();
            DateTime hoy = new DateTime(2026, 9, 11);
            
            if (r < 0.385)
                return hoy.AddDays(-rnd.Next(30, 365));           // < 1 año
            else if (r < 0.711)
                return hoy.AddDays(-rnd.Next(365, 1095));         // 1-3 años
            else if (r < 0.899)
                return hoy.AddDays(-rnd.Next(1095, 1825));        // 3-5 años
            else
                return hoy.AddDays(-rnd.Next(1825, 3650));        // > 5 años
        }
        
        // ============================================================
        // MÉTODO: Insertar clientes en BD
        // ============================================================
        static void InsertarClientes(List<Cliente> clientes)
        {
            using (var conn = new SqlConnection(connectionString))
            {
                conn.Open();
                
                using (var cmd = new SqlCommand())
                {
                    cmd.Connection = conn;
                    cmd.CommandText = @"
                        INSERT INTO dim_cliente 
                        (nombre_cliente, email, telefono, ciudad, fecha_registro, 
                         antiguedad_cliente_dias, segmento_cliente)
                        OUTPUT INSERTED.id_cliente
                        VALUES 
                        (@nombre, @email, @telefono, @ciudad, @fechaRegistro,
                         @antiguedad, @segmento)";
                    
                    cmd.Parameters.Add("@nombre", SqlDbType.NVarChar, 100);
                    cmd.Parameters.Add("@email", SqlDbType.NVarChar, 150);
                    cmd.Parameters.Add("@telefono", SqlDbType.NVarChar, 20);
                    cmd.Parameters.Add("@ciudad", SqlDbType.NVarChar, 50);
                    cmd.Parameters.Add("@fechaRegistro", SqlDbType.Date);
                    cmd.Parameters.Add("@antiguedad", SqlDbType.Int);
                    cmd.Parameters.Add("@segmento", SqlDbType.NVarChar, 50);
                    
                    DateTime hoy = new DateTime(2026, 9, 11);
                    
                    using (var transaction = conn.BeginTransaction())
                    {
                        cmd.Transaction = transaction;
                        int contador = 0;
                        
                        foreach (var c in clientes)
                        {
                            cmd.Parameters["@nombre"].Value = c.Nombre;
                            cmd.Parameters["@email"].Value = c.Email;
                            cmd.Parameters["@telefono"].Value = c.Telefono;
                            cmd.Parameters["@ciudad"].Value = c.Ciudad;
                            cmd.Parameters["@fechaRegistro"].Value = c.FechaRegistro;
                            cmd.Parameters["@antiguedad"].Value = (int)(hoy - c.FechaRegistro).TotalDays;
                            cmd.Parameters["@segmento"].Value = c.Segmento;
                            
                            c.Id = (int)cmd.ExecuteScalar();
                            contador++;
                            
                            if (contador % 500 == 0)
                                Console.Write($"\r   Insertados {contador:N0} clientes...");
                        }
                        
                        transaction.Commit();
                        Console.WriteLine($"\r   ✓ {contador:N0} clientes insertados en dim_cliente");
                    }
                }
            }
        }
        
        // ============================================================
        // MÉTODO: Generar reservas respetando los hallazgos
        // ============================================================
        static void GenerarReservas(List<Cliente> clientes)
        {
            using (var conn = new SqlConnection(connectionString))
            {
                conn.Open();
                
                using (var transaction = conn.BeginTransaction())
                {
                    int reservasPagadas = 0, reservasAbandonadas = 0;
                    int intentosFallidosTotal = 0;
                    
                    for (int i = 0; i < seeds; i++)
                    {
                        // 1. Seleccionar canal según distribución
                        string canal = SeleccionarConDistribucion(DistribucionCanales);
                        int idCanal = ObtenerIdCanal(conn, canal);
                        string dispositivo = ObtenerDispositivo(canal);
                        
                        // 2. Seleccionar tipo de vuelo
                        string tipoVuelo = SeleccionarConDistribucion(DistribucionTipoVuelo);
                        
                        // 3. Seleccionar ruta
                        string ruta = SeleccionarRuta(tipoVuelo);
                        int idVuelo = ObtenerIdVuelo(conn, ruta);
                        
                        // 4. Seleccionar cliente aleatorio
                        var cliente = clientes[rnd.Next(clientes.Count)];
                        
                        // 5. Generar fecha de reserva (con caída progresiva 2024→2025)
                        DateTime fechaReserva = GenerarFechaReserva();
                        
                        // 6. Determinar si abandona (aplicando tasas de abandono)
                        bool abandona = DeterminarAbandono(canal, tipoVuelo, ruta);
                        
                        // 7. Generar precio con variación
                        decimal precioBase = ObtenerPrecioBase(conn, ruta);
                        decimal precioCotizado = precioBase * (decimal)(0.85 + rnd.NextDouble() * 0.35);
                        
                        // 8. Insertar reserva
                        int idReserva = InsertarReserva(conn, transaction, cliente.Id, idVuelo, 
                            idCanal, fechaReserva, tipoVuelo, ruta, precioCotizado, abandona);
                        
                        if (abandona)
                        {
                            reservasAbandonadas++;
                            
                            // Algunos abandonos tienen intentos de pago fallidos
                            if (rnd.NextDouble() < 0.35)  // 35% de abandonos tuvieron intento fallido
                            {
                                int intentos = rnd.Next(1, 4);
                                for (int j = 0; j < intentos; j++)
                                {
                                    InsertarPago(conn, transaction, cliente.Id, idReserva, 
                                        fechaReserva.AddMinutes(rnd.Next(1, 30)),
                                        precioCotizado, "tarjeta", "rechazado", j + 1);
                                    intentosFallidosTotal++;
                                }
                            }
                        }
                        else
                        {
                            reservasPagadas++;
                            
                            // Pago exitoso (puede haber 1 intento fallido previo)
                            if (rnd.NextDouble() < 0.15)  // 15% tuvieron 1 intento fallido antes de éxito
                            {
                                InsertarPago(conn, transaction, cliente.Id, idReserva,
                                    fechaReserva.AddMinutes(rnd.Next(1, 10)),
                                    precioCotizado, "tarjeta", "rechazado", 1);
                                intentosFallidosTotal++;
                            }
                            
                            string metodoPago = canal == "sucursal" ? "efectivo" : 
                                               (rnd.NextDouble() < 0.8 ? "tarjeta" : "transferencia");
                            
                            InsertarPago(conn, transaction, cliente.Id, idReserva,
                                fechaReserva.AddMinutes(rnd.Next(2, 60)),
                                precioCotizado, metodoPago, "exitoso", 1);
                        }
                        
                        if ((i + 1) % 1000 == 0)
                        {
                            Console.Write($"\r   Progreso: {i + 1:N0}/{seeds:N0} reservas " +
                                          $"(Pagadas: {reservasPagadas:N0}, Abandonadas: {reservasAbandadas:N0})");
                        }
                    }
                    
                    transaction.Commit();
                    
                    Console.WriteLine($"\n\n📊 RESUMEN DE GENERACIÓN:");
                    Console.WriteLine($"   ✓ Reservas pagadas:     {reservasPagadas:N0} ({reservasPagadas * 100.0 / seeds:F1}%)");
                    Console.WriteLine($"   ✓ Reservas abandonadas: {reservasAbandonadas:N0} ({reservasAbandonadas * 100.0 / seeds:F1}%)");
                    Console.WriteLine($"   ✓ Intentos pago fallidos: {intentosFallidosTotal:N0}");
                }
            }
        }
        
        // ============================================================
        // MÉTODOS AUXILIARES
        // ============================================================
        
        static string SeleccionarConDistribucion(Dictionary<string, double> distribucion)
        {
            double r = rnd.NextDouble();
            double acumulado = 0;
            
            foreach (var kvp in distribucion)
            {
                acumulado += kvp.Value;
                if (r <= acumulado) return kvp.Key;
            }
            return distribucion.Keys.Last();
        }
        
        static string SeleccionarRuta(string tipoVuelo)
        {
            if (tipoVuelo == "nacional")
            {
                // Rutas nacionales aleatorias
                string[] nacionales = { "MGA-MGA", "MGA-BZE", "MGA-RFS" };
                return nacionales[rnd.Next(nacionales.Length)];
            }
            else
            {
                return SeleccionarConDistribucion(DistribucionRutasIntl);
            }
        }
        
        static bool DeterminarAbandono(string canal, string tipoVuelo, string ruta)
        {
            // Lógica en cascada: la ruta específica tiene prioridad sobre el tipo de vuelo
            double tasaAbandono;
            
            if (TasaAbandonoPorRuta.ContainsKey(ruta))
                tasaAbandono = TasaAbandonoPorRuta[ruta];
            else if (TasaAbandonoPorTipoVuelo.ContainsKey(tipoVuelo))
                tasaAbandono = TasaAbandonoPorTipoVuelo[tipoVuelo];
            else
                tasaAbandono = 0.585;  // Promedio general
            
            // Ajuste por canal (multiplicador)
            double factorCanal = canal switch
            {
                "app"      => 1.25,  // App empeora el abandono
                "sucursal" => 0.60,  // Sucursal mejora el abandono
                "tablet"   => 1.10,
                _          => 1.00
            };
            
            tasaAbandono = Math.Min(0.95, tasaAbandono * factorCanal);
            
            return rnd.NextDouble() < tasaAbandono;
        }
        
        static DateTime GenerarFechaReserva()
        {
            // Genera fecha entre 2024-01-01 y 2026-09-11
            // Con caída progresiva: más reservas en 2024 que en 2025
            DateTime inicio = new DateTime(2024, 1, 1);
            DateTime fin = new DateTime(2026, 9, 11);
            int diasTotales = (int)(fin - inicio).TotalDays;
            
            // Aplicar sesgo: 60% de reservas en 2024, 35% en 2025, 5% en 2026
            double r = rnd.NextDouble();
            DateTime fecha;
            
            if (r < 0.60)
                fecha = new DateTime(2024, 1, 1).AddDays(rnd.Next(0, 366));
            else if (r < 0.95)
                fecha = new DateTime(2025, 1, 1).AddDays(rnd.Next(0, 365));
            else
                fecha = new DateTime(2026, 1, 1).AddDays(rnd.Next(0, 254));
            
            // Hora aleatoria (con más actividad en horario laboral)
            int hora;
            double rh = rnd.NextDouble();
            if (rh < 0.70) hora = rnd.Next(8, 20);      // 70% entre 8am-8pm
            else if (rh < 0.90) hora = rnd.Next(20, 24); // 20% noche
            else hora = rnd.Next(0, 8);                  // 10% madrugada
            
            return fecha.AddHours(hora).AddMinutes(rnd.Next(0, 60));
        }
        
        static int ObtenerIdCanal(SqlConnection conn, string canal)
        {
            return canal switch
            {
                "web" => 1,
                "app" => 2,
                "tablet" => 3,
                "sucursal" => 4,
                _ => 1
            };
        }
        
        static string ObtenerDispositivo(string canal)
        {
            return canal switch
            {
                "web" => "desktop",
                "app" => "móvil",
                "tablet" => "tablet",
                "sucursal" => "presencial",
                _ => "desktop"
            };
        }
        
        static int ObtenerIdVuelo(SqlConnection conn, string ruta)
        {
            using (var cmd = new SqlCommand(
                "SELECT id_vuelo FROM dim_vuelo WHERE codigo_ruta = @ruta", conn))
            {
                cmd.Parameters.AddWithValue("@ruta", ruta);
                var result = cmd.ExecuteScalar();
                return result != null ? (int)result : 1;
            }
        }
        
        static decimal ObtenerPrecioBase(SqlConnection conn, string ruta)
        {
            using (var cmd = new SqlCommand(
                "SELECT precio_base_ruta FROM dim_vuelo WHERE codigo_ruta = @ruta", conn))
            {
                cmd.Parameters.AddWithValue("@ruta", ruta);
                var result = cmd.ExecuteScalar();
                return result != null ? (decimal)result : 300m;
            }
        }
        
        static int InsertarReserva(SqlConnection conn, SqlTransaction tx, int idCliente, 
            int idVuelo, int idCanal, DateTime fechaInicio, string tipoVuelo, 
            string ruta, decimal precio, bool abandona)
        {
            using (var cmd = new SqlCommand())
            {
                cmd.Connection = conn;
                cmd.Transaction = tx;
                cmd.CommandText = @"
                    INSERT INTO hecho_reserva 
                    (id_cliente, id_vuelo, id_canal, fecha_inicio_reserva, estado_reserva,
                     tipo_vuelo, ruta, precio_cotizado, pasajeros, fecha_completado_pago)
                    OUTPUT INSERTED.id_reserva
                    VALUES 
                    (@idCliente, @idVuelo, @idCanal, @fechaInicio, @estado,
                     @tipoVuelo, @ruta, @precio, 1, @fechaPago)";
                
                cmd.Parameters.AddWithValue("@idCliente", idCliente);
                cmd.Parameters.AddWithValue("@idVuelo", idVuelo);
                cmd.Parameters.AddWithValue("@idCanal", idCanal);
                cmd.Parameters.AddWithValue("@fechaInicio", fechaInicio);
                cmd.Parameters.AddWithValue("@estado", abandona ? "abandonada" : "pagada");
                cmd.Parameters.AddWithValue("@tipoVuelo", tipoVuelo);
                cmd.Parameters.AddWithValue("@ruta", ruta);
                cmd.Parameters.AddWithValue("@precio", precio);
                cmd.Parameters.AddWithValue("@fechaPago", abandona ? (object)DBNull.Value : 
                    fechaInicio.AddMinutes(rnd.Next(5, 120)));
                
                return (int)cmd.ExecuteScalar();
            }
        }
        
        static void InsertarPago(SqlConnection conn, SqlTransaction tx, int idCliente, 
            int idReserva, DateTime fechaPago, decimal monto, string metodo, string estado, int intento)
        {
            using (var cmd = new SqlCommand())
            {
                cmd.Connection = conn;
                cmd.Transaction = tx;
                cmd.CommandText = @"
                    INSERT INTO hecho_pago 
                    (id_cliente, id_reserva, fecha_pago, monto_pagado, metodo_pago, 
                     estado_pago, numero_intento)
                    VALUES 
                    (@idCliente, @idReserva, @fechaPago, @monto, @metodo, @estado, @intento)";
                
                cmd.Parameters.AddWithValue("@idCliente", idCliente);
                cmd.Parameters.AddWithValue("@idReserva", idReserva);
                cmd.Parameters.AddWithValue("@fechaPago", fechaPago);
                cmd.Parameters.AddWithValue("@monto", monto);
                cmd.Parameters.AddWithValue("@metodo", metodo);
                cmd.Parameters.AddWithValue("@estado", estado);
                cmd.Parameters.AddWithValue("@intento", intento);
                
                cmd.ExecuteNonQuery();
            }
        }
        
        // ============================================================
        // MÉTODO: Actualizar métricas de clientes (recompra, LTV)
        // ============================================================
        static void ActualizarMetricasClientes()
        {
            using (var conn = new SqlConnection(connectionString))
            {
                conn.Open();
                
                string sql = @"
                    UPDATE dim_cliente
                    SET 
                        fecha_primera_compra = subq.fecha_primera,
                        fecha_ultima_compra = subq.fecha_ultima,
                        total_compras_12m = subq.total_compras,
                        monto_total_12m = subq.monto_total,
                        ruta_mas_frecuente = subq.ruta_frecuente
                    FROM dim_cliente c
                    INNER JOIN (
                        SELECT 
                            id_cliente,
                            MIN(fecha_pago) AS fecha_primera,
                            MAX(fecha_pago) AS fecha_ultima,
                            COUNT(*) AS total_compras,
                            SUM(monto_pagado) AS monto_total
                        FROM hecho_pago
                        WHERE estado_pago = 'exitoso'
                        GROUP BY id_cliente
                    ) subq ON c.id_cliente = subq.id_cliente;
                    
                    -- Ruta más frecuente por cliente
                    UPDATE dim_cliente
                    SET ruta_mas_frecuente = subq.ruta
                    FROM dim_cliente c
                    INNER JOIN (
                        SELECT id_cliente, ruta, COUNT(*) as cnt,
                               ROW_NUMBER() OVER (PARTITION BY id_cliente ORDER BY COUNT(*) DESC) as rn
                        FROM hecho_reserva
                        WHERE estado_reserva = 'pagada'
                        GROUP BY id_cliente, ruta
                    ) subq ON c.id_cliente = subq.id_cliente AND subq.rn = 1;
                ";
                
                using (var cmd = new SqlCommand(sql, conn))
                {
                    int afectados = cmd.ExecuteNonQuery();
                    Console.WriteLine($"   ✓ Métricas actualizadas para {afectados:N0} clientes");
                }
            }
        }
    }
    
    // ============================================================
    // CLASE: Cliente (DTO)
    // ============================================================
    public class Cliente
    {
        public int Id { get; set; }
        public string Nombre { get; set; }
        public string Email { get; set; }
        public string Telefono { get; set; }
        public string Ciudad { get; set; }
        public DateTime FechaRegistro { get; set; }
        public string Segmento { get; set; }
    }
}