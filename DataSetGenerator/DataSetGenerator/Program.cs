
#nullable disable
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;

namespace C15Generator
{
    class Interaccion
    {
        public string Codigo, Usuario, Tipo, Canal, Motivo, Cola, Turno, Fuente, Grabacion;
        public DateTime Fecha;
        public int Dur, Espera, Trans, Casos;
        public int Recontacto;
    }

    class Program
    {
        // ===================== CONFIGURACION =====================
        static int seeds = 12000;
        static string connectionString =
            @"Server=DESKTOP-426D7G4\SQLEXPRESS;Database=ServicioCiudadanoDW;Integrated Security=True;TrustServerCertificate=True";
        static Random rnd = new Random(42);

        // Fuerza de los efectos (calibrada: accuracy ~78%, AUC ~0.86)
        const double KO = 3.0;
        const double KC = 1.0;
        const double TASA_OBJETIVO = 0.41;

        // Efectos calibrados para que las tasas por segmento coincidan con los Informes 2, 3 y 4
        static readonly Dictionary<string, double> EfCanal = new Dictionary<string, double>
            { {"Chat",-0.13}, {"Correo",-0.16}, {"Red social",0.39}, {"Teléfono",-0.57} };

        static readonly Dictionary<string, double> EfMotivo = new Dictionary<string, double>
            { {"Consulta",-0.49}, {"Falla",0.04}, {"Cobro",-0.07}, {"Queja",0.52}, {"Solicitud",0.27} };

        static readonly Dictionary<string, double> EfCola = new Dictionary<string, double>
            { {"Facturación",0.43}, {"Soporte",-0.78}, {"Información",0.54}, {"Reclamos",-1.05}, {"Trámites",0.33} };

        // ===================== HALLAZGOS DE LOS INFORMES =====================
        // Informe 1: registros por mes (ene-2025 a jun-2026)
        static readonly int[] PesoMes = { 12, 18, 17, 11, 18, 23, 21, 20, 24, 28, 24, 24, 15, 20, 20, 20, 24, 21 };

        // Informe 2: canal
        static readonly string[] Canales = { "Chat", "Correo", "Red social", "Teléfono" };
        static readonly double[] PesoCanal = { 96, 83, 81, 91 };

        // Informe 3: motivo (COBRO/Cobro y QUEJA/Queja unidos)
        static readonly string[] Motivos = { "Consulta", "Falla", "Cobro", "Queja", "Solicitud" };
        static readonly double[] PesoMotivo = { 60, 80, 72, 81, 67 };

        // Informe 4: cola
        static readonly string[] Colas = { "Facturación", "Soporte", "Información", "Reclamos", "Trámites" };
        static readonly double[] PesoCola = { 68, 68, 95, 54, 75 };

        // SUPUESTOS (no estan en los informes)
        static readonly string[] Tipos = { "Nuevo", "Recurrente", "Empresa", "Adulto mayor" };
        static readonly double[] PesoTipo = { 30, 35, 15, 20 };
        static readonly int[] UsuariosPorTipo = { 400, 145, 100, 267 };

        static readonly string[] Fuentes = { "ACD", "CRM", "Ticketing", "Calidad" };
        static readonly double[] PesoFuente = { 45, 35, 12, 8 };

        // PROPORCIONES DE SUCIEDAD
        const double P_SIN_DATO_CANAL = 0.025;
        const double P_RESULTADO_VACIO = 0.017;
        const double P_COBRO_MAYUS = 0.028;
        const double P_QUEJA_MAYUS = 0.025;
        const double P_DUPLICADO = 0.03;
        const double P_FUERA_RANGO = 0.02;
        const double P_CARGA_TARDIA = 0.05;
        const double P_FALTANTE_OTRO = 0.02;

        // Mayo 2025: fechas en formato dd/MM/yyyy (error de formato, Informe 1)

        static void Main(string[] args)
        {
            Console.WriteLine("Generando " + seeds + " interacciones...");
            var lista = new List<Interaccion>();

            // 1) Fechas segun el volumen mensual
            DateTime inicio = new DateTime(2025, 1, 1);
            double[] pm = PesoMes.Select(x => (double)x).ToArray();
            string[] nombresMes = Enumerable.Range(0, 18).Select(x => x.ToString()).ToArray();

            for (int i = 0; i < seeds; i++)
            {
                int m = int.Parse(Pick(nombresMes, pm));
                DateTime mes = inicio.AddMonths(m);
                int dia = rnd.Next(0, DateTime.DaysInMonth(mes.Year, mes.Month));
                lista.Add(new Interaccion { Fecha = mes.AddDays(dia) });
            }

            lista = lista.OrderBy(x => x.Fecha).ToList();

            // 2) Atributos, historial y probabilidad de recontacto
            double escala = seeds / 12000.0;
            var historial = new Dictionary<string, Queue<DateTime>>();
            var z = new double[seeds];

            for (int i = 0; i < seeds; i++)
            {
                var it = lista[i];
                it.Codigo = "INT-" + (i + 1).ToString("000000");
                it.Tipo = Pick(Tipos, PesoTipo);

                int ti = Array.IndexOf(Tipos, it.Tipo);
                int pool = Math.Max(5, (int)(UsuariosPorTipo[ti] * escala));
                it.Usuario = "U" + ti + "-" + rnd.Next(0, pool).ToString("0000");

                it.Canal = Pick(Canales, PesoCanal);
                it.Motivo = Pick(Motivos, PesoMotivo);
                it.Cola = Pick(Colas, PesoCola);
                it.Fuente = Pick(Fuentes, PesoFuente);
                it.Grabacion = rnd.NextDouble() < 0.85 ? "Sí" : "No";

                // historial: SOLO contactos anteriores del mismo usuario en 30 dias (sin fuga)
                if (!historial.ContainsKey(it.Usuario))
                    historial[it.Usuario] = new Queue<DateTime>();

                var q = historial[it.Usuario];

                while (q.Count > 0 && (it.Fecha - q.Peek()).TotalDays > 30)
                    q.Dequeue();

                it.Casos = Math.Min(8, q.Count);
                q.Enqueue(it.Fecha);

                bool finde = it.Fecha.DayOfWeek == DayOfWeek.Saturday ||
                             it.Fecha.DayOfWeek == DayOfWeek.Sunday;

                it.Turno = finde
                    ? "Fin de semana"
                    : Pick(new[] { "AM", "PM", "Nocturno" }, new[] { 0.42, 0.42, 0.16 });

                bool noct = it.Turno == "Nocturno";
                bool falla = it.Motivo == "Falla";
                bool queja = it.Motivo == "Queja";
                bool consulta = it.Motivo == "Consulta";
                bool fact = it.Cola == "Facturación";
                bool sop = it.Cola == "Soporte";
                bool recl = it.Cola == "Reclamos";
                bool red = it.Canal == "Red social";
                bool tel = it.Canal == "Teléfono";

                it.Trans = Math.Min(4, Poisson(
                    0.5 + 0.5 * B(recl) + 0.4 * B(falla) + 0.3 * B(sop)));

                double esp = GammaK2(130) *
                    (1 + 0.35 * B(noct) + 0.30 * B(fact) + 0.20 * B(tel))
                    + 60 * it.Trans;

                it.Espera = (int)Math.Round(Clip(esp, 0, 900));

                double du = Normal(900, 300) + 110 * it.Trans
                    + 70 * B(falla) - 60 * B(consulta);

                it.Dur = (int)Math.Round(Clip(du, 30, 1800));

                z[i] = KO * (0.55 * it.Trans + 0.0016 * (it.Espera - 250) + 0.30 * it.Casos
                            + 0.70 * B(it.Dur < 400) + 0.00035 * (it.Dur - 900)
                            + 0.25 * B(noct) + 0.30 * B(it.Tipo == "Adulto mayor") - 0.20 * B(it.Tipo == "Empresa"))
                     + EfCanal[it.Canal] + EfMotivo[it.Motivo] + EfCola[it.Cola];
            }

            // 3) Intercepto para que la tasa global sea ~41%; luego se decide el resultado
            double lo = -12, hi = 12;

            for (int k = 0; k < 60; k++)
            {
                double b = (lo + hi) / 2, m = 0;

                for (int i = 0; i < seeds; i++)
                    m += 1.0 / (1.0 + Math.Exp(-(z[i] + b)));

                m /= seeds;

                if (m > TASA_OBJETIVO) hi = b;
                else lo = b;
            }

            double inter = (lo + hi) / 2;

            for (int i = 0; i < seeds; i++)
            {
                double p = 1.0 / (1.0 + Math.Exp(-(z[i] + inter)));
                lista[i].Recontacto = rnd.NextDouble() < p ? 1 : 0;
            }

            // 4) Capa de suciedad -> tabla staging
            var tabla = new DataTable();

            string[] cols = {
                "id_interaccion", "id_usuario_seud", "fecha_contacto",
                "cola_servicio", "canal", "tipo_usuario", "motivo_contacto",
                "duracion_seg", "espera_seg", "transferencias",
                "casos_previos_30d", "turno", "sistema_origen",
                "recontacto_7_dias", "grabacion_autorizada", "fecha_carga"
            };

            foreach (var c in cols)
                tabla.Columns.Add(c, typeof(string));

            int nSinDato = 0, nVacio = 0, nCobro = 0, nQueja = 0,
                nDup = 0, nRango = 0, nTarde = 0, nFalt = 0, nMayo = 0;

            foreach (var it in lista)
            {
                string fecha = it.Fecha.ToString("yyyy-MM-dd");

                if (it.Fecha.Year == 2025 && it.Fecha.Month == 5)
                {
                    fecha = it.Fecha.ToString("dd/MM/yyyy");
                    nMayo++;
                }

                string carga;

                if (rnd.NextDouble() < P_CARGA_TARDIA)
                {
                    carga = it.Fecha.AddDays(rnd.Next(31, 91)).ToString("yyyy-MM-dd");
                    nTarde++;
                }
                else
                {
                    carga = it.Fecha.AddDays(rnd.Next(0, 3)).ToString("yyyy-MM-dd");
                }

                string canal = it.Canal;

                if (rnd.NextDouble() < P_SIN_DATO_CANAL)
                {
                    canal = "(sin dato)";
                    nSinDato++;
                }

                string motivo = it.Motivo;

                if (motivo == "Cobro" && rnd.NextDouble() < P_COBRO_MAYUS)
                {
                    motivo = "COBRO";
                    nCobro++;
                }
                else if (motivo == "Queja" && rnd.NextDouble() < P_QUEJA_MAYUS)
                {
                    motivo = "QUEJA";
                    nQueja++;
                }

                string rec = it.Recontacto.ToString();

                if (rnd.NextDouble() < P_RESULTADO_VACIO)
                {
                    rec = null;
                    nVacio++;
                }

                string dur = it.Dur.ToString();
                string esp = it.Espera.ToString();
                string tra = it.Trans.ToString();
                string cas = it.Casos.ToString();

                if (rnd.NextDouble() < P_FUERA_RANGO)
                {
                    nRango++;

                    switch (rnd.Next(0, 4))
                    {
                        case 0:
                            dur = rnd.Next(1801, 4000).ToString();
                            break;
                        case 1:
                            esp = rnd.Next(901, 2000).ToString();
                            break;
                        case 2:
                            tra = (rnd.Next(0, 2) == 0 ? -1 : rnd.Next(5, 9)).ToString();
                            break;
                        default:
                            cas = (rnd.Next(0, 2) == 0 ? -1 : rnd.Next(9, 15)).ToString();
                            break;
                    }
                }

                string cola = it.Cola;
                string tipo = it.Tipo;
                string turno = it.Turno;

                if (rnd.NextDouble() < P_FALTANTE_OTRO)
                {
                    nFalt++;

                    switch (rnd.Next(0, 4))
                    {
                        case 0:
                            tipo = null;
                            break;
                        case 1:
                            turno = null;
                            break;
                        case 2:
                            cola = null;
                            break;
                        default:
                            dur = null;
                            break;
                    }
                }

                object[] fila = {
                    it.Codigo, it.Usuario, fecha, cola, canal, tipo, motivo,
                    dur, esp, tra, cas, turno, it.Fuente, rec,
                    it.Grabacion, carga
                };

                tabla.Rows.Add(fila.Select(v => v ?? DBNull.Value).ToArray());

                if (rnd.NextDouble() < P_DUPLICADO)
                {
                    tabla.Rows.Add(fila.Select(v => v ?? DBNull.Value).ToArray());
                    nDup++;
                }
            }

            // 5) Carga a SQL Server
            using (var con = new SqlConnection(connectionString))
            {
                con.Open();

                using (var cmd = new SqlCommand("TRUNCATE TABLE stg_c15_v0", con))
                    cmd.ExecuteNonQuery();

                using (var bulk = new SqlBulkCopy(con))
                {
                    bulk.DestinationTableName = "stg_c15_v0";
                    bulk.BulkCopyTimeout = 300;

                    foreach (var c in cols)
                        bulk.ColumnMappings.Add(c, c);

                    bulk.WriteToServer(tabla);
                }
            }

            // 6) Resumen de lo generado (guardalo: es la evidencia de lo sembrado)
            Console.WriteLine("Filas en staging: " + tabla.Rows.Count);

            Console.WriteLine("Recontacto global (limpio): " +
                (lista.Average(x => (double)x.Recontacto) * 100).ToString("F1") + "%");

            Console.WriteLine("Sembrado -> sin dato canal: " + nSinDato +
                " | resultado vacio: " + nVacio +
                " | COBRO: " + nCobro +
                " | QUEJA: " + nQueja +
                " | duplicados: " + nDup +
                " | fuera de rango: " + nRango +
                " | carga tardia: " + nTarde +
                " | otros faltantes: " + nFalt +
                " | fechas mayo-2025 con formato dd/MM: " + nMayo);

            foreach (var g in lista.GroupBy(x => x.Canal))
                Console.WriteLine("Canal " + g.Key + ": " +
                    (g.Average(x => (double)x.Recontacto) * 100).ToString("F1") + "%");

            foreach (var g in lista.GroupBy(x => x.Motivo))
                Console.WriteLine("Motivo " + g.Key + ": " +
                    (g.Average(x => (double)x.Recontacto) * 100).ToString("F1") + "%");

            foreach (var g in lista.GroupBy(x => x.Cola))
                Console.WriteLine("Cola " + g.Key + ": " +
                    (g.Average(x => (double)x.Recontacto) * 100).ToString("F1") + "%");
        }

        // ===================== UTILIDADES =====================
        static double B(bool v)
        {
            return v ? 1.0 : 0.0;
        }

        static double Clip(double v, double a, double b)
        {
            return Math.Max(a, Math.Min(b, v));
        }

        static string Pick(string[] nombres, double[] pesos)
        {
            double total = pesos.Sum();
            double r = rnd.NextDouble() * total;
            double acc = 0;

            for (int i = 0; i < nombres.Length; i++)
            {
                acc += pesos[i];
                if (r < acc)
                    return nombres[i];
            }

            return nombres[nombres.Length - 1];
        }

        static int Poisson(double lambda)
        {
            double L = Math.Exp(-lambda), p = 1.0;
            int k = 0;

            do
            {
                k++;
                p *= rnd.NextDouble();
            }
            while (p > L);

            return k - 1;
        }

        static double Normal(double mu, double sd)
        {
            double u1 = 1.0 - rnd.NextDouble();
            double u2 = rnd.NextDouble();

            return mu + sd * Math.Sqrt(-2.0 * Math.Log(u1))
                * Math.Cos(2.0 * Math.PI * u2);
        }

        static double GammaK2(double theta)
        {
            return -theta * (
                Math.Log(1.0 - rnd.NextDouble()) +
                Math.Log(1.0 - rnd.NextDouble()));
        }
    }
}
