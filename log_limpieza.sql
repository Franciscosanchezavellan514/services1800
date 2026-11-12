USE ServicioCiudadanoDW;
SET NOCOUNT ON;

/* ---------- Reinicio (se puede volver a correr) ---------- */
TRUNCATE TABLE hecho_interaccion;
TRUNCATE TABLE log_limpieza;
IF OBJECT_ID('wrk_c15') IS NOT NULL DROP TABLE wrk_c15;
IF OBJECT_ID('rechazados_c15') IS NOT NULL DROP TABLE rechazados_c15;

CREATE TABLE rechazados_c15 (
    id_stg INT, id_interaccion NVARCHAR(50),
    regla VARCHAR(80), detalle VARCHAR(250)
);

/* Copia de trabajo: v0 (stg_c15_v0) queda intacta */
SELECT id_stg, id_interaccion, id_usuario_seud,
       fecha_contacto AS fecha_txt, cola_servicio, canal, tipo_usuario, motivo_contacto,
       duracion_seg AS dur_txt, espera_seg AS esp_txt, transferencias AS tra_txt,
       casos_previos_30d AS cas_txt, turno, sistema_origen,
       recontacto_7_dias AS rec_txt, grabacion_autorizada, fecha_carga AS carga_txt,
       CAST(NULL AS DATE) AS fecha_contacto, CAST(NULL AS DATE) AS fecha_carga,
       CAST(NULL AS INT) AS duracion_seg, CAST(NULL AS INT) AS espera_seg,
       CAST(NULL AS INT) AS transferencias, CAST(NULL AS INT) AS casos_previos_30d,
       CAST(NULL AS TINYINT) AS recontacto_7_dias
INTO wrk_c15
FROM stg_c15_v0;

DECLARE @tot INT, @af INT, @a INT, @b INT, @c INT, @d INT, @antes INT, @desp INT;

/* R00 Recepcion */
SET @tot = (SELECT COUNT(*) FROM stg_c15_v0);
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R00 Recepcion v0', '(todos)', 'Copia de trabajo; v0 intacto', @tot, @tot, 'stg_c15_v0 no se modifica');

/* R01 Unicidad: duplicados por id_interaccion */
SET @tot = (SELECT COUNT(*) FROM wrk_c15);
;WITH d AS (SELECT ROW_NUMBER() OVER (PARTITION BY id_interaccion ORDER BY id_stg) AS rn FROM wrk_c15)
DELETE FROM d WHERE rn > 1;
SET @af = @@ROWCOUNT;
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R01 Unicidad', 'id_interaccion', 'Eliminar copias; se conserva la primera', @af, @tot, 'Registros repetidos');

/* R02 Validez de fechas: unificar formato (mayo-2025 llegó como dd/MM/yyyy) */
SET @tot = (SELECT COUNT(*) FROM wrk_c15);
SET @af  = (SELECT COUNT(*) FROM wrk_c15 WHERE fecha_txt LIKE '%/%');
UPDATE wrk_c15
SET fecha_contacto = CASE
        WHEN fecha_txt LIKE '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]' THEN TRY_CONVERT(DATE, fecha_txt, 23)
        WHEN fecha_txt LIKE '[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]' THEN TRY_CONVERT(DATE, fecha_txt, 103)
    END,
    fecha_carga = TRY_CONVERT(DATE, carga_txt, 23);
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R02 Formato de fecha', 'fecha_contacto', 'Convertir dd/MM/yyyy a yyyy-MM-dd', @af, @tot, 'Error de formato (mayo 2025)');

INSERT INTO rechazados_c15
SELECT id_stg, id_interaccion, 'R02 Fecha invalida', CONCAT('fecha=', fecha_txt)
FROM wrk_c15 WHERE fecha_contacto IS NULL;
SET @af = @@ROWCOUNT;
DELETE FROM wrk_c15 WHERE fecha_contacto IS NULL;
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R02b Fecha invalida', 'fecha_contacto', 'Rechazar (no recuperable)', @af, @tot, 'Fecha no interpretable');

/* R03 Consistencia de etiquetas: COBRO -> Cobro, QUEJA -> Queja */
SET @tot = (SELECT COUNT(*) FROM wrk_c15);
UPDATE w SET w.motivo_contacto = d.motivo
FROM wrk_c15 w JOIN dim_motivo d ON w.motivo_contacto = d.motivo
WHERE w.motivo_contacto COLLATE Latin1_General_BIN <> d.motivo COLLATE Latin1_General_BIN;
SET @af = @@ROWCOUNT;
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R03 Etiquetas', 'motivo_contacto', 'Normalizar mayusculas al catalogo', @af, @tot, 'COBRO/Cobro, QUEJA/Queja');

/* R03b Canal sin dato: se conserva como categoria "Sin dato" */
SET @tot = (SELECT COUNT(*) FROM wrk_c15);
UPDATE wrk_c15 SET canal = 'Sin dato'
WHERE canal = '(sin dato)' OR canal IS NULL OR LTRIM(RTRIM(canal)) = '';
SET @af = @@ROWCOUNT;
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R03b Completitud canal', 'canal', 'Marcar como categoria Sin dato (no recuperable)', @af, @tot, 'Canal ausente');

/* R04 Validez de rango */
UPDATE wrk_c15
SET duracion_seg = TRY_CONVERT(INT, dur_txt), espera_seg = TRY_CONVERT(INT, esp_txt),
    transferencias = TRY_CONVERT(INT, tra_txt), casos_previos_30d = TRY_CONVERT(INT, cas_txt);

SET @tot = (SELECT COUNT(*) FROM wrk_c15);
SELECT @a = SUM(CASE WHEN duracion_seg NOT BETWEEN 30 AND 1800 THEN 1 ELSE 0 END),
       @b = SUM(CASE WHEN espera_seg NOT BETWEEN 0 AND 900 THEN 1 ELSE 0 END),
       @c = SUM(CASE WHEN transferencias NOT BETWEEN 0 AND 4 THEN 1 ELSE 0 END),
       @d = SUM(CASE WHEN casos_previos_30d NOT BETWEEN 0 AND 8 THEN 1 ELSE 0 END)
FROM wrk_c15;

INSERT INTO rechazados_c15
SELECT id_stg, id_interaccion, 'R04 Fuera de rango',
       CONCAT('dur=', dur_txt, ' esp=', esp_txt, ' trans=', tra_txt, ' casos=', cas_txt)
FROM wrk_c15
WHERE duracion_seg NOT BETWEEN 30 AND 1800 OR espera_seg NOT BETWEEN 0 AND 900
   OR transferencias NOT BETWEEN 0 AND 4 OR casos_previos_30d NOT BETWEEN 0 AND 8;
DELETE FROM wrk_c15
WHERE duracion_seg NOT BETWEEN 30 AND 1800 OR espera_seg NOT BETWEEN 0 AND 900
   OR transferencias NOT BETWEEN 0 AND 4 OR casos_previos_30d NOT BETWEEN 0 AND 8;

INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle) VALUES
('R04 Rango', 'duracion_seg', 'Rechazar registro (30-1800)', ISNULL(@a,0), @tot, 'Fuera de rango'),
('R04 Rango', 'espera_seg', 'Rechazar registro (0-900)', ISNULL(@b,0), @tot, 'Fuera de rango'),
('R04 Rango', 'transferencias', 'Rechazar registro (0-4)', ISNULL(@c,0), @tot, 'Fuera de rango'),
('R04 Rango', 'casos_previos_30d', 'Rechazar registro (0-8)', ISNULL(@d,0), @tot, 'Fuera de rango');

/* R05 Completitud: recuperar lo posible */
SET @tot = (SELECT COUNT(*) FROM wrk_c15);
SET @antes = (SELECT COUNT(*) FROM wrk_c15 WHERE tipo_usuario IS NULL);
UPDATE w SET tipo_usuario = x.tipo
FROM wrk_c15 w
CROSS APPLY (SELECT TOP 1 w2.tipo_usuario AS tipo FROM wrk_c15 w2
             WHERE w2.id_usuario_seud = w.id_usuario_seud AND w2.tipo_usuario IS NOT NULL
             GROUP BY w2.tipo_usuario ORDER BY COUNT(*) DESC) x
WHERE w.tipo_usuario IS NULL;
SET @desp = (SELECT COUNT(*) FROM wrk_c15 WHERE tipo_usuario IS NULL);
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R05a Completitud', 'tipo_usuario', 'Completar con el tipo del mismo usuario', @antes - @desp, @tot, 'Faltantes: ' + CAST(@antes AS VARCHAR(10)));

SET @antes = (SELECT COUNT(*) FROM wrk_c15 WHERE turno IS NULL);
UPDATE wrk_c15 SET turno = 'Fin de semana'
WHERE turno IS NULL AND (DATEDIFF(DAY, '19000101', fecha_contacto) % 7) IN (5, 6);
SET @desp = (SELECT COUNT(*) FROM wrk_c15 WHERE turno IS NULL);
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R05b Completitud', 'turno', 'Deducir Fin de semana por la fecha', @antes - @desp, @tot, 'Faltantes: ' + CAST(@antes AS VARCHAR(10)));

INSERT INTO rechazados_c15
SELECT id_stg, id_interaccion, 'R05 Faltante no recuperable',
       CONCAT('cola=', ISNULL(cola_servicio,'NULL'), ' tipo=', ISNULL(tipo_usuario,'NULL'),
              ' turno=', ISNULL(turno,'NULL'), ' dur=', ISNULL(CAST(duracion_seg AS VARCHAR(10)),'NULL'))
FROM wrk_c15
WHERE cola_servicio IS NULL OR tipo_usuario IS NULL OR turno IS NULL OR motivo_contacto IS NULL
   OR duracion_seg IS NULL OR espera_seg IS NULL OR transferencias IS NULL OR casos_previos_30d IS NULL;
SET @af = @@ROWCOUNT;
DELETE FROM wrk_c15
WHERE cola_servicio IS NULL OR tipo_usuario IS NULL OR turno IS NULL OR motivo_contacto IS NULL
   OR duracion_seg IS NULL OR espera_seg IS NULL OR transferencias IS NULL OR casos_previos_30d IS NULL;
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R05c Completitud', 'cola/tipo/turno/duracion', 'Rechazar (no recuperable)', @af, @tot, 'Sin forma de reconstruir el dato');

/* R06 Resultado no consolidado: se conserva como NULL */
UPDATE wrk_c15 SET recontacto_7_dias = CASE WHEN rec_txt IN ('0','1') THEN CAST(rec_txt AS TINYINT) END;
SET @tot = (SELECT COUNT(*) FROM wrk_c15);
SET @af  = (SELECT COUNT(*) FROM wrk_c15 WHERE recontacto_7_dias IS NULL);
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R06 Resultado', 'recontacto_7_dias', 'Conservar NULL (no consolidado); excluir de datasets de minería', @af, @tot, 'Variable objetivo vacia');

/* R07 Oportunidad: carga tardía (solo se marca, no se elimina) */
SET @af = (SELECT COUNT(*) FROM wrk_c15 WHERE DATEDIFF(DAY, fecha_contacto, fecha_carga) > 30);
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R07 Oportunidad', 'fecha_carga', 'Solo medir (carga > 30 dias tras el contacto)', @af, @tot, 'Registros cargados tarde');

/* R08 Catálogos: valores que no existen en las dimensiones */
INSERT INTO rechazados_c15
SELECT w.id_stg, w.id_interaccion, 'R08 Fuera de catalogo',
       CONCAT('canal=', w.canal, ' cola=', w.cola_servicio, ' motivo=', w.motivo_contacto, ' fuente=', w.sistema_origen)
FROM wrk_c15 w
WHERE NOT EXISTS (SELECT 1 FROM dim_canal x WHERE x.canal = w.canal)
   OR NOT EXISTS (SELECT 1 FROM dim_cola x WHERE x.cola = w.cola_servicio)
   OR NOT EXISTS (SELECT 1 FROM dim_motivo x WHERE x.motivo = w.motivo_contacto)
   OR NOT EXISTS (SELECT 1 FROM dim_tipo_usuario x WHERE x.tipo_usuario = w.tipo_usuario)
   OR NOT EXISTS (SELECT 1 FROM dim_turno x WHERE x.turno = w.turno)
   OR NOT EXISTS (SELECT 1 FROM dim_fuente x WHERE x.fuente = w.sistema_origen);
SET @af = @@ROWCOUNT;
DELETE w FROM wrk_c15 w
WHERE NOT EXISTS (SELECT 1 FROM dim_canal x WHERE x.canal = w.canal)
   OR NOT EXISTS (SELECT 1 FROM dim_cola x WHERE x.cola = w.cola_servicio)
   OR NOT EXISTS (SELECT 1 FROM dim_motivo x WHERE x.motivo = w.motivo_contacto)
   OR NOT EXISTS (SELECT 1 FROM dim_tipo_usuario x WHERE x.tipo_usuario = w.tipo_usuario)
   OR NOT EXISTS (SELECT 1 FROM dim_turno x WHERE x.turno = w.turno)
   OR NOT EXISTS (SELECT 1 FROM dim_fuente x WHERE x.fuente = w.sistema_origen);
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R08 Catalogos', '(dimensiones)', 'Rechazar valores fuera de catalogo', @af, @tot, 'Integridad referencial');

/* R09 Carga al DW */
INSERT INTO hecho_interaccion
 (codigo_interaccion, id_usuario_seud, fecha_contacto, id_canal, id_cola, id_motivo, id_tipo_usuario,
  id_turno, id_fuente, duracion_seg, espera_seg, transferencias, casos_previos_30d,
  recontacto_7_dias, grabacion_autorizada, fecha_carga)
SELECT w.id_interaccion, w.id_usuario_seud, w.fecha_contacto, c.id_canal, co.id_cola, m.id_motivo,
       t.id_tipo_usuario, tu.id_turno, f.id_fuente, w.duracion_seg, w.espera_seg, w.transferencias,
       w.casos_previos_30d, w.recontacto_7_dias, w.grabacion_autorizada, w.fecha_carga
FROM wrk_c15 w
JOIN dim_canal c ON c.canal = w.canal
JOIN dim_cola co ON co.cola = w.cola_servicio
JOIN dim_motivo m ON m.motivo = w.motivo_contacto
JOIN dim_tipo_usuario t ON t.tipo_usuario = w.tipo_usuario
JOIN dim_turno tu ON tu.turno = w.turno
JOIN dim_fuente f ON f.fuente = w.sistema_origen;
SET @af = @@ROWCOUNT;
INSERT INTO log_limpieza (regla, campo, accion, registros_afectados, registros_total, detalle)
VALUES ('R09 Carga DW', 'hecho_interaccion', 'Insertar registros limpios', @af, @tot, 'Resultado final');

/* ---------- RESULTADOS ---------- */
SELECT id_log, regla, campo, accion, registros_afectados, registros_total, detalle FROM log_limpieza ORDER BY id_log;
SELECT regla, COUNT(*) AS rechazados FROM rechazados_c15 GROUP BY regla ORDER BY regla;
SELECT (SELECT COUNT(*) FROM stg_c15_v0) AS recibidos,
       (SELECT registros_afectados FROM log_limpieza WHERE regla = 'R01 Unicidad') AS duplicados_eliminados,
       (SELECT COUNT(*) FROM rechazados_c15) AS rechazados,
       (SELECT COUNT(*) FROM hecho_interaccion) AS cargados;



