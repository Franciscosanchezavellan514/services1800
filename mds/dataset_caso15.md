USE ServicioCiudadanoDW;
GO
DROP TABLE IF EXISTS dbo.dm_dataset_recontacto;
GO

SELECT
    -- === IDENTIFICADOR (no entra al modelo) ===
    h.id_interaccion,

    -- === VARIABLE TARGET (Y) ===
    h.recontacto_7_dias AS es_recontacto,

    -- === GRUPO 1: MEDIDAS OPERATIVAS (conocidas al cerrar la interaccion) ===
    h.duracion_seg,
    h.espera_seg,
    h.transferencias,
    h.casos_previos_30d,          -- historial: solo contactos ANTERIORES (sin fuga)

    -- === GRUPO 2: BANDERAS DERIVADAS ===
    CASE WHEN h.duracion_seg < 400  THEN 1 ELSE 0 END AS es_duracion_corta,
    CASE WHEN h.espera_seg   > 400  THEN 1 ELSE 0 END AS es_espera_alta,
    CASE WHEN h.transferencias >= 2 THEN 1 ELSE 0 END AS es_multi_transferencia,
    CASE WHEN h.casos_previos_30d > 0 THEN 1 ELSE 0 END AS tiene_historial,

    -- === GRUPO 3: CANAL ===
    CASE WHEN ca.canal = 'Chat'       THEN 1 ELSE 0 END AS canal_chat,
    CASE WHEN ca.canal = 'Correo'     THEN 1 ELSE 0 END AS canal_correo,
    CASE WHEN ca.canal = 'Red social' THEN 1 ELSE 0 END AS canal_red_social,
    CASE WHEN ca.canal = 'Teléfono'   THEN 1 ELSE 0 END AS canal_telefono,
    CASE WHEN ca.canal = 'Sin dato'   THEN 1 ELSE 0 END AS canal_sin_dato,

    -- === GRUPO 4: MOTIVO ===
    CASE WHEN m.motivo = 'Consulta'  THEN 1 ELSE 0 END AS motivo_consulta,
    CASE WHEN m.motivo = 'Falla'     THEN 1 ELSE 0 END AS motivo_falla,
    CASE WHEN m.motivo = 'Cobro'     THEN 1 ELSE 0 END AS motivo_cobro,
    CASE WHEN m.motivo = 'Queja'     THEN 1 ELSE 0 END AS motivo_queja,
    CASE WHEN m.motivo = 'Solicitud' THEN 1 ELSE 0 END AS motivo_solicitud,

    -- === GRUPO 5: COLA ===
    CASE WHEN co.cola = 'Facturación' THEN 1 ELSE 0 END AS cola_facturacion,
    CASE WHEN co.cola = 'Soporte'     THEN 1 ELSE 0 END AS cola_soporte,
    CASE WHEN co.cola = 'Información' THEN 1 ELSE 0 END AS cola_informacion,
    CASE WHEN co.cola = 'Reclamos'    THEN 1 ELSE 0 END AS cola_reclamos,
    CASE WHEN co.cola = 'Trámites'    THEN 1 ELSE 0 END AS cola_tramites,

    -- === GRUPO 6: TIPO DE USUARIO ===
    CASE WHEN t.tipo_usuario = 'Nuevo'        THEN 1 ELSE 0 END AS usuario_nuevo,
    CASE WHEN t.tipo_usuario = 'Recurrente'   THEN 1 ELSE 0 END AS usuario_recurrente,
    CASE WHEN t.tipo_usuario = 'Empresa'      THEN 1 ELSE 0 END AS usuario_empresa,
    CASE WHEN t.tipo_usuario = 'Adulto mayor' THEN 1 ELSE 0 END AS usuario_adulto_mayor,

    -- === GRUPO 7: TURNO ===
    CASE WHEN tu.turno = 'AM'            THEN 1 ELSE 0 END AS turno_am,
    CASE WHEN tu.turno = 'PM'            THEN 1 ELSE 0 END AS turno_pm,
    CASE WHEN tu.turno = 'Nocturno'      THEN 1 ELSE 0 END AS turno_nocturno,
    CASE WHEN tu.turno = 'Fin de semana' THEN 1 ELSE 0 END AS turno_fin_semana,

    -- === VARIABLE DE CONTROL (no entra al modelo) ===
    h.fecha_contacto

INTO dbo.dm_dataset_recontacto
FROM hecho_interaccion h
    INNER JOIN dim_canal ca        ON h.id_canal = ca.id_canal
    INNER JOIN dim_motivo m        ON h.id_motivo = m.id_motivo
    INNER JOIN dim_cola co         ON h.id_cola = co.id_cola
    INNER JOIN dim_tipo_usuario t  ON h.id_tipo_usuario = t.id_tipo_usuario
    INNER JOIN dim_turno tu        ON h.id_turno = tu.id_turno
WHERE h.recontacto_7_dias IS NOT NULL;   -- solo interacciones con resultado consolidado
GO

-- VERIFICACION
SELECT COUNT(*) AS filas,
       SUM(CAST(es_recontacto AS INT)) AS recontactos,
       CAST(AVG(CAST(es_recontacto AS FLOAT)) * 100 AS DECIMAL(5,2)) AS pct_recontacto
FROM dbo.dm_dataset_recontacto;