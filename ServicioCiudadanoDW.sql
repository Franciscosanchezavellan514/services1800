/* ============================================================
   DW Caso 15 - Servicio Ciudadano 1800 (Integrador VIII)
   Hecho: interaccion | Periodo: 2025-01-01 a 2026-06-30
   ============================================================ */
USE master;
GO
IF DB_ID('ServicioCiudadanoDW') IS NULL
    CREATE DATABASE ServicioCiudadanoDW;
GO
USE ServicioCiudadanoDW;
GO
SET DATEFIRST 1;  -- lunes = 1
GO

/* ---------- DIMENSIONES ---------- */
CREATE TABLE dim_tiempo (
    fecha          DATE PRIMARY KEY,
    anio           SMALLINT    NOT NULL,
    mes            TINYINT     NOT NULL,
    nombre_mes     VARCHAR(15) NOT NULL,
    trimestre      TINYINT     NOT NULL,
    dia_semana     TINYINT     NOT NULL,
    nombre_dia     VARCHAR(15) NOT NULL,
    es_fin_semana  BIT         NOT NULL
);

CREATE TABLE dim_canal (
    id_canal   INT IDENTITY(1,1) PRIMARY KEY,
    canal      VARCHAR(30) NOT NULL UNIQUE
);

CREATE TABLE dim_cola (
    id_cola    INT IDENTITY(1,1) PRIMARY KEY,
    cola       VARCHAR(30) NOT NULL UNIQUE
);

CREATE TABLE dim_motivo (
    id_motivo  INT IDENTITY(1,1) PRIMARY KEY,
    motivo     VARCHAR(30) NOT NULL UNIQUE
);

CREATE TABLE dim_tipo_usuario (
    id_tipo_usuario INT IDENTITY(1,1) PRIMARY KEY,
    tipo_usuario    VARCHAR(30) NOT NULL UNIQUE
);

CREATE TABLE dim_turno (
    id_turno   INT IDENTITY(1,1) PRIMARY KEY,
    turno      VARCHAR(30) NOT NULL UNIQUE
);

CREATE TABLE dim_fuente (
    id_fuente      INT IDENTITY(1,1) PRIMARY KEY,
    fuente         VARCHAR(30) NOT NULL UNIQUE,
    responsable    VARCHAR(60) NOT NULL,
    formato        VARCHAR(40) NOT NULL,
    actualizacion  VARCHAR(40) NOT NULL
);

/* ---------- HECHO ---------- */
CREATE TABLE hecho_interaccion (
    id_interaccion       INT IDENTITY(1,1) PRIMARY KEY,
    codigo_interaccion   VARCHAR(30) NOT NULL UNIQUE,   -- id seudonimizado
    id_usuario_seud      VARCHAR(30) NOT NULL,          -- dimension degenerada
    fecha_contacto       DATE NOT NULL,
    id_canal             INT  NOT NULL,
    id_cola              INT  NOT NULL,
    id_motivo            INT  NOT NULL,
    id_tipo_usuario      INT  NOT NULL,
    id_turno             INT  NOT NULL,
    id_fuente            INT  NOT NULL,
    duracion_seg         INT  NOT NULL CHECK (duracion_seg BETWEEN 30 AND 1800),
    espera_seg           INT  NOT NULL CHECK (espera_seg BETWEEN 0 AND 900),
    transferencias       INT  NOT NULL CHECK (transferencias BETWEEN 0 AND 4),
    casos_previos_30d    INT  NOT NULL CHECK (casos_previos_30d BETWEEN 0 AND 8),
    recontacto_7_dias    TINYINT NULL CHECK (recontacto_7_dias IN (0,1)),  -- NULL = no consolidado
    grabacion_autorizada VARCHAR(10) NULL,
    fecha_carga          DATE NOT NULL,
    CONSTRAINT FK_int_tiempo  FOREIGN KEY (fecha_contacto)  REFERENCES dim_tiempo(fecha),
    CONSTRAINT FK_int_canal   FOREIGN KEY (id_canal)        REFERENCES dim_canal(id_canal),
    CONSTRAINT FK_int_cola    FOREIGN KEY (id_cola)         REFERENCES dim_cola(id_cola),
    CONSTRAINT FK_int_motivo  FOREIGN KEY (id_motivo)       REFERENCES dim_motivo(id_motivo),
    CONSTRAINT FK_int_tipo    FOREIGN KEY (id_tipo_usuario) REFERENCES dim_tipo_usuario(id_tipo_usuario),
    CONSTRAINT FK_int_turno   FOREIGN KEY (id_turno)        REFERENCES dim_turno(id_turno),
    CONSTRAINT FK_int_fuente  FOREIGN KEY (id_fuente)       REFERENCES dim_fuente(id_fuente)
);

/* ---------- CALIDAD (evidencia para Eddy) ---------- */
-- Copia "sucia" del v0: todo como texto, tal como llega de la fuente
CREATE TABLE stg_c15_v0 (
    id_stg               INT IDENTITY(1,1) PRIMARY KEY,
    id_interaccion       NVARCHAR(50)  NULL,
    id_usuario_seud      NVARCHAR(50)  NULL,
    fecha_contacto       NVARCHAR(30)  NULL,
    cola_servicio        NVARCHAR(50)  NULL,
    canal                NVARCHAR(50)  NULL,
    tipo_usuario         NVARCHAR(50)  NULL,
    motivo_contacto      NVARCHAR(50)  NULL,
    duracion_seg         NVARCHAR(30)  NULL,
    espera_seg           NVARCHAR(30)  NULL,
    transferencias       NVARCHAR(30)  NULL,
    casos_previos_30d    NVARCHAR(30)  NULL,
    turno                NVARCHAR(50)  NULL,
    sistema_origen       NVARCHAR(50)  NULL,
    recontacto_7_dias    NVARCHAR(30)  NULL,
    grabacion_autorizada NVARCHAR(30)  NULL,
    fecha_carga          NVARCHAR(30)  NULL
);

-- Bitacora de limpieza: una fila por regla aplicada
CREATE TABLE log_limpieza (
    id_log              INT IDENTITY(1,1) PRIMARY KEY,
    fecha_ejecucion     DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
    regla               VARCHAR(100) NOT NULL,
    campo               VARCHAR(50)  NOT NULL,
    accion              VARCHAR(60)  NOT NULL,
    registros_afectados INT NOT NULL,
    registros_total     INT NOT NULL,
    detalle             VARCHAR(300) NULL
);
GO

/* ---------- DATOS DE DIMENSIONES ESTATICAS ---------- */
INSERT INTO dim_canal (canal) VALUES
 ('Teléfono'),('Chat'),('Correo'),('Red social'),('Sin dato');
INSERT INTO dim_cola (cola) VALUES
 ('Facturación'),('Soporte'),('Información'),('Reclamos'),('Trámites');
INSERT INTO dim_motivo (motivo) VALUES
 ('Consulta'),('Falla'),('Cobro'),('Solicitud'),('Queja');
INSERT INTO dim_tipo_usuario (tipo_usuario) VALUES
 ('Nuevo'),('Recurrente'),('Empresa'),('Adulto mayor');
INSERT INTO dim_turno (turno) VALUES
 ('AM'),('PM'),('Nocturno'),('Fin de semana');
INSERT INTO dim_fuente (fuente, responsable, formato, actualizacion) VALUES
 ('ACD','Operaciones','Base SQL / vista','Diaria'),
 ('CRM','Gerencia de Servicio','API / eventos','Casi en tiempo real'),
 ('Ticketing','Operaciones','Archivo XLSX','Semanal'),
 ('Calidad','Calidad','CSV exportado','Mensual');

/* dim_tiempo: 2025-01-01 a 2026-06-30 (546 dias) */
;WITH d AS (
    SELECT CAST('2025-01-01' AS DATE) AS f
    UNION ALL
    SELECT DATEADD(DAY,1,f) FROM d WHERE f < '2026-06-30'
)
INSERT INTO dim_tiempo (fecha, anio, mes, nombre_mes, trimestre, dia_semana, nombre_dia, es_fin_semana)
SELECT f, YEAR(f), MONTH(f), DATENAME(MONTH,f), DATEPART(QUARTER,f),
       DATEPART(WEEKDAY,f), DATENAME(WEEKDAY,f),
       CASE WHEN DATEPART(WEEKDAY,f) >= 6 THEN 1 ELSE 0 END
FROM d OPTION (MAXRECURSION 1000);
GO

/* ---------- INDICES ---------- */
CREATE INDEX IX_int_usuario_fecha ON hecho_interaccion(id_usuario_seud, fecha_contacto);
CREATE INDEX IX_int_fecha         ON hecho_interaccion(fecha_contacto);
CREATE INDEX IX_int_canal         ON hecho_interaccion(id_canal);
CREATE INDEX IX_int_motivo        ON hecho_interaccion(id_motivo);
CREATE INDEX IX_int_cola          ON hecho_interaccion(id_cola);
CREATE INDEX IX_int_recontacto    ON hecho_interaccion(recontacto_7_dias);
GO

/* ---------- VERIFICACION ---------- */
SELECT 'dim_tiempo' tabla, COUNT(*) filas FROM dim_tiempo UNION ALL
SELECT 'dim_canal', COUNT(*) FROM dim_canal UNION ALL
SELECT 'dim_cola', COUNT(*) FROM dim_cola UNION ALL
SELECT 'dim_motivo', COUNT(*) FROM dim_motivo UNION ALL
SELECT 'dim_tipo_usuario', COUNT(*) FROM dim_tipo_usuario UNION ALL
SELECT 'dim_turno', COUNT(*) FROM dim_turno UNION ALL
SELECT 'dim_fuente', COUNT(*) FROM dim_fuente;