CREATE DATABASE IF NOT EXISTS industrial_sql;

USE industrial_sql;

CREATE TABLE IF NOT EXISTS equipment (
    equipment_id INT PRIMARY KEY,
    equipment_name VARCHAR(50) NOT NULL,
    equipment_type VARCHAR(30) NOT NULL,
    location VARCHAR(50),
    criticality VARCHAR(10),
    installation_date DATE
);


CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    reading_time DATETIME NOT NULL,
    temperature_c DECIMAL(6,2),
    vibration_mm_s DECIMAL(6,2),

    FOREIGN KEY (equipment_id)
        REFERENCES equipment(equipment_id)
);

CREATE TABLE IF NOT EXISTS maintenance_events (
    event_id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    maintenance_date DATE NOT NULL,
    maintenance_type VARCHAR(30),
    maintenance_cost DECIMAL(10,2),

    FOREIGN KEY (equipment_id)
        REFERENCES equipment(equipment_id)
);

INSERT INTO equipment (
    equipment_id,
    equipment_name,
    equipment_type,
    location,
    criticality,
    installation_date
)
VALUES
(1, 'Motor A', 'Motor', 'Line 1', 'High', '2022-03-10'),
(2, 'Pump B', 'Pump', 'Line 1', 'Medium', '2021-06-05'),
(3, 'Robot C', 'Robot', 'Line 2', 'High', '2023-11-20'),
(4, 'Compressor D', 'Compressor', 'Utilities', 'High', '2020-09-12');

INSERT INTO sensor_readings (
    equipment_id,
    reading_time,
    temperature_c,
    vibration_mm_s
)
VALUES
(1, '2026-11-01 08:00:00', 72.5, 2.1),
(1, '2026-11-01 09:00:00', 82.3, 4.1),
(1, '2026-11-01 10:00:00', 86.0, 5.2),

(2, '2026-11-01 08:00:00', 60.2, 1.2),
(2, '2026-11-01 09:00:00', 65.0, 1.5),

(3, '2026-11-01 08:00:00', 48.0, 0.8),
(3, '2026-11-01 09:00:00', 50.1, 1.2),
(3, '2026-11-01 10:00:00', 54.2, 1.5),

(4, '2026-11-01 08:00:00', 89.0, 6.0),
(4, '2026-11-01 09:00:00', 94.0, 6.5);

INSERT INTO maintenance_events (
    equipment_id,
    maintenance_date,
    maintenance_type,
    maintenance_cost
)
VALUES
(1, '2026-10-20', 'Preventive', 2500),
(2, '2026-10-15', 'Inspection', 300),
(3, '2026-10-29', 'Calibration', 500),
(4, '2026-10-27', 'Corrective', 3500);

