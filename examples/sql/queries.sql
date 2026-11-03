SELECT * 
FROM equipment;

SELECT equipment_name, criticality 
FROM equipment
WHERE criticality = 'High'; # finding high criticality equipment for maintenance


SELECT equipment_id, temperature_c
FROM sensor_readings
ORDER BY temperature_c DESC
LIMIT 3;

SELECT equipment_id, temperature_c, vibration_mm_s
FROM sensor_readings
WHERE temperature_c > 80 AND vibration_mm_s > 4; # assume temperature over 80 and vibration over 4 anomaly 


SELECT equipment_id, temperature_c,
    CASE
        WHEN temperature_c >= 85 THEN 'Alert'
        ELSE 'Normal'
    END AS temperature_status
FROM sensor_readings
ORDER BY temperature_c DESC;  # creating alert for temperature

SELECT equipment_id, vibration_mm_s,
    CASE
        WHEN vibration_mm_s >= 4 THEN 'Alert'
        ELSE 'Normal'
    END AS vibration_status
FROM sensor_readings
ORDER BY vibration_mm_s DESC; # creating alert for vibration
