WITH flights AS
    (
        SELECT gufi, arrival_runway_actual_time
        FROM matm_flight_summary
        -- The start and end time of the operational day being looked at
        WHERE arrival_runway_actual_time >= :start_time
          AND arrival_runway_actual_time <= :end_time
          AND arrival_aerodrome_icao_name = :airport_icao
    ),
tfm_data AS
    (
        SELECT *
        FROM tfm_extension_all
        -- NOTE: This timestamp is 24 hours prior to the start of the
        --       day in question. That allows the query to pick up
        --       data in case the airline provided times 12 hours or more
        --       in advance
        WHERE timestamp >= (:start_time - interval '1 day')
          AND timestamp <= :end_time
    )
SELECT 
    DISTINCT ON (f.gufi)
    f.gufi,
    f.arrival_runway_actual_time,
    t.timestamp,
    t.arrival_runway_airline_time as arrival_runway_airline_time_at_landing,
    t.arrival_stand_airline_time as arrival_stand_airline_time_at_landing,
    EXTRACT (EPOCH FROM t.arrival_stand_airline_time - t.arrival_runway_airline_time) AS airline_taxi_in_prediction_seconds
FROM flights f
JOIN tfm_data t
ON t.gufi = f.gufi AND t.record_timestamp < f.arrival_runway_actual_time
ORDER BY f.gufi, t.record_timestamp DESC