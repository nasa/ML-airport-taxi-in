WITH mf AS 
	(
	SELECT DISTINCT ON (gufi) 
		gufi,
		arrival_aerodrome_icao_name
	FROM 
		matm_flight_summary 
	WHERE 
		arrival_aerodrome_icao_name = :airport_icao 
		and arrival_runway_actual_time between :start_time and :end_time
),
tfm_flight AS (
	SELECT DISTINCT ON (gufi) 
	    gufi, 
	    message_type, 
	    "timestamp", 
	    arrival_runway_airline_time, 
	    arrival_stand_airline_time,
	    EXTRACT (EPOCH FROM arrival_stand_airline_time - arrival_runway_airline_time) AS scheduled_taxi_in_prediction_seconds
	FROM 
		tfm_extension_all
	--NOTE: Starting 24 hours prior to the start of the operational day being looked at
	--Flight creates can happen up to 24 hours in advance
	WHERE 
	    "timestamp" >= (:start_time - interval '1 Day')
	    AND "timestamp" <= :end_time
	    AND arrival_runway_airline_time >= :start_time
	    AND arrival_runway_airline_time <= :end_time
		AND arrival_stand_airline_time is not null
	ORDER BY gufi, "timestamp"
)
SELECT 
	tfm_flight.gufi,
	tfm_flight.message_type,
	tfm_flight.timestamp as source_timestamp,
	tfm_flight.arrival_runway_airline_time as airline_on_time,
	tfm_flight.arrival_stand_airline_time as airline_in_time,
	tfm_flight.scheduled_taxi_in_prediction_seconds
FROM 
	tfm_flight inner join mf on mf.gufi=tfm_flight.gufi
ORDER BY 
	tfm_flight.gufi, tfm_flight.timestamp
