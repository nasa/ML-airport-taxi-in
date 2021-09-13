select 
    gufi,
    isArrival,
    isDeparture,

    acid,
    departure_stand_initial_time,
    departure_aerodrome_icao_name,
    arrival_aerodrome_icao_name,

    undelayed_arrival_ramp_transit_time,
    undelayed_arrival_ama_transit_time,
    undelayed_arrival_total_transit_time

from flight_summary_kclt_v3_1
where 
(
    isDeparture
    and
    departure_runway_actual_time between :start_time and :end_time
)
or 
(
    isArrival
    and
    arrival_stand_actual_time between :start_time and :end_time
)