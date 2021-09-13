select 
    gufi,
    isArrival,
    isDeparture,

    departure_stand_actual_time,
    departure_stand_scheduled_time,
    departure_runway_actual_time,
    departure_runway_undelayed_time_at_out,
    departure_runway_targeted_time_at_out,
    departure_runway_metered_time_at_out,
    departure_movement_area_actual_time,
    departure_movement_area_undelayed_time_at_out,
    departure_stand_actual,
    
    arrival_stand_actual_time,
    arrival_stand_scheduled_time,
    arrival_runway_actual_time,
    arrival_stand_undelayed_time_at_landing,
    arrival_stand_undelayed_time_at_spot,
    arrival_movement_area_actual_time,
    arrival_movement_area_undelayed_time_at_landing,
    arrival_stand_predicted_at_landing,
    arrival_stand_predicted_at_landing_source,
    arrival_stand_actual,

    metered_indicator,
    edct_at_ready,
    apreq_at_ready,
    actual_gate_hold,

    departure_runway_actual,
    arrival_runway_actual,

    gate_conflict_values_present,
    first_gate_conflict_report,
    last_gate_conflict_report,
    gate_conflict_at_landing,
    start_of_expected_gate_conflict_at_landing,
    end_of_expected_gate_conflict_at_landing,
    gate_conflict_duration_at_landing,
    associated_gate_conflict_gufi,
    associated_gate_conflict_gate,

    flow_at_out,
    flow_at_off,
    flow_at_in,
    flow_at_on,

    actual_arrival_ramp_taxi_time,
    excess_arrival_ramp_taxi_time,
    actual_arrival_ama_taxi_time,
    excess_arrival_ama_taxi_time,

    arrival_stand_actual,
    arrival_stand_ramp_area,
    arrival_terminal,
    arrival_spot_actual,
    departure_stand_actual,
    departure_stand_ramp_area,
    departure_terminal,
    departure_spot_actual,

    aircraft_type,
    aircraft_engine_class,
    wake_turbulence_category,
    carrier,
    major

from flight_summary_kclt_v3_0
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