select
        asdxt.gufi as gufi,
        AVG((asdxt.last_asdex_position_speed >= :threshold_knots)::int::float4) as fraction_speed_gte_threshold
from asdex_extension asdxt
inner join matm_flight_summary msum on asdxt.gufi = msum.gufi
where 
      msum.arrival_aerodrome_icao_name = :airport_icao
      and (msum.arrival_stand_actual_time between :start_time and :end_time)
      and asdxt.timestamp > (:start_time - '1 hour'::interval)
      and asdxt.timestamp < (:end_time + '1 hour'::interval)
      and asdxt.timestamp > msum.arrival_runway_actual_time
group by asdxt.gufi
