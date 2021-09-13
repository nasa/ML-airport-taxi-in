select
        mall.gufi as gufi,
        AVG((mall.position_speed >= :threshold_knots)::int::float4) as fraction_speed_gte_threshold
from matm_position_all mall
inner join matm_flight_summary msum on mall.gufi = msum.gufi
where 
      msum.arrival_aerodrome_icao_name = :airport_icao
      and (msum.arrival_stand_actual_time between :start_time and :end_time)
      and mall.system_id = 'ASDEX'
      and mall.timestamp > (:start_time - '1 hour'::interval)
      and mall.timestamp < (:end_time + '1 hour'::interval)
      and mall.timestamp > msum.arrival_runway_actual_time
group by mall.gufi
