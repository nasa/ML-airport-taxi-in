select
        mall.gufi as gufi,
        AVG((mall.position_speed >= 4)::int::float4) as fraction_speed_gte_threshold
from matm_position_all mall
inner join flight_summary_kclt_v2_2 flsum on mall.gufi = flsum.gufi
where 
      flsum.isArrival
      flsum.arrival_stand_actual_time between '2019-10-01 08:00' and '2019-10-01 10:00'
      and mall.system_id = 'ASDEX'
      and mall.timestamp > '2019-10-01 08:00'
      and mall.timestamp <  ('2019-10-01 10:00'::timestamp + '1 hour'::interval)
      and mall.timestamp > flsum.arrival_runway_actual_time
group by mall.gufi
