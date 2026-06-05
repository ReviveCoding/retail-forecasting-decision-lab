-- Demand mart example
SELECT date, state_id, store_id, cat_id, SUM(demand) AS total_demand
FROM demand_panel
GROUP BY date, state_id, store_id, cat_id;
