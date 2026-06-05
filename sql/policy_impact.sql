-- Policy impact example
SELECT forecast_model, lead_time, AVG(stockout_flag) AS stockout_rate, AVG(total_cost) AS avg_total_cost
FROM inventory_policy_summary
GROUP BY forecast_model, lead_time;
