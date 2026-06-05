-- Forecast error slice example
SELECT cat_id, store_id, AVG(pred_corrected - demand) AS mean_bias
FROM forecast_output
GROUP BY cat_id, store_id;
