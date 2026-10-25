-- Рабочий пример GROUP BY и LEFT JOIN. Добавьте оконный запрос по условию DOCX.
SELECT o.city, COUNT(*) AS n, AVG(o.late) AS late_rate,
       SUM(CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END) AS unmatched
FROM orders o LEFT JOIN customers c ON o.customer_id=c.customer_id
GROUP BY o.city ORDER BY o.city;
