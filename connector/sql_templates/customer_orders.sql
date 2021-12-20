SELECT store_type_l2,
        ROUND(delivery_location.latitude, @round_off) lat,
        ROUND(delivery_location.longitude, @round_off) long,
        COUNT(distinct analytical_customer_id) num_customers,
        COUNT(1) num_orders,
        SUM(i.quantity) num_items,
        SUM(value.gmv_local) gmv
        FROM `fulfillment-dwh-production.curated_data_shared_central_dwh.orders` o, UNNEST(items) AS i
        LEFT JOIN `fulfillment-dwh-production.curated_data_shared_central_dwh.vendors` USING(vendor_id)
        WHERE o.global_entity_id = @global_entity_id
--        AND o.city_id = @city_id
--        AND o.country_code = @country_code
        AND DATE(placed_at) BETWEEN @date_from AND @date_to
        AND order_status = 'sent'
        AND o.is_own_delivery
        AND store_type_l2 in ('restaurants')
        AND delivery_location.latitude IS NOT NULL
        AND delivery_location.longitude IS NOT NULL
        GROUP BY 1, 2, 3