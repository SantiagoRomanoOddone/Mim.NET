events.csv: contiene los eventos realizados por el usuario durante su navegacion en el sitio web. Un evento es una accion realizada en un momento especifico.

Columnas:
user_id: identificador unico del usuario
event_type: tipo de evento realizado (Page, Product Viewed, Product Added, etc)
event_id: identificador unico del evento
ts: fecha y hora del evento en UTC

order_completed.csv: contiene las compras realizadas por los usuarios. Se puede unir con events.csv a traves de events.event_id = order_completed.id

Columnas:
id: identificador unico de la orden
timestamp: fecha y hora de la orden en UTC
total_price: precio total de la orden en dolares
discount_codes: json con codigos de descuento aplicados
total_discounts: suma de los descuentos aplicados en dolares
shipping_country: pais de envio
variant_item_1: identificador del item 1 del carrito
variant_item_2: identificador del item 2 del carrito
variant_item_3: identificador del item 3 del carrito
variant_item_4: identificador del item 4 del carrito
variant_item_5: identificador del item 5 del carrito

checkout_started.csv: contiene un registro por cada vez que un usuario llega al checkout (para ingresar los datos de pago). Se puede unir con events.csv a traves de events.event_id = checkout_started.id

Columnas:
timestamp: fecha y hora del evento en UTC
id: identificador unico del evento
funnel_id: identificador del embudo al que llego el usuario
total_price: subtotal de la orden en dolares
without_discount_price: precio previo a aplicacion de descuentos
variant_item_1: identificador del item 1 del carrito
variant_item_2: identificador del item 2 del carrito
variant_item_3: identificador del item 3 del carrito
variant_item_4: identificador del item 4 del carrito
variant_item_5: identificador del item 5 del carrito

product_viewed.csv: contiene un registro por cada vez que un usuario llega a una pagina de producto. Se puede unir con events.csv a traves de events.event_id = product_viewed.id

Columnas:
id: identificador unico del evento
timestamp: fecha y hora del evento en UTC
variant: identificador del item
price: precio en dolares del item

product_added.csv: contiene un registro por cada vez que un usuario agrega un producto al carrito. Se puede unir con events.csv a traves de events.event_id = product_added.id

Columnas:
id: identificador unico del evento
timestamp: fecha y hora del evento en UTC
variant: identificador del item
price: precio en dolares del item

popup_view.csv: contiene un registro por cada vez que a un usuario se le muestra un pop up de descuento. El usuario puede aceptar el descuento a cambio de su email. Se puede unir con events.csv a traves de events.event_id = popup_view.id

Columnas:
id: identificador unico del evento
day: dia del evento
popup_id: identificador unico del pop up
popup_name: nombre del pop up
popup_strategy: la estrategia indica en que momento se muestra el popup
popup_discount: oferta del popup

popup_accepted.csv: contiene un registro por cada vez que un usuario acepta acceder al descuento a cambio de su email. Se puede unir con events.csv a traves de events.event_id = popup_accepted.id

Columnas:
id: identificador unico del evento
day: dia del evento
popup_id: identificador unico del pop up
popup_name: nombre del pop up
popup_strategy: la estrategia indica en que momento se muestra el popup
popup_discount: oferta del popup

first_pages.csv: contiene un registro por cada vez que el usuario llega al sitio web desde otro sitio, las navegaciones internas estan excluidas. Se puede unir con events.csv a traves de events.event_id = first_pages.id

Columnas:
id: identificador unico del evento
ts: fecha y hora del evento en UTC
url: url donde aterrizo el usuario

ads_metrics.csv: contiene metricas de performance de los anuncios publicitarios

Columnas:
period: dia en el que las metricas ocurrieron
channel: canal publicitario
campaign_id: identificador unico de la campaña de publicidad
adset_id: identificador unico del grupo de anuncios
ad_id: identificador unico del anuncio
ret_orders: cantidad de ordenes que corresponden a clientes previos
orders: cantidad de ordenes total
users: cantidad de usuarios que visitaron el sitio a traves del anuncio
atc_users: cantidad de usuarios que agregaron productos al carrito
ic_users: cantidad de usuarios que llegaron al checkout
impressions: cantidad de impresiones del anuncio
clicks: cantidad de clicks realizados en el anuncio