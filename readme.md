# First Action Booster - Product Documentación

Boost First Action es una herramienta diseñada para optimizar las estrategias de marketing digital mediante el análisis de comportamiento de usuario y la recomendación de acciones de marketing personalizadas.

## Componentes Clave
La lógica principal del producto se encuantra en: mim.net/work/bedrock_api.py

### get_response
- **Descripción:** Esta función se encarga de enviar un prompt a la API de Bedrock AI para obtener predicciones basadas en texto.
- **Parámetros:**
  - `prompt`: Texto que se envía a la IA para generar la respuesta.
- **Retorna:** El texto generado por el modelo AI21 J2-Ultra, que sugiere la acción de marketing más efectiva.

### promptear_users_2
- **Descripción:** Genera datos simulados de usuarios para pruebas.
- **Parámetros:**
  - `users`: Número de usuarios para los cuales generar datos.
- **Retorna:** Un diccionario con prompts y IDs de usuario simulados.

### save_list_to_text_file
- **Descripción:** Guarda listas de datos en archivos de texto.
- **Parámetros:**
  - `list_data`: Lista de datos a guardar.
  - `file_path`: Ruta del archivo donde se guardarán los datos.
- **Uso:** Esta función es útil para almacenar resultados de simulaciones o respuestas de la IA.

### finding_best_match
- **Descripción:** Busca los usuarios más similares dentro de una base de datos, basándose en la inserción de embeddings.
- **Parámetros:**
  - `text_items`: Textos de los cuales extraer los embeddings.
  - `ids`: Identificadores de los usuarios a los que corresponden los textos.
  - `new_user`: Texto descriptivo del nuevo usuario para comparar.
- **Retorna:** IDs de los usuarios más similares.

### get_success_events
- **Descripción:** Recupera los caminos de eventos exitosos de usuarios similares para ayudar a predecir acciones eficaces.
- **Parámetros:**
  - `user_ids`: Lista de IDs de usuarios cuyos caminos de eventos se desean consultar.
- **Retorna:** Caminos de eventos que han llevado a conversiones exitosas.

## Utilización

La combinación de estas funciones permite implementar una solución potente para el marketing dirigido basado en análisis de datos y predicciones de IA, proporcionando recomendaciones personalizadas que pueden ser decisivas para el éxito de las campañas de marketing.


