import random
import pandas as pd 
from joblib import load, dump
import numpy as np


def promptear_users_2(users=10):
	user_ids = []
	prompt = []  
    
	unique_users = load('/home/ubuntu/environment/workshop/mim.net/user_ids')
	prods = ['Shapermint Essentials Smoothing Comfort Wireless Bra', 'Truekind Everyday Throw-on Wireless Bralette', 'Truekind Convertible Strapless Bandeau Bra', 'Truekind Seamless Racerback Sports Bra', 'Shapermint Essentials All Day Every Day High-Waisted Shaper Panty', 'Truekind Seamless Stretch Mid-Waist Brief', 'Shapermint Essentials All Day Every Day High-Waisted Shaper Thong', 'Curveez Light Shaping Brief', 'Truekind 5-Pack Seamless Stretch Mid-Waist Brief', 'Shapermint Essentials High-Waisted Control Bikini Bottom', 'Shapermint Essentials High Waisted Full Coverage Swim Skirt', 'Shapermint Essentials One Shoulder Control Swimsuit', 'Bundle Shapermint Essentials - 1 Bikini Bottom + 1 Halter Bikini Top', 'Shapermint Essentials All Day Every Day Scoop Neck Cami', 'Curveez Essential Open Bust Control Tank', 'Maidenform Firm Control Shapewear Camisole', 'Curveez Essential Square Neck Control Tank', 'Shapermint Essentials High Waisted Shaping Leggings', 'Shapermint Essentials All Day Every Day Scoop Neck Bodysuit', 'Shapermint Essentials Ultra-Resistant Shaping Tights']
	
	for _ in range(users):
	    edades = random.choices(['18-25','26-40','41-55','56-70','71 o más'], weights=[30, 50, 50, 30, 5])
	    sexos = random.choices(['F','M'], weights=[9, 1])
	    compras_life = random.randint(1, 40)
	    tkt_prom = max(round(random.gauss(70, 20), 2), 10)
	    frec_compra = max(int(random.gauss(45, 20)), 10)
	    prod_visita = max(round(random.gauss(5, 2), 2), 1)
	    #moda_prod = random.choice(prods)
	    moda_dscto = random.choices([0, 10, 20, 30, 40, 50, 60, 70], weights=[10, 50, 50, 35, 30, 20, 20, 5])
	
	    #frase_a = f'El user tiene una edad de {edades[0]}, es {sexos[0]}, hizo {compras_life} compras en su vida, tiene un ticket promedio de ${tkt_prom}'
	    #frase_b = f', compra cada {frec_compra} días, cada sesión visita {prod_visita} productos en promedio, el producto más comprado es {moda_prod}'
	    frase_a = f'El user tiene una edad de {edades[0]}, es {sexos[0]}, hizo {compras_life} compras en su vida'
	    frase_b = f', compra cada {frec_compra} días, cada sesión visita {prod_visita} productos en promedio'
	    frase_c = f', el pop-up con mayor aceptación fue con {moda_dscto[0]}% de descuento.'
	
	    prompt.append(f'{frase_a} {frase_b} {frase_c}')
	    
	    random_usr = random.choices(unique_users)[0]
	    unique_users.remove(random_usr)
	    user_ids.append(random_usr)
	    
	dic = dict()
	dic['ids'] = user_ids
	dic['prompts'] = prompt	
		
	return dic
	
	
def save_list_to_text_file(list_content, file_path):
	with open(file_path, 'w') as file:
	    for item in list_content:
	        file.write(str(item) + '\n')	      


def event_interpreter(path):
    # Lectura de file 
    events = pd.read_csv(path, parse_dates=[3])
    events = events.sort_values(by='ts')
    
    # Timestamp y tipo de eventos anteriores
    events['last_event_ts'] = events.groupby('user_id')['ts'].shift(1)
    events['last_event_type'] = events.groupby('user_id')['event_type'].shift(1)
    
    # Tiempo entre eventos en minutos
    events['diff_min'] = (events['ts'] - events['last_event_ts']).dt.total_seconds() / 60
    
    # Identificamos cambios de sesiones
    events['session_change'] = (~((events['diff_min'] <= 30) & (events['last_event_type']!='Order Completed'))).astype(int)
    
    # Generamos el session_id para cada user
    events['session_id'] = events.groupby('user_id')['session_change'].cumsum()

    # Sesiones exitosas
    exitos = events[events['event_type']=='Order Completed'][['user_id', 'session_id']]
    exitos['exito'] = True

    # Incorporamos las sesiones exitosas a events
    events = pd.merge(events, exitos, how='left', on=['user_id','session_id'])
    events['exito'] = events['exito'].fillna(False)

    return events



def get_conversion_paths(df):
    df = df[df['exito']]
    df = pd.merge(df, df[df['event_type']=='Product Added'].groupby(['session_id', 'user_id']).agg({'event_type': 'count'}).reset_index(), on=['session_id', 'user_id'], how='left')
    df = df.rename(columns={'event_type_x': 'event_type', 'event_type_y': 'product_added'})
    df.fillna(0, inplace=True)
    df['recovered_cart'] = df['recovered_cart'] = (df['product_added'] == 0) & df['exito']

    # Building paths
    path = 0
    paths_of_session = [] 

    for i in df['session_id'].unique():
        df_temp = df[df['session_id'] == i]
        for j in df_temp['user_id'].unique():
            df_temp_2 = df_temp[df_temp['user_id'] == j]
            path += 1
            sliced_df = df_temp_2.copy()
            sliced_df.sort_values('ts', inplace=True)  
            
            steps_list = []
            step = 0
            for index, row in sliced_df.iterrows():
                steps_list.append(sliced_df.loc[index, 'event_type'])
    
            paths_of_session.append({
                'steps': steps_list,
                'transaction_details': {
                    'discount_offered': np.random.choice([0, 0.1, 0.2, 0.3], 1)[0],
                    'recovered_cart': row['recovered_cart'],
                    'free_shipping': np.random.choice(['True', 'False'], 1)[0],
                },
                'user_id': j
            })
    return paths_of_session


def consolidar(events, path_user_ids, success_paths, path_success_paths):
    # events.to_csv(path_events)
    # Archivo joblib de user_ids
    dump(list(np.unique(events[events['exito']]['user_id'])), path_user_ids)
    # Archivo joblib de casos de exito
    dump(success_paths, path_success_paths)

# Traemos y procesamos eventos
#eventos = event_interpreter('/home/ubuntu/environment/workshop/mim.net/eventos.csv')

# Creamos sesiones exitosas
#paths_of_session = get_conversion_paths(eventos)

# Bajamos el trabajo a alrchivos de joblib
#consolidar(eventos, '/home/ubuntu/environment/workshop/mim.net/user_ids', paths_of_session ,'/home/ubuntu/environment/workshop/mim.net/paths_of_session')




