import json
import boto3
import chromadb
chroma_client = chromadb.Client()
from user_mocker import promptear_users_2, save_list_to_text_file
from bedrock_embedding import finding_best_match, get_success_events
from joblib import load


# LLM PRODUCT 
def get_response(prompt):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime') #creates a Bedrock client
    bedrock_model_id = "ai21.j2-ultra-v1" #set the foundation model
    prompt = prompt #the prompt to send to the model
    body = json.dumps({
        "prompt": prompt, #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 }
    }) #build the request payload
    response = bedrock.invoke_model(body=body, modelId=bedrock_model_id, accept='application/json', contentType='application/json') #send the payload to Bedrock
    response_body = json.loads(response.get('body').read()) # read the response
    response_text = response_body.get("completions")[0].get("data").get("text") #extract the text from the JSON response
    return response_text
    
#---------------------------------------------

# CREATING OUTPUTS
file_path = '/home/ubuntu/environment/workshop/mim.net/output.txt' 
file_path_id = '/home/ubuntu/environment/workshop/mim.net/output_ids.txt' 
print('Mockeando 10 usuarios con atributos relevantes para encontrar simitudes con el usuario..\n')
dic = promptear_users_2(users=10)  

save_list_to_text_file(dic['prompts'], file_path)
save_list_to_text_file(dic['ids'], file_path_id)


# READING OUTPUTS
with open("/home/ubuntu/environment/workshop/mim.net/output.txt", "r") as f:
    text_items = f.read().splitlines()
with open("/home/ubuntu/environment/workshop/mim.net/output_ids.txt", "r") as f:
    ids = f.read().splitlines()
    

# SIMULATING A NEW USER WITH ATTRIBUTES

new_user = 'El usuario tiene una edad de 56-70, es F, hizo 36 compras en su vida , compra cada 25 días, cada sesión visita 1 productos en promedio'
new_user_id = "mTWpNpZeonb+t4Kvid9xdPMj4hAAAw9NHFeu0vAtG8U="
print(f"Usuario que llega a la página de Shapermint: {new_user}\n")

# FINDING THE BEST THREE SIMILAR USERS
print('Buscando usuarios  con atributos similares mediante embedding (BedrockEmbeddings)..\n')
similar_ids = finding_best_match(text_items,ids, new_user)
print(f'Usuarios con mayor similitud: {similar_ids}\n')
similar_ids.append(new_user_id)


# OBTAINING THE BEST SUCCESS PATHS FOR THE THREE SIMILAR USERS
print('Obteniendo paths de compra de los usuarios con mayor similitud que convirtieron..\n')
success_paths = get_success_events(similar_ids)
print(f'Eventos Encontrados: {success_paths}\n')

# CREATING THE FINAL PROMPT 
print("preparando el prompt..\n")
decisions = """
- Opción 1: Carrito recomendado
- Opción 2: Carrito de look prearmado
- Opción 3: 2x1 en productos seleccionados
- Opción 4: 50% de descuento en productos seleccionados
- Opción 5: 50% de descuento en costo de envío
- Opción 6: Envío gratis
- Opción 7: 20% de descuento en el valor final de la compra"""


prompt_marco = 'Hola, te vamos a pedir que te pongas en el lugar de un agente de marketing y elijas entre 4 alternativas , para eso vas a usar informacion del user: ' + new_user + 'el objetivo es, dado los atributos del usuario, dar la mejor recomendacion.'   

events = 'A continuación te presentaremos los tres paths de compra mas recientes de usuarios similares al usuario actual considerados como exitosos por que existió una conversión.' + str(success_paths)  +' El objetivo, es acortar dichos paths con acciones concretas conservando la concersion pero a su vez intentando minimizar los esfuerzos de marketing de no ser necesario.'

final_aclarations = 'Solo debes responder una de las opciones especificadas que maximice la probabilidad de conversion, sin dar NINGUN TIPO DE EXPLICACION de porque lo elegiste'

final_prompt = prompt_marco +"\n" + events + "\n" + decisions + "\n" + final_aclarations

print("Generando respuesta con modelo LLM (ai21.j2-ultra-v1)..\n")
response = get_response(final_prompt)

print("Respuesta:\n")
print(response)

#----------------------------FIN--------------