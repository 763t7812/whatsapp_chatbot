from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import requests
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from langchain_community.vectorstores import FAISS
import aiofiles
import asyncio
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import tempfile
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from fastapi import BackgroundTasks
from twilio.http.http_client import TwilioHttpClient
import httpx


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
message_info = None

app = FastAPI()

def parse_response(response_str):
    # Remove the square brackets and split by comma
    items = response_str.strip('[]').split(', ')

    # Initialize an empty dictionary
    response_dict = {}

    # Process each item
    for item in items:
        # Split only on the first occurrence of ': '
        if ': ' in item:
            key, value = item.split(': ', 1)
            key = key.strip().strip('\'"')
            value = value.strip().strip('\'"')
            response_dict[key] = value
    
    return response_dict

def messagepurpose(message):
    global message_info

    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": f"You have to detect if the user is asking for the client's information whether about the tasks or other info by giving their phone number, or they want to schedule a new task, if message is for client info/tasks then the word (customer/client/etc...) will be mentioned only then return '[ 'purpose': 'client', 'phone': 'phone number mentioned in the message' ]', if its for scheduling new task/meeting/etc, also if the message contain just date and time then return '[ 'purpose': 'task', 'phone': 'phone number mentioned in the message', 'date': 'date mentioned in the message in format YYYY-MM-DD HH:MM:SS', 'duration': 'duration in minutes (only if the word 'duration is mentioned' and also duration is given)', 'email': 'email if mentioned', 'name':'name if mentioned', 'subject': 'subject if mentioned', 'description': 'description if mentioned' ]' If the message does not contain date or time for example if the user says 'I want to schedule a meeting' or 'I want to schedule a meeting without obligation' or 'Agendar uma visita Comercial sem compromisso' or 'Agendar uma visita Comercial' then return [ 'purpose': 'incorrecttask', 'response': 'Para agendar uma reunião a sua mensagem deve conter a 'data' e a 'hora' da reunião Pode também adicionar o seu nome e a duração da reunião.'], always respond in the same format as defined. and if its asking for anything else then return 'llm'."},
            {"role": "user", "content": f"message: {message}"}
        ],
        "temperature": 0.1,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("YOUR_OPENAI_API_KEY")
    }

    response = requests.post(URL, headers=headers, json=payload, stream=False)
    res = response.json()
    response1 = res['choices'][0]['message']['content']

    if response1 != 'llm':
        message_info = parse_response(response1)

    else:
        message_info = response1

    print(message_info)



# Set up LangChain components
llm = OpenAI(api_key=os.getenv("YOUR_OPENAI_API_KEY"))

# Dictionary to store memory for each user
user_memories = {}

# Define prompt template for LangChain
prompt_template = PromptTemplate(
    input_variables=["message", "history"],
    template="""
    Your task as a conversational AI is to engage in a conversation with the user. You should never generate the user messages by yourself, you just respond to the user's each query according to the following conditions. You are Bia, Telecof's Virtual Assistant. You are helpful, creative, clever, and very friendly. Bia always addresses the user by their name when available,  if the name is unknown then address the user by "cliente"

    If user greets, then always respond with the following message:
    
    Ola. "user's name", Sou a Bia, a assistente Virtual da Telecof. Já posso dar todas as informações que
    precisa. Escolha por favor, a opção pretendida. 
    Departamento Comercial - 1 
    Suporte Técnico - 2 
    Outros assuntos - 3

    Following are the responses that you have to give to each user query, unless the user's query is not specified below you should always respond according to the following conditions. always replace "user's name" with provided user's name.

    - Always respond with the following message If the user messages '1' (Departamento Comercial) and the user is a verified customer then say:
    Departamento Comercial Olá "user's name"! Verificamos que é nosso cliente. Diga-nos o que pretende, por favor. Agendar uma visita comercial ou esclarecer dúvidas comerciais? 

    - Always respond with the following message If the user messages '1' (Departamento Comercial) and the user is not a verified customer then say:
    Olá "user's name"! Verificamos que ainda não é nosso cliente. Diga-nos o que pretende, por favor. Conhecer as Aplicações de Atendimento Telefónico, Automatizar Processos de Atendimento, ou Agendar uma visita Comercial sem compromisso? Conhecer as nossas aplicações de atendimento Telefonico ou Automatizar Processos de Atendimento

    - Always respond with the following message If the user messages '2' (Suporte Técnico) and the user is verified then say:
    Suporte Técnico Olá "user's name" Vou-lhe enviar um Link com o whatsapp do nosso Suporte – link do whatsapp +351 934 750 410 3.

    - Always respond with the following message If the user messages '2' (Suporte Técnico) and the user is not verified then say:
    Suporte Técnico Olá "user's name" Vou-lhe enviar um Link com o whatsapp do nosso Suporte – link do whatsapp +351 934 750 410 3.

    - Always respond with the following message If the user messages '3' (Outros assuntos) and the user is verified then say:
    Outros assuntos. Por Favor diga o que pretende.

    - Always respond with the following message If the user messages '3' (Outros assuntos) and the user is not verified then say:
    Outros assuntos. Por Favor diga o que pretende.

    - If the user messages 'Clarify commercial doubts' or 'esclarecer dúvidas comerciais' then say:
    Muito bem. O nosso gestor comercial vai entrar diretamente em contacto consigo. Muito obrigada

    - (After the option 3 user will respond with the matter they want the manager to handle then you say):
    If the user specifies an interest or any complaint or anything (e.g. I want refund/ I have a complaint):
    Muito bem. O nosso gestor vai entrar diretamente consigo. Muito obrigada "user's name"!

    - If the user message just contains a number that is not among the choices for example ('4') then say ('A opção que você selecionou não é válida. Escolha uma das seguintes opções:\n\n1. Departamento Comercial\n2. Suporte Técnico\n3. Outros assuntos') 

    additionally always reply in the portuguese (PT-PT) language, If the user responds with any message other than the specified ones then generate an appropriate response, for example if user says ('ok, thanks!'), then your response should be (De nada! Se você tiver mais dúvidas ou precisar de mais assistência, sinta-se à vontade para perguntar. Tenha um ótimo dia!).

    - If user asks for the information about any services or products or information about the company itself (for example: "o que é telesip" or "Conhecer as Aplicações de Atendimento Telefónico" or "aprenda sobre aplicativos de atendimento telefônico" or "Automatizar Processos de Atendimento" or "Automate customer service processes" or "tell me about telecof in 2 lines" or etc) then you should respond with 'bot', and after that if the user continues the conversation you should again respond with the choices message.

    Never generate whole conversation by yourself.

    {history}
    User: {message}
    """
)

# Define LangChain
langchain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=ConversationBufferMemory(),
)


async def load_combined_text():
    base_folder = os.path.join('.', 'scrap', '+14155238886')
    combined_filename = os.path.join(base_folder, '+14155238886_combined_data.txt')

    if not os.path.exists(combined_filename):
        print(f"File not found: {combined_filename}")
        return None

    combined_text = ""
    async with aiofiles.open(combined_filename, 'r', encoding='utf-8') as file:
        combined_text = await file.read()

    return combined_text

os.environ["OPENAI_API_KEY"] = os.getenv("YOUR_OPENAI_API_KEY")


async def get_general_answer(query: str, combined_text: str) -> str:
    chunk_size = 2000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.create_documents([combined_text])

    directory = tempfile.mkdtemp()
    vector_index = FAISS.from_documents(texts, OpenAIEmbeddings())
    vector_index.save_local(directory)
    vector_index = FAISS.load_local(directory, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    conv_interface = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.5, max_tokens=1024), retriever=retriever)

    # Retrieve the relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    # Log the retrieved chunks and their similarity scores
    # for idx, doc in enumerate(retrieved_docs):
    #     logger.info(f"Retrieved chunk {idx+1}: {doc.page_content}")

    # Initialize chat history
    chat_history = []

    # Check for exact matches within the retrieved documents
    exact_match_found = False
    for doc in retrieved_docs:
        if query.lower() in doc.page_content.lower():
            logger.info(f"Exact match found in chunk: {doc.page_content}")
            exact_match_found = True
            chat_history.append(("system", doc.page_content))

    portuguese_query = "Answer the following question in european portuguese: " + query
    # Process the query with the conv_interface
    result = conv_interface({"question": portuguese_query, "chat_history": chat_history})
    final_answer = result["answer"]

    if not final_answer:
        final_answer = "Sorry, I couldn't find relevant information."

    return final_answer


def get_task_status(user_id: str) -> dict:
    api_url = f"https://mytelecof.com/api/cliente.php?phonenumber={user_id}&auth=Ym90Ond1UE5qYW05TVNNZFpsMzE2VDlJ"
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error retrieving data from external API: {response.status_code}"}
    
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}


def scheduletask(date: str, phonenumber: str, duration: int = None, 
                 name: str = None, subject: str = None, description: str = None, email: str = None) -> str:
    api_url = "https://mytelecof.com/api/marcar-demo.php?"
    
    # Prepare the payload with mandatory parameters
    payload = {
        "auth": 'Ym90Ond1UE5qYW05TVNNZFpsMzE2VDlJ',
        "date": date,
        "phonenumber": phonenumber
    }
    
    # Add optional parameters if they are provided
    if duration is not None:
        payload["duration"] = str(duration)  # Convert to string for form data
    if name is not None:
        payload["name"] = name
    if subject is not None:
        payload["subject"] = subject
    if description is not None:
        payload["description"] = description
    if email is not None:
        payload["email"] = email
    
    try:
        # Make the POST request to the API using form data
        response = requests.post(api_url, data=payload)
        
        # Print request payload and response for debugging
        print("Request Payload:", payload)
        print("Response Status Code:", response.status_code)
        print("Response Content:", response.content)
        
        # Handle the responses
        if response.status_code == 401:
            return "Authentication failed. Please check your credentials."

        if response.status_code == 200:
            response_json = response.json()
            print(response_json)
            if response_json["success"]:
                return "Task successfully scheduled."
            else:
                return f"Failed to schedule task: {response_json['message']}"

        # Catch all other statuses
        return f"Error scheduling task: {response.status_code}"

    except Exception as e:
        return f"Internal server error: {str(e)}"



@app.post("/query")
async def query_webhook(query: str = Form(...), user_name: str = Form(None), is_verified: str = Form(...)):
    user_id = user_name or "anonymous"
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory()

    memory = user_memories[user_id]

    if len(memory.load_memory_variables({}).get("history", "").split("User: ")) > 6:
        memory.clear()

    formatted_input = f"user_name: {user_name}\nquery: {query}\nis_verified: {is_verified}"
    history = memory.load_memory_variables({}).get("history", "")
    bot_response = langchain.run({"message": formatted_input, "history": history})

    bot_response = bot_response.replace("Bia:", "").replace("AI:", "").replace("Response:", "").strip()

    if 'bot' in bot_response.lower():
        combined_text = await load_combined_text()
        if combined_text:
            bot_response = await get_general_answer(query, combined_text)
        else:
            bot_response = "Sorry, I couldn't retrieve the necessary data."

        memory.save_context({"message": formatted_input}, {"response": bot_response})
        return PlainTextResponse(bot_response)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    global purpose, phone, date, name, duration, email, subject, description
    try:
        form = await request.form()
        to_number = form.get('To')
        from_number = form.get('From')
        sender_name = form.get('ProfileName')
        incoming_msg = form.get('Body', '').lower()
        cleaned_number = from_number.replace('whatsapp:', '')

        test_number = 999333999
        # Fetch client info
        client_info = get_task_status(cleaned_number)

        # Determine if the client is verified
        if isinstance(client_info, dict) and "id" in client_info:
            name = client_info.get("nome", "User")
            is_verified = "yes"
        else:
            name = sender_name
            is_verified = "no"
        
        logging.info(f"Client name: {name}")
        logging.info(f"Is verified: {is_verified}")

        user_id = from_number  # Using phone number as unique identifier
        if user_id not in user_memories:
            user_memories[user_id] = ConversationBufferMemory()

        memory = user_memories[user_id]

        resp = MessagingResponse()
        msg = resp.message()

        # Determine the purpose of the message
        messagepurpose(incoming_msg)

        if message_info == 'llm':
            # Use LangChain to process the incoming message
            formatted_input = f"user_name: {name}\nquery: {incoming_msg}\nis_verified: {is_verified}"
            history = memory.load_memory_variables({}).get("history", "")
            bot_response = langchain.run({"message": formatted_input, "history": history})
            # Clean up the response
            bot_response = bot_response.replace("Bia:", "").replace("AI:", "").strip()

            if 'bot' in bot_response.lower():
                combined_text = await load_combined_text()
                if combined_text:
                    bot_response = await get_general_answer(incoming_msg, combined_text)
                else:
                    bot_response = "Sorry, I couldn't retrieve the necessary data."

            # Save the conversation context
            memory.save_context({"message": formatted_input}, {"response": bot_response})

            # Reset the history if the bot response contains "De nada"
            if "de nada" in bot_response.lower():
                user_memories[user_id] = ConversationBufferMemory()  # Reset the memory for the user

            msg.body(bot_response)

            # Use the async client without idempotency key for bot responses
            account_sid = os.getenv('T_sid')
            auth_token = os.getenv('T_token')
            async with httpx.AsyncClient() as client:
                twilio_client = Client(account_sid, auth_token, http_client=TwilioHttpClient(client))

                async def send_message():
                    try:
                        # Send the bot response to the user
                        await twilio_client.messages.create_async(
                            from_='whatsapp:+14155238886',
                            body=bot_response,
                            to=from_number
                        )
                    except Exception as e:
                        logging.error(f"Failed to send message: {e}")

                background_tasks.add_task(send_message)

                if "nosso gestor" in bot_response.lower():
                    logging.info("Forwarding message to manager")
                    try:
                        await twilio_client.messages.create(
                            from_='whatsapp:+14155238886',
                            body=f"Forwarded message: {incoming_msg}, \nnumber: {from_number}",
                            to='whatsapp:+917999882'
                        )
                    except Exception as e:
                        logging.error(f"Failed to forward message: {e}")

            return PlainTextResponse(str(resp), media_type='text/xml')

        # Handle incorrect task purpose
        if message_info.get('purpose') == 'incorrecttask':
            response_message = message_info.get('response', 'Message purpose is incorrect.')

            msg.body(response_message)

            account_sid = os.getenv('T_sid')
            auth_token = os.getenv('T_token')
            async with httpx.AsyncClient() as client:
                twilio_client = Client(account_sid, auth_token, http_client=TwilioHttpClient(client))

                async def send_message():
                    try:
                        await twilio_client.messages.create_async(
                            from_='whatsapp:+14155238886',
                            body=response_message,
                            to=from_number,
                            idempotency_key=f"{from_number}-{hash(response_message)}"
                        )
                    except Exception as e:
                        logging.error(f"Failed to send message: {e}")

                background_tasks.add_task(send_message)

            return PlainTextResponse(str(resp), media_type='text/xml')

        # Handle task purpose
        if message_info.get('purpose') == 'task':
            phone = cleaned_number
            date = message_info.get('date')
            name = message_info.get('name')
            duration = message_info.get('duration')
            email = message_info.get('email')
            subject = message_info.get('subject')
            description = message_info.get('description')

            schedule = scheduletask(date, phone, duration, name, subject, description, email)

            if schedule == "Task successfully scheduled.":
                response_message = f"Tarefa para {phone} em {date} agendada com sucesso."
            elif "Authentication failed" in schedule:
                response_message = f"Falha na autenticação para o usuário {phone}. Por favor, verifique seu nome de usuário e senha e tente novamente. Se o problema persistir, entre em contato com o suporte para assistência."
            elif "Phonenumber not supplied" in schedule:
                response_message = f"Número de telefone não fornecido para {phone}. Certifique-se de inserir um número de telefone válido no formato +1234567890."
            elif "Date not available" in schedule:
                response_message = f"A data e hora selecionadas {date} não estão disponíveis. Por favor, escolha uma data ou horário diferente e tente novamente."
            elif "Date not supplied" in schedule:
                response_message = f"Uma data válida não foi fornecida para {phone}. Por favor, forneça uma data no formato correto (por exemplo, AAAA-MM-DD) e tente novamente."
            else:
                response_message = f"Falha ao agendar a tarefa para {phone}. Você pode visitar o seguinte site para agendar a reunião manualmente: https://mytelecof.com/m/agendar-reuniao.php"

            msg.body(response_message)

            account_sid = os.getenv('T_sid')
            auth_token = os.getenv('T_token')
            async with httpx.AsyncClient() as client:
                twilio_client = Client(account_sid, auth_token, http_client=TwilioHttpClient(client))

                async def send_message():
                    try:
                        await twilio_client.messages.create_async(
                            from_='whatsapp:+14155238886',
                            body=response_message,
                            to=from_number,
                            idempotency_key=f"{from_number}-{hash(response_message)}"
                        )
                    except Exception as e:
                        logging.error(f"Failed to send message: {e}")

                background_tasks.add_task(send_message)

            return PlainTextResponse(str(resp), media_type='text/xml')

        # Handle client purpose
        if message_info.get('purpose') == 'client':
            phone = message_info.get('phone')
            status = get_task_status(phone)

            # Formatting the status response
            if not status or 'tarefas' not in status:  # Check if status is empty or doesn't contain 'tarefas'
                formatted_status = f"No tasks found for customer {phone}.\n"
            else:
                customer_id = status.get('id', 'Unknown ID')
                tarefas = status['tarefas']
                formatted_status = f"Customer ID: {customer_id}\n\n"
                formatted_status += f"The tasks scheduled with customer {phone} are as follows:\n\n"
                for idx, task in enumerate(tarefas, 1):  # tarefas is a list of tasks
                    formatted_status += f"{idx}. Task #{idx}:\n"
                    formatted_status += f"   - Date: {task['data']}\n"
                    formatted_status += f"   - Time: {task['hora']}\n"
                    formatted_status += f"   - Subject: {task['assunto']}\n"
                    if task['descricao']:
                        formatted_status += f"   - Description: {task['descricao']}\n"
                    formatted_status += f"   - Task Type: {task['descrTipoTarefa']}\n"
                    formatted_status += "\n"

            msg.body(formatted_status)

            account_sid = os.getenv('T_sid')
            auth_token = os.getenv('T_token')
            async with httpx.AsyncClient() as client:
                twilio_client = Client(account_sid, auth_token, http_client=TwilioHttpClient(client))

                async def send_message():
                    try:
                        await twilio_client.messages.create_async(
                            from_='whatsapp:+14155238886',
                            body=formatted_status,
                            to=from_number,
                            idempotency_key=f"{from_number}-{hash(formatted_status)}"
                        )
                    except Exception as e:
                        logging.error(f"Failed to send message: {e}")

                background_tasks.add_task(send_message)

            return PlainTextResponse(str(resp), media_type='text/xml')

        # Save the conversation context if it doesn't match task, client, or greet purposes
        memory.save_context({"message": formatted_input}, {"response": bot_response})

        # Reset the history if the bot response contains "De nada"
        if "de nada" in bot_response.lower():
            user_memories[user_id] = ConversationBufferMemory()  # Reset the memory for the user

        msg.body(bot_response)

        # Use the async client with idempotency key for non-bot responses
        account_sid = os.getenv('T_sid')
        auth_token = os.getenv('T_token')
        async with httpx.AsyncClient() as client:
            twilio_client = Client(account_sid, auth_token, http_client=TwilioHttpClient(client))

            async def send_message():
                try:
                    await twilio_client.messages.create_async(
                        from_='whatsapp:+14155238886',
                        body=bot_response,
                        to=from_number,
                        idempotency_key=f"{from_number}-{hash(bot_response)}"
                    )
                except Exception as e:
                    logging.error(f"Failed to send message: {e}")

            background_tasks.add_task(send_message)

        return PlainTextResponse(str(resp), media_type='text/xml')
    
    except Exception as e:
        logging.error(f"Error processing WhatsApp webhook: {e}")
        return PlainTextResponse("An error occurred.", media_type='text/xml', status_code=500)
