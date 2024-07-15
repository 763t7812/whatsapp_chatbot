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

# Set up LangChain components
llm = OpenAI(api_key=os.getenv("YOUR_OPENAI_API_KEY"))

# Dictionary to store memory for each user
user_memories = {}

# Define prompt template for LangChain
prompt_template = PromptTemplate(
    input_variables=["message", "history"],
    template="""
    Your task as a conversational AI is to engage in a conversation with the user. You should never generate the user messages by yourself, you just respond to the user's each query according to the following conditions. You are Bia, Telecof's Virtual Assistant. You are helpful, creative, clever, and very friendly. Bia always addresses the user by their name when available,  if the name is unknown then address the user by "customer"

    If user greets then always respond with the following message:
    
    Ola. Sou a Bia, a assistente Virtual da Telecof. Já posso dar todas as informações que
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

    - (After the option 3 user will respond with the matter they want the manager to handle then you say):
    If the user specifies an interest or any complaint or anything (e.g. I want refund/ I have a complaint):
    Muito bem. O nosso gestor vai entrar diretamente consigo. Muito obrigada "user's name"!

    - If the user chooses 'Schedule a commercial visit' then say:
    Very good. Our commercial manager will contact you directly. Thank you very much.

    - If the user chooses 'Clarify commercial doubts' then say:
    Muito bem. O nosso gestor comercial vai entrar diretamente em contacto consigo. Muito obrigada

    - If the user chooses 'Schedule a Commercial Visit without obligation' then say:
    Very good. Our commercial manager will contact you directly. Thank you very much. 

    - If the user message just contains a number that is not among the choices for example ('4') then say ('A opção que você selecionou não é válida. Escolha uma das seguintes opções:\n\n1. Departamento Comercial\n2. Suporte Técnico\n3. Outros assuntos') 

    additionally always reply in the portuguese (PT-PT) language, If the user responds with any message other than the specified ones then generate an appropriate response, for example if user says ('ok, thanks!'), then your response should be (You're welcome! If you have any more questions or need further assistance feel free to ask. Have a great day!), but if user asks for the information about any services or products (for example: "o que é telesip" or "Conhecer as Aplicações de Atendimento Telefónico" or "aprenda sobre aplicativos de atendimento telefônico") then you should response with 'bot', and after that if the user continues the conversation you should again respond with the choices message.
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

    bot_response = bot_response.replace("Bia:", "").replace("AI:", "").strip()

    if 'bot' in bot_response.lower():
        combined_text = await load_combined_text()
        if combined_text:
            bot_response = await get_general_answer(query, combined_text)
        else:
            bot_response = "Sorry, I couldn't retrieve the necessary data."

    memory.save_context({"message": formatted_input}, {"response": bot_response})
    return PlainTextResponse(bot_response)


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


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    global purpose, phone, date, name, duration, email, subject, description
    try:
        form = await request.form()
        to_number = form.get('To')
        from_number = form.get('From')
        incoming_msg = form.get('Body', '').lower()
        cleaned_number = to_number.replace('whatsapp:', '')

        # Fetch client info
        client_info = get_task_status(cleaned_number)

        # Determine if the client is verified
        if isinstance(client_info, dict) and "id" in client_info:
            name = client_info.get("nome", "User")
            is_verified = "yes"
        else:
            name = "unknown"
            is_verified = "no"
        
        logging.info(f"Client name: {name}")
        logging.info(f"Is verified: {is_verified}")

        user_id = from_number  # Using phone number as unique identifier
        if user_id not in user_memories:
            user_memories[user_id] = ConversationBufferMemory()

        memory = user_memories[user_id]

        resp = MessagingResponse()
        msg = resp.message()

        # Use LangChain to process the incoming message
        formatted_input = f"user_name: {name}\nquery: {incoming_msg}\nis_verified: {is_verified}"
        # Retrieve the chat history
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

            # Reset the history if the bot response contains "You're welcome!"
            if "you're welcome!" in bot_response.lower():
                user_memories[user_id] = ConversationBufferMemory()  # Reset the memory for the user

            msg.body(bot_response)

            # Use the async client with idempotency key
            account_sid = os.getenv('T_sid')
            auth_token = os.getenv('T_token')
            async with httpx.AsyncClient() as client:
                twilio_client = Client(account_sid, auth_token, http_client=TwilioHttpClient(client))

                def send_message():
                    try:
                        twilio_client.messages.create(
                            from_='whatsapp:+14155238886',
                            body=bot_response,
                            to=from_number
                        )
                    except Exception as e:
                        logging.error(f"Failed to send message: {e}")

                background_tasks.add_task(send_message)

                # Send incoming message to commercial manager if condition met
                if 'nosso gestor comercial vai entrar diretamente em contacto' in bot_response.lower():
                    def send_to_commercial_manager():
                        try:
                            twilio_client.messages.create(
                                from_='whatsapp:+14155238886',
                                body=incoming_msg,
                                to='whatsapp:+923312682192'
                            )
                        except Exception as e:
                            logging.error(f"Failed to send message to commercial manager: {e}")

                    background_tasks.add_task(send_to_commercial_manager)

            return PlainTextResponse(str(resp), media_type='text/xml')

        if 'task' in bot_response.lower():
            if isinstance(bot_response, dict) and bot_response.get('purpose') != 'bot':
                purpose = bot_response.get('purpose')
                phone = bot_response.get('phone')
                date = bot_response.get('date')
                name = bot_response.get('name')
                duration = bot_response.get('duration')
                email = bot_response.get('email')
                subject = bot_response.get('subject')
                description = bot_response.get('description')
                print(bot_response)
            schedule = scheduletask(date, phone, duration, name, subject, description, email)

            if schedule == "Task successfully scheduled.":
                response_message = f"Task for {phone} on {date} has been successfully scheduled."
            elif "Authentication failed" in schedule:
                response_message = f"Authentication failed for user {phone}. Please check your username and password, and try again. If the issue persists, contact support for assistance."
            elif "Phonenumber not supplied" in schedule:
                response_message = f"Phone number not supplied for {phone}. Ensure you have entered a valid phone number in the format +1234567890."
            elif "Date not available" in schedule:
                response_message = f"The selected date and time {date} are not available. Please choose a different date or time slot and try again."
            elif "Date not supplied" in schedule:
                response_message = f"A valid date was not supplied for {phone}. Please provide a date in the correct format (e.g., YYYY-MM-DD) and try again."
            else:
                response_message = f"Failed to schedule task for {phone} due to an unexpected error: {schedule}. Please try again or contact support if the issue continues."

            msg.body(response_message)

            client.messages.create(
                from_='whatsapp:+14155238886',
                body=response_message,
                to=from_number  
            )

            return PlainTextResponse(str(resp), media_type='text/xml')

        else:
            # Save the conversation context
            memory.save_context({"message": formatted_input}, {"response": bot_response})

            # Reset the history if the bot response contains "You're welcome!"
            if "you're welcome!" in bot_response.lower():
                user_memories[user_id] = ConversationBufferMemory()  # Reset the memory for the user

            msg.body(bot_response)

            # Use the async client with idempotency key
            account_sid = os.getenv('T_sid')
            auth_token = os.getenv('T_token')
            client = Client(account_sid, auth_token, http_client=TwilioHttpClient(pool_connections=False))

            async def send_message():
                try: 
                    await client.messages.create_async(
                        from_='whatsapp:+14155238886',
                        body=bot_response,
                        to=from_number, # Set the idempotency key
                    )
                except Exception as e:
                    logging.error(f"Failed to send message: {e}")

            background_tasks.add_task(send_message)

            return PlainTextResponse(str(resp), media_type='text/xml')
    
    except Exception as e:
        logging.error(f"Error processing WhatsApp webhook: {e}")
        return PlainTextResponse("An error occurred.", media_type='text/xml', status_code=500)

