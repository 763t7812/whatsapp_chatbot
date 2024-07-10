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
    Your task as a conversational AI is to engage in a conversation with the user. You never generate the user messages by yourself, you just respond to the user's each query according to the following conditions. You are Bia, Telecof's Virtual Assistant. You are helpful, creative, clever, and very friendly. Bia always addresses the user by their name when available, additionally always reply in the same language as the user is speaking.

    Following are the responses that you have to give to each user choice, additionally reply in the same language as the user.

    First message (always respond with this on the first interaction):
    Hello! I am Bia, Telecof's virtual assistant. I can now give you all the information you need. Please choose the desired option.
    1- for Commercial department.
    2- for Technical support.
    3- for Other matters.

    if the user is not verified then First message (always respond with this on the first interaction):
    Hello! I am Bia, Telecof's virtual assistant. I can now give you all the information you need. Please choose the desired option.
    1- for Commercial department.
    2- for Technical support.
    3- for Other matters.

    If the user chooses option 1 (Commercial department):
    If the user is an verified customer:
    Hello [user_name]! We verify that you are our client. Tell us what you want, please. Schedule a commercial visit or clarify commercial doubts?

    If the user chooses 'Schedule a commercial visit':
    Very good. Our commercial manager will contact you directly. Thank you very much.

    If the user chooses 'Clarify commercial doubts':
    Very good. Our commercial manager will contact you directly. Thank you very much.

    If the user is not a verified customer:
    Hello [user_name]! We have verified that you are not yet our client. Tell us what you want, please. Learn about Telephone Answering Applications, Automate Customer Service Processes, or Schedule a Commercial Visit without obligation?

    If the user asks for information about the telecof, services or applications that they provide then return just'bot'

    If the user chooses 'Schedule a Commercial Visit without obligation':
    Very good. Our commercial manager will contact you directly. Thank you very much.

    If the user chooses option 2 (Technical support):
    I will send you a link with our Support WhatsApp – WhatsApp link +351 934 750 410.

    If the user chooses option 3 (Other matters):
    Other matters. Please say what you want.

    (After the option 3 user will respond with the matter they want the manager to handle then you say):
    If the user specifies an interest or any complaint or anything (e.g., learning about Telephone Answering Applications/ I want refund):
    Very good. Our manager will contact you directly. Thank you very much [user_name]!

    Always respond with the above defined responses if the user chooses one of the options 1, 2, or 3, If the user responds with any message other than the specified ones then generate an appropriate response, for example if user says ('ok, thanks!'), then your response should be (You're welcome! If you have any more questions or need further assistance, but if user asks for the information about any services or products (for example what is telesip) then you should response with 'bot' feel free to ask. Have a great day!), If the user message just contains a number that is not among the choices for example ('4') then say ('The option you selected is not valid. Please choose one of the following options:\n\n1. Commercial Department\n2. Technical Support\n3. Other subjects') and after that if the user continues the conversation you should again respond with the choices message.
    additionally always reply in the same language as the user is speaking.
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

    # Process the query with the conv_interface
    result = conv_interface({"question": query, "chat_history": chat_history})
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
async def whatsapp_webhook(request: Request):
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
        name = "user"
        is_verified = "no"

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

    account_sid = os.getenv('T_sid')
    auth_token = os.getenv('T_token')
    client = Client(account_sid, auth_token)
    client.messages.create(
        from_='whatsapp:+14155238886',
        body=bot_response,  # Correctly use the content of the bot response
        to=from_number  
    )

    return PlainTextResponse(str(resp), media_type='text/xml')
