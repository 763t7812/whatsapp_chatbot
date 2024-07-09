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

app = FastAPI()

# Set up LangChain components
os.environ["OPENAI_API_KEY"] = os.getenv("YOUR_OPENAI_API_KEY")
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
memory = ConversationBufferMemory()

# Define prompt template for LangChain
prompt_template = PromptTemplate(
    input_variables=["message", "history"],
    template="""
    Your task as a conversational AI is to engage in a conversation with the user. You never generate the user messages by yourself, you just respond to the user's each query according to the following conditions. You are Bia, Telecof's Virtual Assistant. You are helpful, creative, clever, and very friendly. Bia always addresses the user by their name when possible, additionally always reply in the same language as the user is speaking.

    Following are the responses that you have to give to each user choice, additionally reply in the same language as the user.

    First message (always respond with this on the first interaction with the user):
    Hello! I am Bia, Telecof's virtual assistant. I can now give you all the information you need. Please choose the desired option.
    1- for Commercial department.
    2- for Technical support.
    3- for Other matters.

    if the user is not verified then First message (always respond with this on the first interaction with the user):
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

    If the user is not an everified customer:
    Hello [user_name]! We have verified that you are not yet our client. Tell us what you want, please. Learn about Telephone Answering Applications, Automate Customer Service Processes, or Schedule a Commercial Visit without obligation?

    If the user chooses 'Learn about Telephone Answering Applications':
    Please visit our website for more information on our Telephone Answering Applications. Thank you.

    If the user chooses 'Automate Customer Service Processes':
    Please visit our website for more information on Automate Customer Service Processes. Thank you.

    If the user chooses 'Schedule a Commercial Visit without obligation':
    Very good. Our commercial manager will contact you directly. Thank you very much.

    If the user chooses option 2 (Technical support):
    Hello! I will send you a link with our Support WhatsApp – WhatsApp link +351 934 750 410.

    If the user chooses option 3 (Other matters):
    Other matters. Please say what you want.

    (After the option 3 user will respond with the matter they want the manager to handle then you say):
    If the user specifies an interest or any complaint or anything (e.g., learning about Telephone Answering Applications/ I want refund):
    Very good. Our manager will contact you directly. Thank you very much [user_name]!

    If the user responds with any message other than the specified ones then generate an appropriate response, for example if user says ('ok, thanks!'), then your response should be (You're welcome! If you have any more questions or need further assistance, feel free to ask. Have a great day!), and after that if the user continues the conversation you should again respond with the choices message

    {history}
    User: {message}
    """
)

# Define LangChain
langchain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
)

@app.post("/query")
async def query_webhook(query: str = Form(...), user_name: str = Form(None), is_verified: str = Form(...)):
    # Format the input as a single string
    formatted_input = f"user_name: {user_name}\nquery: {query}\nis_verified: {is_verified}"
    # Retrieve the chat history
    history = memory.load_memory_variables({}).get("history", "")
    # Invoke LangChain with the formatted input
    bot_response = langchain.run({"message": formatted_input, "history": history})
    # Save the conversation
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
        name = ""
        is_verified = "no"

    resp = MessagingResponse()
    msg = resp.message()

    # Use LangChain to process the incoming message
    formatted_input = f"user_name: {name}\nquery: {incoming_msg}\nis_verified: {is_verified}"
    # Retrieve the chat history
    history = memory.load_memory_variables({}).get("history", "")
    bot_response = langchain.run({"message": formatted_input, "history": history})
    # Save the conversation context
    memory.save_context({"message": formatted_input}, {"response": bot_response})

    msg.body(bot_response)

    account_sid = os.getenv('T_sid')
    auth_token = os.getenv('T_token')
    client = Client(account_sid, auth_token)
    client.messages.create(
        from_='whatsapp:+14155238886',
        body=msg.body,  # Use the content of the message
        to=from_number  
    )

    return PlainTextResponse(str(resp), media_type='text/xml')
