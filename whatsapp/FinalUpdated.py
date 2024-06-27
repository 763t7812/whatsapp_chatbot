from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict
import os
import requests
from bs4 import BeautifulSoup
import uuid
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import asyncio
import fitz  # PyMuPDF
import re
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
message_info = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_statuses: Dict[str, str] = {}
chat_history = []

async def load_combined_text(user_id):
    base_folder = os.path.join('.', 'scrap', user_id)
    combined_filename = os.path.join(base_folder, f'{user_id}_combined_data.txt')

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

def clean_html(soup):
    for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
        script_or_style.decompose()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

async def extract_links(url, base_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = {a['href'] for a in soup.find_all('a', href=True)}
    links = {link if link.startswith('http') else base_url + link for link in links}
    return links

async def extract_text_from_pdf(pdf_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(pdf_url, headers=headers)
    response.raise_for_status()
    with open("temp.pdf", 'wb') as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    os.remove("temp.pdf")
    return text

async def extract_text_from_local_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

async def append_text_to_combined(user_id, text):
    base_folder = os.path.join('.', 'scrap', user_id)
    combined_filename = os.path.join(base_folder, f'{user_id}_combined_data.txt')
    os.makedirs(base_folder, exist_ok=True)
    async with aiofiles.open(combined_filename, 'a', encoding='utf-8') as file:
        await file.write(text)
        await file.write("\n\n")

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

async def scrape_website(url, base_url, visited, extracted, base_folder, depth=2, retries=5, delay=5, timeout=10):
    if depth == 0 or url in visited or url in extracted:
        return

    visited.add(url)
    attempt = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    while attempt < retries:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = clean_html(soup)

            # Extract contact details
            contact_details = extract_contact_details(soup)
            if contact_details:
                text += "\n\nContact Details Found:\n" + "\n".join(contact_details)

            links = await extract_links(url, base_url)
            links_text = "\n\nLinks found on this page:\n" + "\n".join(links)
            full_text = text + links_text

            sanitized_filename = sanitize_filename(url.replace('https://', '').replace('http://', '').replace('www.', '').replace('/', '_'))
            filename = os.path.join(base_folder, f'data_from_{sanitized_filename}.txt')
            os.makedirs(base_folder, exist_ok=True)
            async with aiofiles.open(filename, 'w', encoding='utf-8') as file:
                await file.write(full_text)
            print(f"Data from {url} has been saved to {filename}")
            extracted.add(url)

            for link in links:
                if link not in visited and link not in extracted:
                    if link.endswith('.pdf'):
                        pdf_text = await extract_text_from_pdf(link)
                        sanitized_pdf_filename = sanitize_filename(link.replace('https://', '').replace('http://', '').replace('www.', '').replace('/', '_').replace('.pdf', ''))
                        pdf_filename = os.path.join(base_folder, f'data_from_{sanitized_pdf_filename}.txt')
                        async with aiofiles.open(pdf_filename, 'w', encoding='utf-8') as pdf_file:
                            await pdf_file.write(pdf_text)
                        print(f"PDF content from {link} has been saved to {pdf_filename}")
                        extracted.add(link)
                    else:
                        await scrape_website(link, base_url, visited, extracted, base_folder, depth - 1, retries, delay, timeout)
            break
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1} of {retries}: Timeout. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            attempt += 1
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred for {url}: {str(e)}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} of {retries}: Error during requests to {url}: {str(e)}")
            break
        except Exception as e:
            print(f"An unexpected error occurred for {url}: {str(e)}")
            break

def extract_contact_details(soup):
    contact_details = []

    # Find all email addresses
    emails = set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", soup.get_text()))
    if emails:
        contact_details.extend(emails)

    # Find all phone numbers (simple regex for illustration; can be improved)
    phone_numbers = set(re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", soup.get_text()))
    if phone_numbers:
        contact_details.extend(phone_numbers)

    return contact_details

async def combine_and_clean_up(base_folder, combined_filename):
    async with aiofiles.open(combined_filename, 'w', encoding='utf-8') as combined_file:
        for root, _, files in os.walk(base_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path == combined_filename:
                    continue
                if file.endswith('.txt'):
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        await combined_file.write(await f.read())
                        await combined_file.write("\n\n")
                    os.remove(file_path)
    print(f"All data combined into {combined_filename} and individual files deleted.")

async def main_scraper(user_id, base_url, job_id, file_path=None):
    job_statuses[job_id] = "in_progress"
    try:
        main_page_url = base_url
        visited_links = set()
        extracted_links = set()

        base_folder = os.path.join('.', 'scrap', user_id)
        os.makedirs(base_folder, exist_ok=True)

        await scrape_website(main_page_url, main_page_url, visited_links, extracted_links, base_folder, depth=5)  # Adjust depth as needed

        combined_filename = os.path.join(base_folder, f'{user_id}_combined_data.txt')
        await combine_and_clean_up(base_folder, combined_filename)

        if file_path:
            extracted_text = await extract_text_from_local_pdf(file_path)
            await append_text_to_combined(user_id, extracted_text)

        job_statuses[job_id] = "completed"
    except Exception as e:
        job_statuses[job_id] = f"failed: {str(e)}"


def messagepurpose(message, name):
    global message_info

    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": f"You have to detect if the user is asking for the client's information whether about the tasks or other info by giving their phone number, or they want to schedule a new task, if message is for client info/tasks then the word (customer/client/etc...) will be mentioned only then return '[ 'purpose': 'client', 'phone': 'phone number mentioned in the message' ]', if its for scheduling new task/meeting/etc then return '[ 'purpose': 'task', 'phone': 'phone number mentioned in the message', 'date': 'date mentioned in the message in format YYYY-MM-DD HH:MM:SS', 'duration': 'duration in minutes (only if the word 'duration is mentioned' and also duration is given)', 'email': 'email if mentioned', 'name':'name if mentioned', 'subject': 'subject if mentioned', 'description': 'description if mentioned' ]', or if its a greeting message then return '[ 'purpose': 'greet', 'response': 'your resonse to the greeting, also this is the name of the user {name}, please always address the user with this name, after greeting also ask them 'how can I assist you' ' ]', and if its asking for anything else then return 'bot'."},
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

    if response1 != 'bot':
        message_info = parse_response(response1)

    else:
        message_info = response1

    print(message_info)

def get_task_status(user_id: str) -> dict:
    api_url = f"https://mytelecof.com/api/cliente.php?phonenumber={user_id}&auth=Ym90Ond1UE5qYW05TVNNZFpsMzE2VDlJ"
    
    try:
        # Make the GET request to the external API
        response = requests.get(api_url)
        
        # Check if the response status code is OK (200)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error retrieving data from external API: {response.status_code}"}
    
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}


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


def scheduletask(date: str, phonenumber: str, duration: int = None, 
                 name: str = None, subject: str = None, description: str = None, email: str = None) -> str:
    api_url = "https://mytelecof.com/api/marcar-demo.php"
    
    # Prepare the payload with mandatory parameters
    payload = {
        "auth": 'Ym90Ond1UE5qYW05TVNNZFpsMzE2VDlJ',
        "date": date,
        "phonenumber": phonenumber
    }
    
    # Add optional parameters if they are provided
    if duration is not None:
        payload["duration"] = duration
    if name is not None:
        payload["name"] = name
    if subject is not None:
        payload["subject"] = subject
    if description is not None:
        payload["description"] = description
    if email is not None:
        payload["email"] = email
    
    try:
        # Make the POST request to the API
        response = requests.post(api_url, data=payload)
        
        # Handle the responses
        if response.status_code == 401:
            return "Authentication failed. Please check your credentials."

        if response.status_code == 200:
            response_json = response.json()
            if response_json["success"]:
                return "Task successfully scheduled."
            else:
                return f"Failed to schedule task: {response_json['message']}"

        # Catch all other statuses
        return f"Error scheduling task: {response.status_code}"

    except Exception as e:
        return f"Internal server error: {str(e)}"
    

@app.post("/scrape")
async def scrape(
    background_tasks: BackgroundTasks,
    base_url: str = Form(...),
    mobile_number: str = Form(...),
    whatsapp_key: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(None),
):
    file_path = None
    if file:
        upload_folder = os.path.join('.', 'uploads', mobile_number)
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await file.read())

    job_id = str(uuid.uuid4())
    job_statuses[job_id] = "pending"
    background_tasks.add_task(main_scraper, mobile_number, base_url, job_id, file_path)

    return {"message": "Scraping started", "job_id": job_id}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    status = job_statuses.get(job_id)
    if status is None:
        return {"job_id": job_id, "status": "not_found"}
    return {"job_id": job_id, "status": status}
    

@app.post("/query")
async def query(
    user_id: str = Form(...),
    query: str = Form(...),
):
    messagepurpose(query)

    combined_text = await load_combined_text(user_id)  # Await the async function

    if combined_text:
        try:
            answer = await get_general_answer(query, combined_text)
            return {"answer": answer}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Combined text could not be loaded or is empty"}


@app.post("/querycheck")
async def query(
    date: str = Form(...),
    phone: str = Form(...)
):
    status = scheduletask(date, phone)
    return status

@app.get("/folder/{phone_number}")
async def get_folder_contents(phone_number: str):
    folder_path = os.path.join('.', 'scrap', phone_number)

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    files = []

    for root, _, file_names in os.walk(folder_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            files.append(file_path)

    if not files:
        raise HTTPException(status_code=404, detail="No files found in the folder")

    # Find the first .txt file
    for file in files:
        if file.endswith('.txt'):
            return FileResponse(path=file, media_type='text/plain', filename=os.path.basename(file))

    raise HTTPException(status_code=404, detail="No .txt files found in the folder")


# Twilio credentials
account_sid = os.getenv('T_sid')
auth_token = os.getenv('T_token')
client = Client(account_sid, auth_token)

class Message(BaseModel):
    to: str
    body: str


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    global purpose, phone, date, name, duration, email, subject, description
    form = await request.form()
    to_number = form.get('To')
    from_number = form.get('From')
    cleaned_number = to_number.replace('whatsapp:', '')
    print(cleaned_number)
    print(form)

    incoming_msg = form.get('Body', '').lower()
    sender_name = form.get('ProfileName', '')  # Extract the sender's name from the request form if available
    resp = MessagingResponse()
    msg = resp.message()

    print(incoming_msg)
    messagepurpose(incoming_msg, sender_name)

    if message_info == 'bot':
        # Ensure to await the async function
        combined_text = await load_combined_text(cleaned_number)

        if combined_text is not None:
            # Ensure to await the async function
            answer = await get_general_answer(incoming_msg, combined_text)
            print(answer)
        else:
            answer = "Sorry, I couldn't find any information."

        msg.body(answer)

        client.messages.create(
            from_='whatsapp:+14155238886',
            body=answer,
            to=from_number  
        )

        return PlainTextResponse(str(resp), media_type='text/xml')

    if isinstance(message_info, dict) and message_info.get('purpose') != 'bot':
        purpose = message_info.get('purpose')
        phone = message_info.get('phone')
        date = message_info.get('date')
        name = message_info.get('name')
        duration = message_info.get('duration')
        email = message_info.get('email')
        subject = message_info.get('subject')
        description = message_info.get('description')
        response = message_info.get('response')
        print(message_info)

        if purpose == 'client':
            status = get_task_status(phone)
            
            # Debugging print statement
            print(f"Status: {status}")

            # Formatting the status response
            if not status or 'tarefas' not in status:  # Check if status is empty or doesn't contain 'tarefas'
                formatted_status = f"No tasks found for customer {phone}.\n"
            else:
                customer_id = status.get('id', 'Unknown ID')
                tarefas = status['tarefas']
                formatted_status = f"Customer ID: {customer_id}\n\n"
                formatted_status += f"The tasks scheduled with customer {phone} are as follows:\n\n"
                for idx, task in enumerate(tarefas, 1):  # tarefas is a list of tasks
                    # Debugging print statement for each task
                    print(f"Task {idx}: {task}")

                    formatted_status += f"{idx}. Task #{idx}:\n"
                    formatted_status += f"   - Date: {task['data']}\n"
                    formatted_status += f"   - Time: {task['hora']}\n"
                    formatted_status += f"   - Subject: {task['assunto']}\n"
                    if task['descricao']:
                        formatted_status += f"   - Description: {task['descricao']}\n"
                    formatted_status += f"   - Task Type: {task['descrTipoTarefa']}\n"
                    formatted_status += "\n"

            msg.body(formatted_status)

            client.messages.create(
                from_='whatsapp:+14155238886',
                body=formatted_status,
                to=from_number  
            )

            return PlainTextResponse(str(resp), media_type='text/xml')
        
        if purpose == 'task':
            print(date, phone, duration)
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
        
        if purpose == 'greet':
            print('1')
            print(purpose)
            print(response)
            msg.body(response)

            client.messages.create(
                from_='whatsapp:+14155238886',
                body=response,
                to=from_number  
            )

            return PlainTextResponse(str(resp), media_type='text/xml')