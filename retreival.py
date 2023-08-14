import PyPDF2

from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
import langchain
import bs4
import requests
import io
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain
import tiktoken
import aiohttp
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)



llm = ChatOpenAI(model = 'gpt-3.5-turbo',temperature=0)
executor = ThreadPoolExecutor(max_workers=5)

content_string = '''\
You are SummarizeGPT, a LLM that summarizes research papers into their main ideas.
Suggest a title for the summarised content and write a concise and comprehensive summary of the paper.
 '''

system_message_prompt = SystemMessagePromptTemplate.from_template(content_string)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = tokenizer.encode(text)
    return len(tokens)

template_len = count_tokens(content_string)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000 - template_len,
    chunk_overlap=100,
    length_function=count_tokens,
    separators=["\n\n","\n",'']
)



def scrape_hf(url):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    sections = soup.find_all('article', class_ = 'flex flex-col overflow-hidden rounded-xl border')
    hyperlink_texts = [section.find('h3') for section in sections]
    results = []
    for text in hyperlink_texts:
        data = text.find('a', class_ = 'cursor-pointer')
        results.append((data.text,data['href'].replace('papers','pdf')))
    return results

async def scrape_arxiv(codes,batch_size = 5):
    texts = []
    async with aiohttp.ClientSession() as session:
        for count in range(0,len(codes),batch_size): #Loads 5 papers worth of text, and then sends them to the summarizer
            links =  codes[count:count+batch_size] 
            tasks = [process_pdf(session, 'https://arxiv.org'+link+'.pdf') for link in links]
            batch_results = await asyncio.gather(*tasks)
            texts.extend(batch_results)
    return texts
        
           
async def download_pdf(session,url):
    async with session.get(url) as response:
        return await response.read()
    

async def parse_pdf(session,url):
    pdf_data = await download_pdf(session,url)

    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(pool, parse_pdf_sync, pdf_data)
    return result

def parse_pdf_sync(pdf_data):
    file = io.BytesIO(pdf_data)
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text+=page.extract_text()
    return text


async def process_pdf(session,url):
    pdf_text = await  parse_pdf(session,url)
    result = await summarize(pdf_text)
    return result

##Count tokens then summarize
async def summarize(text):
    portions = text_splitter.split_text(text)
    docs = []
    for portion in portions:
        if 'References\n' in portion:
            portion = portion.split('References\n')[0]
            docs.append(portion)
            break
        else:
            docs.append(portion)
    results = [async_generate(chat_prompt,doc) for doc in docs]
    results = await asyncio.gather(*results)
    return ''.join(results) if len(docs) == 1 else '\n\n'.join(results)

async def async_generate(prompt,text):
    print(prompt.format_prompt(text = text).to_messages())
    return await llm.agenerate(prompt.format_prompt(text = text).to_messages())

async def main():
    res = scrape_hf('https://huggingface.co/papers')
    names,codes = zip(*res)
        
    posts = await scrape_arxiv(codes)
    for post in zip(names,codes,posts):
        print(
            f"{post[0]}: {post[1]}\n\n{post[2]}"
        )
        

asyncio.run(main())


