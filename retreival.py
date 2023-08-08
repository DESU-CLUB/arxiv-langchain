import PyPDF2

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import MathpixPDFLoader
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

loader = PyPDFLoader( '/home/spooky/Documents/Deep Learning/papers/1506.02640.pdf')
pages = loader.load_and_split()


# use the function
print(pages[5].page_content.replace('-\n', '').replace('\n',' '))

llm = ChatOpenAI(model = 'gpt-3.5-turbo',temperature=0)
executor = ThreadPoolExecutor(max_workers=5)

content_string = '''\
You are SummarizeGPT, a LLM that summarizes research papers into their main ideas.
Summarize the following dialogue: 
{text}

Summarised content:'''
content_template = PromptTemplate(input_variables=['text'], template=content_string)


def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = tokenizer.encode(text)
    return len(tokens)

template_len = count_tokens(content_string)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000 - template_len,
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
            tasks = [process_pdf('https://arxiv.org'+link+'.pdf') for link in links]
            await asyncio.gather(*tasks)
            texts.append(tasks)
    return texts
        
           
async def download_pdf(session,url):
    async with session.get(url) as response:
        return await response.read()
    

async def parse_pdf(session,url):
    pdf_data = await download_pdf(session,url)
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, parse_pdf_sync, pdf_data)
    return result

async def parse_pdf_sync(pdf_data):
    file = io.BytesIO(pdf_data.content)
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
    docs = [Document(page_content = portion) for portion in portions]
    chain = load_summarize_chain(llm, chain_type = 'stuff', prompt = content_template)    
    results = await chain.arun(docs)
    return ''.join(results) if len(docs) == 1 else '\n\n'.join(results)
    


async def main():
    res = scrape_hf('https://huggingface.co/papers')
    names,codes = zip(*res)
        
    posts = await scrape_arxiv(codes)
    for post in zip(names,codes,posts):
        print(
            f"{post[0]}: {post[1]}\n\n{post[2]}"
        )
        






