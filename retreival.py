import PyPDF2

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
import tiktoken
import aiohttp
from reliablegpt import reliableGPT
import openai
from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()
with open('email.txt','r') as f:
    email =  f.readline()
openai.ChatCompletion.create = reliableGPT(openai.ChatCompletion.create, user_email= email)

executor = ThreadPoolExecutor(max_workers=5)

content_string = '''\
You are SummarizeGPT, a LLM that summarizes research papers into their main ideas.
Suggest a title for the summarised content and write a concise and comprehensive summary of the paper.
 '''

tagger_string = '''\
You are TaggerGPT, a LLM that tags research papers with their respective fields. Given the content of a paper, suggest a title and tag the paper with the appropriate field.
The given tags allowed are the following: Language, Vision, RL, Alignment, Robotics, Audio and Miscellaneous.
You can give more than 1 tag to the paper if you think it is appropriate. You will only respond with the tag, and will delimit multiple tags with commas. 
'''

def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = tokenizer.encode(text)
    return len(tokens)

tag_prompt_len = count_tokens(tagger_string)

template_len = count_tokens(content_string)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3200 - template_len,
    chunk_overlap=100,
    length_function=count_tokens,
    separators=["\n\n","\n",'']
)



def scrape_hf(url,current_posts= []):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    sections = soup.find_all('article', class_ = 'flex flex-col overflow-hidden rounded-xl border')
    hyperlink_texts = [section.find('h3') for section in sections]
    upvotes = [section.find('div', class_ = 'leading-none').text for section in sections]
    upvotes = list(map(lambda x: int(x) if x !='-' else 0,upvotes))
    results = []
    filtered = []
    for count,text in enumerate(hyperlink_texts):
        data = text.find('a', class_ = 'cursor-pointer')
        results.append((upvotes[count],data.text,data['href'].replace('papers','pdf')))
    filtered = list(filter(lambda x: x[0] >8,results))
    return filtered if len(filtered)>=3 else results[:3]

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
    tags = await tagger(result)
    return (tags,result)

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
    results = [async_generate(content_string,doc) for doc in docs]
    results = await asyncio.gather(*results)
    out = ''.join(results) if len(docs) == 1 else '\n\n'.join(results)
    if count_tokens(out) > 4000:
        print('Summarizing summary......')
        return await summarize(out)
    else:
        results = await async_generate(content_string,out)
        print('Paper summarized!')
    return results

async def async_generate(prompt,text):
    print('Calling API to generate.......')
    messages = [{'role':'system','content':prompt},{'role':'user','content':text}]
    res = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages,
    )
    return res['choices'][0]['message']['content']

async def tagger(text):
    while count_tokens(text) >= 4000:
        text = text[:-50] #Truncate the text until it is below 4000 tokens
    results = await async_generate(tagger_string,text)
    return results

##### DISCORD LOGIC ########

def format_post(post):
    if '\n\nSummary:' in post:
        post = 'Summary:'+post.split('\n\nSummary:')[1]
        return post

def format_tags(tags):
    approved_tags = {'Language','Vision','RL','Alignment','Robotics','Audio','Miscellaneous'}
    print(tags)
    ##Validate and format tags
    if 'Tags:' in tags:
        post_tags = tags.split('Tags:')[1].split(',')
        valid_tags = {tag.strip() for tag in post_tags} & approved_tags
    return ', '.join(valid_tags) if valid_tags else 'Miscellaneous'
        
async def main():
    res = scrape_hf('https://huggingface.co/papers')
    _,names,codes = zip(*res)
    tags,posts = zip(*await scrape_arxiv(codes))
    posts = list(map(format_post,posts))
    tags = list(map(format_tags,tags))
    for post in zip(names,codes,posts,tags):
        print(
            f"{post[0]}: https://arxiv.org{post[1]}.pdf\n\n{post[2]}\nTags given: {post[3]}\n"
        )
        

asyncio.run(main())


