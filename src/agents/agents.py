from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv
GROQ_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_KEY, max_tokens=900)


# Agent 1 : Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )

# Agent 2 : Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )

# Writter Chain
write_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are am expert research writer. Write clear, structured and insightful reports.'),
    ('human', """Write a detailed research report on the topic below.

Topic : {topic}

Research Gathered : {research}

Structure the report as : 
- Introduction
- key findings (minium 3 well explained points)
- Conclusion
- Sources (List all URLs found in the research)

Be detailed, factual and professional""")
])

writter_chain = write_prompt | llm | StrOutputParser()

# Critic Chain
Crictic_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a sharp and constructive research critic. Be honest and specific'),
    ('human', """Review the research report below and evaluate it strictly.

Report : {report}

Respond in this exact format :

Score : X/10

Strenghts : 
-....
-...

Areas to improve :
- ...
- ...

one lineverdict:
...""")
])

critic_chain = Crictic_prompt | llm | StrOutputParser()