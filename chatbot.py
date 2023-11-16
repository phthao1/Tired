from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv(("OPENAI_API_KEY"))
chat = ChatOpenAI(api_key=openai_api_key)
chat([
    HumanMessage(content="Translate this sentence from English to French: I love programming.")
])

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming."),
]

chat(messages)

conversation = ConversationChain(llm=chat)
conversation.run("Translate this sentence from English to French: I love programming.")
conversation.run("Translate it to German.")

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})
memory.load_memory_variables({})

llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "im working on better docs for chatbots"}, {"output": "oh, that sounds like a lot of work"})
memory.save_context({"input": "yes, but it's worth the effort"}, {"output": "agreed, good docs are important!"})

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})
