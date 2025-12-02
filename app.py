import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA,create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
import torch
from langchain_groq import ChatGroq
import chromadb
import logging 
from dotenv import load_dotenv 
import os
import time


logger = logging.getLogger(__name__)

load_dotenv()
model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2',model_kwargs = {'device': 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'},
                              encode_kwargs = {'normalize_embeddings': False})
st.title(':red[DataAns] :grey[-] :blue[Your] :green[Data] :orange[Science] :violet[Assistant]')
st.markdown('**Your AI-powered Data Science Companion for In-Depth Understanding and Expert Guidance!**')


#for local run
#vectorDB = chromadb.(persist_directory='chromadb', embedding_function=model)#temp provide access to the non root user in docker container.
# for production level vectorDB we have call the chromadb server instance, with host and port then we load the collection.
host_db = os.environ.get('chromadb_server','localhost')
tries = 30
client_i = None

for i in range(tries):
    try:
        temp_client = chromadb.HttpClient(host=host_db,port=8000)# 8000 is default port for chromadb server.
        temp_client.heartbeat()
        logger.info("Connected to ChromaDB server successfully.")
        client_i = temp_client
        break
    except Exception as e:
        logger.warning(f"Attempt {i+1} of {tries} failed: {e}")
        time.sleep(2)

if client_i is None:
    logger.error("Failed to connect to ChromaDB server after multiple attempts.")
    st.error("Unable to connect to the ChromaDB server. Please try again later.")
    st.stop()
vectorDB = Chroma(collection_name='data_science_data',client=client_i,embedding_function= model)


llm = ChatGroq(model = 'llama-3.1-8b-instant', api_key = os.getenv('GROQ_API_KEY'), temperature=0.6, max_tokens=1000) #context_window for lamma is 4096.


with st.chat_message('assistant'):
    st.markdown("Hello! How can I assist you today?")



if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
#def chat_memory():
 #   chat_memory = ''

  #  for message in st.session_state.messages[-10:]:
   #     role = message['role']
    #    content = message['content']
     #   chat_memory += f"{role}: {content}\n"
    #return chat_memory"""


history_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

histroy_based_prompt = ChatPromptTemplate.from_messages([
    ('system', history_prompt),
    ('placeholder', '{chat_history}'),
    ('user', '{input}')

])


prompt = """ you are a **Data Scientist expert**. you are helping the user, who is a **junior Data Scientist** by default. Your task is to provide a **comprehensive and detailed** answer to the user's question based ONLY on the provided context.
                                                                        
## INSTRUCTIONS: ##
- Answer the user's **Question** based **ONLY** on the content provided within the **<SOURCE_CONTEXT>** tags.
- **Detail Level:** Provide **detailed, in-depth explanations** suitable for a junior Data Scientist. Do not be too brief; expand on the key concepts found in the context.
- Use the conversation history to understand the context better, sentiment and find the best possible answer.
- **Handling Complexity:**
   - If the concept is complex, use the "Dual-Layer" approach (Simple Explanation -> Technical Mastery).
   - If the context allows, provide examples or bullet points to make the answer richer.
- If the answer is not contained within the provided document, politely respond that you are unable to provide an answer based on the given information.
- when the user first start the conversation, **do not say you do not have any chat history, just answer the question based on the provided context.**

### NEGATIVE CONSTRAINTS (CRITICAL) ###
- **DO NOT** mention the XML tag names (like <SOURCE_CONTEXT> or <CHAT_HISTORY>) in your final response. Refer to them naturally as "the provided documents" or "our conversation".
- **DO NOT** start your response with meta-fillers like "Based on the context...". Go straight to the answer.

## INPUT DATA: ##
<SOURCE_CONTEXT>
{context}
</SOURCE_CONTEXT>
                                                                                                                                                                                                                                              
## ZERO SHOT EXAMPLES: ##
input: 'Importance of Data Normalization?' output: Data Normalization is crucial in Supervised Learning... (Detailed explanation follows)
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ('system', prompt),
    ('placeholder', '{chat_history}'),
    ('user', '{input}')])

retriever_data = vectorDB.as_retriever(search_kwargs = {'k':4})
document_chain = create_stuff_documents_chain(llm,qa_prompt)
history_retriver_chain = create_history_aware_retriever(llm,retriever_data,histroy_based_prompt)


retrieval_chain =create_retrieval_chain(history_retriver_chain,document_chain)


if user_input := st.chat_input('Type your question here!'):
    logger.info(f'User Input: {user_input}')
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    #retriever = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=retriever_data)
    #result_dict = retriever.invoke({'query': user_input}) we use query in retrievalQA and similarity_search but for LCEL we use input.
    #result = result_dict['result']

    
    history_memory = []

    for message in st.session_state.messages[:-1]:
        if message['role'] == 'user':
            history_memory.append(HumanMessage(content=message['content']))
        else:
            history_memory.append(AIMessage(content=message['content']))

    response = retrieval_chain.invoke({'input': user_input, 'chat_history': history_memory})
    logger.info(f'Assistant Response: {response["answer"]}')
    st.session_state.messages.append({'role': 'assistant', 'content': response['answer']})

for message in st.session_state.messages:
     with st.chat_message(message['role']):
        st.markdown(message['content'])



counter = vectorDB._collection.count()
st.sidebar.write(f"Total Documents in DB: {counter}")
st.sidebar.write(f'number of query processed: {len(st.session_state.messages)//2}')

st.sidebar.header('RAG Architecture')

# Use an Expander instead of a Button (It stays open while you chat)
with st.sidebar.expander("Show Graph Schema"):
    # 1. Use draw_ascii() to get the string
    ascii_diagram = retrieval_chain.get_graph().draw_ascii()
    
    # 2. Use st.code to preserve the strict spacing/indentation
    st.code(ascii_diagram, language='text')