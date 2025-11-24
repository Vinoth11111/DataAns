# we have to use similarity_search for manual retriving rather than as_retriever()
import time
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.evaluation import load_evaluator
from langchain_classic.chains import RetrievalQA, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from torch import torch
from bert_score import BERTScorer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langchain_openai import OpenAI
from nltk.tokenize import regexp_tokenize
import os
import nltk
import pandas as pd
from dotenv import load_dotenv
#from langchain_classic import evaluation.scoring.eval_chain.ScoreStringEvalChain

model = HuggingFaceEmbeddings(model = 'all-MiniLM-L6-v2', model_kwargs = {'device': 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'},
                              encode_kwargs = {'normalize_embeddings': False})

llm = ChatGroq(model = 'llama-3.1-8b-instant', api_key = os.getenv('GROQ_API_KEY'), temperature=0.6, max_tokens=1000)
vectorDB = Chroma(persist_directory='chromadb/',embedding_function=model)
stop_words = set(stopwords.words('english')) 
eval_model = OpenAI(model='gpt-40',temperature=0)

def sementic_similarity(prediction, actual):
    scoring_model = BERTScorer(model_type='bert-base-uncased',lang='en', rescale_with_baseline=True)# rescale_with_baseline for scaling the values b default cosine similarites will be high if the 2 words belongs to some category but diff meaning, so we use rescale_with_baseline to avoid this
    # example: "dog" and "cat" are similar in terms of animals but different in meaning
    p,r, f1 = scoring_model.score(prediction, actual)
    return p.mean().item(),r.mean().item(),f1.mean().item()

def handling_sentence(preds):
        lemmatizer = WordNetLemmatizer()
        han_sen = preds.lower().strip()
        han_sen = regexp_tokenize(han_sen,pattern=r'\w+')
        predicted_chars = []
        for char in han_sen:
            if char not in stop_words:
                tag = nltk.pos_tag(char)
                predicted_chars.append(lemmatizer.lemmatize(char),tag)
        return predicted_chars

def exact_match(prediction, actual):    
    exact_match = [] 
    f1_score = []
    def each_f1(prediction,actual):
        if prediction == actual:
            exact_match.append(1)
        for sentence in prediction:
            pred_char = handling_sentence(sentence)
            actual_char = handling_sentence(actual)
            counts = 0
            for char in pred_char:
                if char in actual_char:
                    counts += 1
            precesion = counts /len(pred_char) if len(pred_char) >0 else 0
            recall = counts / len(actual_char) if len(actual_char) >0 else 0
            f1_score.append(2 *(precesion*recall)/(precesion+recall) if (precesion+recall) > 0 else 0)
    each_f1(prediction, actual)
    return exact_match,f1_score,sum(f1_score)/len(prediction), sum(exact_match)/len(prediction)
  



prompt = prompt = """ you are a **Data Scientist expert**. you are helping the user, who is a **junior Data Scientist** by default. Your task is to provide a **comprehensive and detailed** answer to the user's question based ONLY on the provided context.
                                                                        
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
prompt_template = ChatPromptTemplate.from_template(prompt)

vector_retriver_chain = vectorDB.as_retriever(search_kwargs={"k":5})
document_retrival_chain = create_stuff_documents_chain(llm,prompt_template)
retrival_chain = create_retrieval_chain(vector_retriver_chain,document_retrival_chain)

def retreival_qa_chain(inputs):

    results = []
    for i,questions in enumerate(inputs):
        result = retrival_chain.invoke({'input': questions})
        results.append({
            "question": questions,
            "generated_answer": result["answer"],
            "retrieved_docs": result["context"] # This returns a list of Document objects
        })
        print(f"Processed question {i+1}/{len(questions)}")
    return results
# result will be a list of dict with question, generated_answer, retrieved_docs

faithful_eval_prompt = {'faithfullness':("Is the submitted answer completely supported by the retrieved context provided? "
        "Answer 'Y' if all facts in the answer are found in the context. "
        "Answer 'N' if the answer includes any information NOT found in the context."
    )}

relevance_eval_prompt = {'relevance': (
        "Does the submission directly answer the user's input question? "
        "Answer 'Y' if it is relevant, 'N' if it is off-topic."
    )}

faithful_evaluator = load_evaluator(evaluator='labeled_criteria',llm=eval_model, criteria = faithful_eval_prompt, eval_prompt=faithful_eval_prompt) # labeled_criteria is used for finding faithfulness and .
relevance_evaluator = load_evaluator(llm=eval_model,evaluator='criteria', criteria=relevance_eval_prompt) # criteria is used for finding relevance and Helpfulness.

def halucination_evaluation(results):
    metrics = {
        'faithfulness': [],
        'relevance': [],
        'latency': []
    }
    for i,content in enumerate(results):
        start_time = time.time()
        faithful_result = faithful_evaluator.evaluate_strings(input = content['question'],
                                                              prediction = content['generated_answer'],
                                                              content = content['retrived_docs']) #result will be dict which contain {score: 0 or 1, value: Y or N,reasoning: content}
        relevance_result = relevance_evaluator.evaluate_strings(input = content['question'],
                                                                prediction = content['generated_answer'],
                                                                content = content['retrived_docs']) #result will be dict which contain {score: 0 or 1, value: Y or N,reasoning: content}
        end_time = time.time()
        latency = end_time - start_time
        metrics['latency'].append(latency)
        metrics['faithfulness'].append(faithful_result['score']) # we want the score
        metrics['relevance'].append(relevance_result['score'])
    return metrics


def overlap_score(question,answer):
    answer_char = set(handling_sentence(answer))
    retreived_data = vectorDB.similarity_search(question,k=3)

    context_token = [handling_sentence(doc.page_content) for doc in retreived_data]
    overlap_counts = 0
    for char in answer_char:
        if char in context_token:
            overlap_counts += 1
    return overlap_counts / len(answer_char) if len(answer_char) >0 else 0



df = pd.read_csv('evaluate_data.csv')

questions = df['question'].tolist()
truth = df['answer'].tolist()
answers = retreival_qa_chain(questions)