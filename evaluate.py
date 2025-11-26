# i have to use similarity_search for manual retriving rather than as_retriever()
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
from nltk.tokenize import regexp_tokenize
import os
import nltk
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
#from langchain_classic import evaluation.scoring.eval_chain.ScoreStringEvalChain


try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Fallback for older NLTK versions
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


model = HuggingFaceEmbeddings(model = 'all-MiniLM-L6-v2', model_kwargs = {'device': 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'},
                              encode_kwargs = {'normalize_embeddings': False})

llm = ChatGroq(model = 'llama-3.1-8b-instant', api_key = os.getenv('GROQ_API_KEY'), temperature=0.6, max_tokens=1000)
vectorDB = Chroma(persist_directory='chromadb/',embedding_function=model)
stop_words = set(stopwords.words('english')) 
eval_model = ChatGroq(model='openai/gpt-oss-20b',temperature=0,api_key=os.getenv('OPENAI_API_KEY'))

def sementic_similarity(prediction, actual):
    scoring_model = BERTScorer(model_type='bert-base-uncased',lang='en', rescale_with_baseline=True)# rescale_with_baseline for scaling the values b default cosine similarites will be high if the 2 words belongs to some category but diff meaning, so we use rescale_with_baseline to avoid this
    # example: "dog" and "cat" are similar in terms of animals but different in meaning
    p,r, f1 = scoring_model.score(prediction, actual)
    return p.tolist(),r.tolist(),f1.tolist(),p.mean().item(),r.mean().item(),f1.mean().item()

def handling_sentence(preds):
        lemmatizer = WordNetLemmatizer()
        han_sen = preds.lower().strip()
        tokens = regexp_tokenize(han_sen, pattern=r'\w+')
        # remove stopwords
        filtered_tokens = [t for t in tokens if t not in stop_words]
        # POS tag expects a list of strings
        tagged_tokens = nltk.pos_tag(filtered_tokens)
        predicted_chars = []
        for tok, tag in tagged_tokens:
            predicted_chars.append((lemmatizer.lemmatize(tok), tag))
        return predicted_chars

def exact_match(prediction, actual):    
    exact_match_list = [] # Use list for clarity
    f1_score_list = []    # Use list for clarity
    
    # Iterate directly over the paired sentences
    for i in range(len(prediction)):
        
        # 1. Exact Match Logic
        if prediction[i] == actual[i]:
            exact_match_list.append(1)
        else:
            exact_match_list.append(0)
            
        # 2. F1 Score Logic
        pred_char = handling_sentence(prediction[i])
        actual_char = handling_sentence(actual[i])
        
        counts = 0
        for char in pred_char:
            if char in actual_char:
                counts += 1
                
        precesion = counts /len(pred_char) if len(pred_char) >0 else 0
        recall = counts / len(actual_char) if len(actual_char) >0 else 0
        f1 = 2 *(precesion*recall)/(precesion+recall) if (precesion+recall) > 0 else 0
        f1_score_list.append(f1)
        
    # Return results
    return exact_match_list, f1_score_list, sum(f1_score_list)/len(prediction), sum(exact_match_list)/len(prediction)
  



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
        print(f"Processing question {i+1}/20")
        result = retrival_chain.invoke({'input': questions})
        results.append({
            "question": questions,
            "generated_answer": result["answer"],
            "retrieved_docs": result["context"] # This returns a list of Document objects
        })
        print(f"Processed question {i+1}")
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

def halucination_evaluation(results,truths):
    metrics = {
        'faithfulness': [],
        'relevance': [],
        'latency': []
    }
    for content,truth in zip(results,truths):
        start_time = time.time()
        faithful_result = faithful_evaluator.evaluate_strings(input = content['question'],
                                                              prediction = content['generated_answer'],
                                                              reference = truth,
                                                              content = content['retrieved_docs']) #result will be dict which contain {score: 0 or 1, value: Y or N,reasoning: content}
        relevance_result = relevance_evaluator.evaluate_strings(input = content['question'],
                                                                prediction = content['generated_answer'],
                                                                content = content['retrieved_docs']) #result will be dict which contain {score: 0 or 1, value: Y or N,reasoning: content}
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

halucination_metrics = halucination_evaluation(answers,truth)
preds = [ pred['generated_answer'] for pred in answers]
sementic_precisions,sementic_recall,sementic_f1_scores,mean_sementic_precisions,mean_sementic_recall,mean_sementic_f1_scores = sementic_similarity(prediction=preds,actual=truth)
exact_matches, f1_scores,avg_f1,avg_exactmatch = exact_match(prediction=preds, actual=truth)
latencys = halucination_metrics['latency']

with open('evaluation_report.txt','w') as f:
    print('---'*10+ ' ' + 'RAG Evaluation Results' + ' ' + '---' *10,file=f)
    print(' ',file=f)
    print('total questions evaluated: ', len(questions),file=f)
    print(' ',file=f)
    print('---'*10+ ' ' + 'Answers Evaluation Results' + ' ' + '---' *10,file=f)
    print('Avg exact match scores is'+ ' '* 5 + ':' +' '+ str(avg_exactmatch),file=f)
    print('Avg f1 scores is'+ ' '* 8 + ':' +' '+ str(avg_f1),file=f)
    print('Avg sementic precision is'+ ' '* 5 + ':' +' '+ str(mean_sementic_precisions),file=f)
    print('Avg sementic recall is'+ ' '* 7 + ':' +' '+ str(mean_sementic_recall),file=f)
    print('Avg sementic f1 scores is'+ ' '* 6 + ':' +' '+ str(mean_sementic_f1_scores),file=f)


    print(' ',file=f)
    print('---'*10+ ' ' + 'Retrival Evaluation Results' + ' ' + '---' *10,file=f)
    print('Avg faithfulness score is'+ ' '* 6 + ':' + ' ' + str(sum(halucination_metrics['faithfulness'])/len(questions)),file=f)
    print('Avg relevance score is'+ ' '* 8 + ':' + ' ' + str(sum(halucination_metrics['relevance'])/len(questions)),file=f)

    print(' ',file=f)
    print('---'*10+ ' ' + 'Latency Evaluation Results' + ' ' + '---' *10,file=f)
    print('Avg latency is'+ ' '* 13 + ':' + ' ' + str(sum(halucination_metrics['latency'])/len(questions)) + ' seconds',file=f)
    print(' ',file=f)

    print('---'*10+' '+'individual questions report:' +' '+'---'*10,file=f)
    for question,answers,actutal_answer,match,f1_score,sementic_precision,sementic_recall,sementic_f1_score,relevance,latency in zip(questions,preds,truth,exact_matches,f1_scores,sementic_precisions,sementic_recall,sementic_f1_scores,halucination_metrics['relevance'],latencys):
        print(f'question: {question},relevance score: {relevance}',file=f)
        print(' ',file=f)
        print(f'generated answer: {answers}',file=f)
        print(' ',file=f)
        print(f'actual answer: {actutal_answer}',file=f)
        print(' ',file=f)
        print(f'exact match score: {match}',file=f)
        print(f'f1 score: {f1_score}',file=f)
        print(f'sementic precision: {sementic_precision}',file=f)
        print(f'sementic recall: {sementic_recall}',file=f)
        print(f'sementic f1 score: {sementic_f1_score}',file=f)
        print(f'Latency: {latency:.2f} seconds',file=f)
