import streamlit as st
from streamlit_chat import message
import dotenv
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from pymongo.mongo_client import MongoClient
from langchain_community.utils.math import cosine_similarity
import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import smtplib

print(dotenv.load_dotenv())

from_email = "animalbiteschatbot@gmail.com"
to_email="pandasrit5@gmail.com"
password = os.environ["APP_PASSWORD"]

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)
# for classification betweeen casual and subject 
class casual_subject(BaseModel):
    description: str = Field(
        description="""Classify the given user query into one of two categories:
Casual Greeting - If the query is a generic greeting or social pleasantry (e.g., 'Hi', 'How are you?', 'Good morning').
Subject-Specific - If the query is about a particular topic or seeks information (e.g., 'What is Python?', 'Tell me about space travel').
Return only the category name: 'Casual Greeting' or 'Subject-Specific'.""",
    enum=['Casual Greeting','Subject-Specific']
    )

#for checking if it is related to
class related_not(BaseModel):
    description: str = Field(
        description="""Determine whether the given user query is related to animal bites.
Categories:
Animal Bite-Related - If the query mentions animal bites, their effects, treatment, prevention, or specific cases (e.g., 'What to do after a dog bite?', 'Are cat bites dangerous?').
Not Animal Bite-Related - If the query does not pertain to animal bites.
Return only the category name: 'Animal Bite-Related' or 'Not Animal Bite-Related'.""",
    enum=['Animal Bite-Related','Not Animal Bite-Related']
    )

#chat bot stuff 
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ["OPENAI_KEY"])
llm=ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0,api_key=os.environ["OPENAI_KEY"])
smaller_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125",api_key=os.environ["OPENAI_KEY"])

#mongodb initialization
client = MongoClient(os.getenv("MONGODB_URI"))
db=client["pdf_file"]
collection=db["animal_bites"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_input():
    user_input = st.session_state.user_input.strip()
    #print(st.session_state.chat_history)

    if user_input:
        #converting user input into standalone prompt
        retrival_prompt_template=f"""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
chat_history: {st.session_state.chat_history}
latest_user_input:{user_input}"""

        modified_user_input=llm.invoke(retrival_prompt_template).content
        print(modified_user_input)

        #casual or subject related?
        prompt = tagging_prompt.invoke({"input": modified_user_input})
        response = smaller_llm.with_structured_output(casual_subject).invoke(prompt)
        print(response.model_dump()["description"])
        if response.model_dump()["description"]=='Subject-Specific':

            embedding=embeddings_model.embed_query(modified_user_input)

            result=collection.aggregate([
            {
                "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": 3
                }
            }
            ])

            context=""

            for i in result:
                db_embedding=i["embeddings"]
                val=cosine_similarity([db_embedding],[embedding])[0][0]
                print(round(val,2))
                if round(val,2)>=0.44:
                    context=context+i["raw_data"]+"\n\n"
            print(len(context))

            #if context is available 
            if context:
                prompt_template=f"""you are a chatbot meant to answer questions related to animal bites, answer the question based on the given context. 
                context:{context}
                question:{modified_user_input}"""
                response=llm.invoke(prompt_template)
                bot_response=response.content
            #context is not available 
            else:
                prompt = tagging_prompt.invoke({"input": modified_user_input})
                response = smaller_llm.with_structured_output(related_not).invoke(prompt)
                if response.model_dump()["description"]=='Not Animal Bite-Related':
                    bot_response="Sorry, but I specialize in answering questions related to animal bites.\
                            I may not be able to help with your query, but if you have any questions about animal bites, \
                            their effects, treatment, or prevention, I'd be happy to assist!"
                else:
                    # Create the email message
                    message = f"Subject: \n\n{modified_user_input}"
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(from_email, password)
                    server.sendmail(from_email, to_email, message)
                    server.quit()
                    print("Email sent successfully!")
                    bot_response="I am able to answer your question at the moment. The Doctor has been notified, please check back in a few days."  

            #     bot_response=llm.invoke(f""""you are a chatbot that specilizes in questions related to animal bites
            # if the question is a general question about you or any form of greetings such as how are and such, you are allowed to answer freely.
            # else if the question is related to animal bites then reply with 'yes' and nothing more.
            # else reply saying that you only have knowledge related to animal bites and that you are unable to answer your question because of it.
            # question: {user_input}""").content

            #     pattern = r"(.{0,4})([Yes|yes])(.{0,4})"
            #     match = re.search(pattern, bot_response, re.DOTALL)
            #     if match:
            #         print("sending message to doc")
            #         bot_response="I am able to answer your question at the moment. The Doctor has been notified, please check back in a few days."


        #its a casual question 
        else:
            bot_response=llm.invoke(f"""system:you are a chatbot that specilizes in medical questions related to animal bites
                                    question: {user_input}""").content

        #bot_response = "This is bot response."
        st.session_state.chat_history.append((user_input, bot_response))
        st.session_state.user_input = ""

def display_chat():
    # Display chat messages
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        message(user_msg, is_user=True, key=f"user_msg_{i}")
        message(bot_msg, key=f"bot_msg_{i}")

def main():
    st.title("Chatbot for Animal Bites")

    # Chat display container
    chat_container = st.container()
    with chat_container:
        display_chat()
    st.text_input(
        "Type something...",
        key="user_input",
        placeholder="Enter your message here",
        on_change=process_input
    )

if __name__ == "__main__":
    main()
