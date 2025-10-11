import os
import pickle
import json
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

with gr.Blocks() as demo:
    #config environment variable
    MESSAGE_INDEX = 0
    # ----- Agentic Functions -----
    @tool
    def loan_application(dependents: int, income: int, loan_amount: int, loan_term: int, credit_score: int, graduate: bool, self_employed: bool) -> str:
        "Function that takes in feature inputs about and individual to input to a trained Machine Learning model. The Machine Learning model then predicts if the user will or will not be accepted for the loan application."

        #load in machine learning model
        classifier = pickle.load(open("models/clf1.pkl", "rb"))
        #load in standard scaler
        scaler = pickle.load(open("models/clf1_scaler.pkl", "rb"))

        #build a new data entry
        entry = pd.DataFrame(
            [[dependents, income, loan_amount, loan_term, credit_score, graduate, self_employed]], 
            columns=['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'education_ Graduate', 'self_employed']
        )
        #scale the new entry
        entry = scaler.transform(entry)
        #make a prediction based on the input
        result = classifier.predict(entry)
        #return the result
        if bool(result[0]):
            return "You got approved!"
        return "Unfortunately you did not get approved."

    @tool
    def search_web(query: str) -> list | None:
        "Function that allows for searching the web for information"

        try:
            #search the web using tavily
            search = TavilySearchResults(max_results=3)
            #get the results
            results = search.invoke(query)
            #return results if they exist
            if results:
                return results
            return None
        except Exception as e:
            return None
        
    #create our openai model
    model = init_chat_model(model="gpt-4.1-mini", model_provider="openai")
    #create the tools for our ReAct agent
    tools = [loan_application, search_web]
    #build the ReAct agent
    agent = create_react_agent(model=model, tools=tools, checkpointer=MemorySaver())
    #create the config for memory saving
    config = {"configurable": {"thread_id": "lesson_3_thread"}}

    
    # ----- Helper Functions -----
    #helper function to create agent output
    def beautify_output(agent_response):
        #access global variable
        global MESSAGE_INDEX
        #build response string
        response_string = ""
        #loop through message and build a response
        for message in agent_response["messages"][MESSAGE_INDEX:]:
            #increment message_index
            MESSAGE_INDEX += 1
            #check to see if we have an AI message
            if type(message) == AIMessage:
                #check to see if there is content in the AIMessage
                if message.content:
                    #build the response string
                    response_string = message.content + response_string
            #check to see if we have a tool message
            elif type(message) == ToolMessage:
                #check to see what tool we used
                if message.name == "search_web":
                    #get the tool call results
                    web_results = json.loads(message.content)
                    #update response string
                    response_string = response_string + "\n\nResources:\n"
                    #loop through the web results
                    for curr_result in web_results:
                        response_string = response_string + curr_result["url"] + "\n"
                #update message if used ML model
                elif message.name == "loan_application":
                    #update response string
                    response_string = response_string + "\n\nMachine Learning Model Used!"
        return response_string

    # ----- Components -----
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                graduate = gr.Radio(["Yes", "No"], value="Yes", label="Education", info="Are you a college graduate?", interactive=True)
                
                self_employed = gr.Radio(["Yes", "No"], value="No", label="Employment", info="Are you self employed?", interactive=True)
            
            with gr.Row():
                dependents = gr.Radio(["0", "1", "2", "3", "4", "5+"], value="0", label="Dependents", info="How many dependents do you have?", interactive=True)
            
            credit_score = gr.Slider(300, 850, precision=0, step=1, value=650, label="Credit Score", info="What is your credit score?", interactive=True)

            income = gr.Number(label="Income", info="Your annual income", minimum=0, maximum=10000000, value=50000, interactive=True)

            loan_term = gr.Slider(2, 20, precision=0, step=1, value=5, label="Loan Term", info="Loan Length (In Years)", interactive=True)

            loan_amount = gr.Number(label="Amount", info="The requested loan amount", minimum=100, maximum=40000000, value=5000, interactive=True)

            with gr.Row():
                apply_button = gr.Button("Apply", interactive=True)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages")
            
            msg = gr.Textbox()

            with gr.Row():
                msg_clear = gr.Button("Clear", interactive=True)

                chat_button = gr.Button("Chat", interactive=True)
    
    # ----- Functions -----

    #save user message
    def user(msg, chatbot):
        #append the user message
        chatbot.append({"role": "user", "content": msg})
        return chatbot
    
    def loan_request(chatbot):
        print("ran loan request")
        chatbot.append({"role": "user", "content": "Loan Request!"})
        chatbot.append({"role": "assistant", "content": "Looks like you submitted a loan request!\n\nStarting Up The Loan Request Process!"})
        return chatbot
    
    #chat with the chatbot
    def bot(chatbot):
        #get the content of the user message
        user_message = chatbot[-1]["content"]
        #call the agent for a response
        response = agent.invoke({
            "messages": [HumanMessage(content=user_message)]
        }, config)

        #add response to the chatbot interface
        chatbot.append({"role": "assistant", "content": beautify_output(response)})
        return chatbot
    
    #check input when applying for job
    def check_input(income, loan_amount, chatbot):

        #check to for missing data
        if loan_amount is None or income is None:
            raise gr.Error("Make sure to fill out the entire form!", duration=5)

        new_message = f"Loan Amount: {loan_amount}\nIncome: {income}"

        chatbot.append({"role": "user", "content": new_message})

        return income, loan_amount, chatbot
    
    def reset_form(income, loan_amount, chatbot):
        print("ran reset")
        income = gr.Number(label="Income", info="Your annual income", minimum=0, maximum=10000000, value=50000, interactive=True)

        loan_amount = gr.Number(label="Amount", info="The requested loan amount", minimum=100, maximum=40000000, value=5000, interactive=True)

        return income, loan_amount, chatbot

    #clear chatbot history
    def clear_chat(chatbot):
        #reset the history
        chatbot = []
        return chatbot
    
    def clear_textbox(msg):
        msg = gr.Textbox(value=None, interactive=True)
        return msg

    # ----- Actions -----

    # #handle user clicks apply success
    apply_button.click(
        check_input, [income, loan_amount, chatbot], [income, loan_amount, chatbot]
    ).success(
        loan_request, chatbot, chatbot
    )

    #handle user clicks apply failure
    apply_button.click(
        check_input, [income, loan_amount, chatbot], [income, loan_amount, chatbot]
    ).failure(
        reset_form, [income, loan_amount, chatbot], [income, loan_amount, chatbot]
    )

    #handle user clicks clear message
    msg_clear.click(
        clear_chat, chatbot, chatbot
    )

    gr.on(
        triggers=[msg.submit, chat_button.click],
        fn=user,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(
        clear_textbox, msg, msg
    ).then(
        bot, chatbot, chatbot
    )

if __name__ == "__main__":
    demo.launch(show_error=True)