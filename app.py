import os
import pickle
import json
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from models.builder import GetDummies
from sklearn.base import BaseEstimator, TransformerMixin
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
    def loan_application(loan_amount: float, loan_term: float, interest_rate: float, loan_installment:float, annual_income: float, debt_to_income_ratio: float, earliest_credit_line: float, open_credit_accounts: float, derogatory_record: bool, 
                    revolving_balance: float, revolving_utilization: float, total_accounts: float, mortgage_account: bool, past_bankruptcy: bool, loan_grade: str, ownership_status: str, verification_status: str) -> str:
        "Function that takes in feature inputs about and individual to input to a trained Machine Learning model. The Machine Learning model then predicts if the user will or will not be accepted for the loan application."

        #save all numeric features for new entry
        data_entry_numeric = {
            "loan_amnt": [loan_amount],
            "term": loan_term,
            "int_rate": interest_rate,
            "installment": loan_installment,
            "annual_inc": annual_income,
            "dti": debt_to_income_ratio,
            "earliest_cr_line": earliest_credit_line,
            "open_acc": open_credit_accounts,
            "pub_rec": int(derogatory_record),
            "revol_bal": revolving_balance,
            "revol_util": revolving_utilization,
            "total_acc": total_accounts,
            "mort_acc": int(mortgage_account),
            "pub_rec_bankruptcies": int(past_bankruptcy)
        }
        #save all the categorical features for the new entry
        data_entry_categorical = {
            "sub_grade": [loan_grade],
            "home_ownership": ownership_status,
            "verification_status": verification_status
        }
        #build dataframe from numeric entries
        df_numeric_entry = pd.DataFrame(data_entry_numeric)
        #build dataframe from categorical entries
        df_categorical_entry = pd.DataFrame(data_entry_categorical)

        #build dummies using loaded dummies builder
        df_categorical_entry = loaded_builder.transform(df_categorical_entry)
        #concat the new entry into one
        X_new = pd.concat([df_numeric_entry, df_categorical_entry], axis=1)
        #load in the scaler
        with open("models/xgboost_scaler.pkl", "rb") as f:
            loaded_scaler = pickle.load(f)
        #scale the new entry
        X_new = loaded_scaler.transform(X_new)
        #load in the xgboost model
        with open("models/xgboost_clf.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        #predict the current entry
        prediction = int(loaded_model.predict(X_new)[0])
        #get the probability of the entry
        probability = loaded_model.predict_proba(X_new)[0][prediction] * 100

        #identify credit risk of the user
        if prediction:
            #identify low risk
            if probability >= 85:
                user_risk = "Low Risk"
            #identify medium risk
            elif probability >= 70 and probability < 85:
                user_risk = "Medium Risk"
            #identify low risk
            else:
                user_risk = "High Risk"
        else:
            #user got denied therefore default very high risk
            user_risk = "Very High Risk"

        #return result based on prediction
        if prediction:
            #user got approved
            return f"Status: Approved\nLoan Payoff Probability: {probability:.2f}\nRisk Category: {user_risk}"
        else:
            #user did not get approved
            return f"Status: Denied\nLoan Default Probability: {probability:.2f}\nRisk Category: {user_risk}"

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

        with gr.Column():
            with gr.Row():
                pub_rec = gr.Radio(["Yes", "No"], value="No", label="Does User Have A Derogatory Record?", interactive=True)

                mort_acc = gr.Radio(["Yes", "No"], value="Yes", label="Does The User Have A Mortgage Account?", interactive=True)

                bankruptcies = gr.Radio(["Yes", "No"], value="No", label="Does The User Have Any Bankruptcies", interactive=True)

        with gr.Column():
            with gr.Row():
                home_ownership = gr.Radio(["Own", "Mortgage", "Rent", "Other"], value="Rent", label="Home Ownership Status", interactive=True)

                verification = gr.Radio(["Verified", "Source Verified", "Not Verified"], value="Verified", label="Verification Status", interactive=True)

    with gr.Row():
        #built grade letters
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        #save a list of possible grades for the loan
        possible_grades = []
        #build out the full list
        for curr_letter in letters:
            for i in range(1, 6):
                #save current iteration of possible grade
                possible_grades.append(curr_letter + str(i))

        loan_grade = gr.Dropdown(possible_grades, value="B4", multiselect=False, label="Loan Grade", interactive=True)

        earliest_credit_line = gr.Dropdown(list(range(1944, 2014)), value=2000, multiselect=False, label="Users Earliest Credit Line", interactive=True)

        open_accounts = gr.Number(label="Number of Open Accounts", minimum=1, maximum=40, value=1, interactive=True, precision=0)

        total_acc = gr.Number(label="Total Credit Accounts", minimum=1, maximum=80, value=1, interactive=True, precision=0)

    with gr.Row():
        with gr.Column(scale=1):
            loan_amount = gr.Number(label="Loan Amount", minimum=500, maximum=40000.0, value=25000, interactive=True)

            loan_term = gr.Radio(["36 Months", "60 Months"], value="36 Months", label="Users Loan Term", interactive=True)

            interest_rate = gr.Slider(5, 31, step=0.1, value=14.5, label="Loan Interest Rate", interactive=True)

            income = gr.Number(label="Users Annual Income", minimum=4000, maximum=250000.0, value=50000, interactive=True)

            dti = gr.Slider(0, 50, step=0.1, value=25.5, label="Users Debt To Income Ratio", interactive=True)

            revol_bal = gr.Number(label="User Total Credit Revolving Balance", minimum=0, maximum=250000, value=20000, interactive=True)

            revol_util = gr.Slider(0, 100, step=0.1, value=30, label="Revolving Utilization Rate", interactive=True)

            apply_button = gr.Button("Apply", interactive=True)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=460)
            
            msg = gr.Textbox(scale=0)

            with gr.Row():
                msg_clear = gr.Button("Clear", interactive=True)

                chat_button = gr.Button("Chat", interactive=True)
    
    # ----- Functions -----

    #save user message
    def user(msg, chatbot):
        #append the user message
        chatbot.append({"role": "user", "content": msg})
        return chatbot
    
    #use fills out loan form and clicks apply
    def loan_request(loan_amount, loan_term, interest_rate, income, dti, earliest_credit_line, open_accounts, pub_rec, mort_acc, bankruptcies, revol_bal, revol_util, total_acc, loan_grade, home_ownership, verification, chatbot):
        #loan term data conversion
        term_conversion = {
            "36 Months": 36.0,
            "60 Months": 60.0
        }
        #conver loan term
        term_converted = term_conversion[loan_term]
        #calculate the monthly payment for the loan
        r = round((interest_rate / 100) / 12, 6)
        #calculate the top of the formula
        top = r * ((1 + r) ** term_converted)
        #calculate the botom of the formula
        bottom = ((1 + r) ** term_converted) - 1
        #calculate monthly installment
        loan_installment = loan_amount * (top / bottom)
        #build a data dictionary
        data_dict = {
            "loan_amount": loan_amount,
            "loan_term": term_converted,
            "interest_rate": interest_rate,
            "loan_installment": loan_installment,
            "annual_income": income,
            "debt_to_income_ratio": dti,
            "earliest_credit_line": earliest_credit_line,
            "open_credit_accounts": open_accounts,
            "derogatory_record": pub_rec == "Yes",
            "revolving_balance": revol_bal,
            "revolving_utilization": revol_util,
            "total_accounts": total_acc,
            "mortgage_account": mort_acc == "Yes",
            "past_bankruptcy": bankruptcies == "Yes",
            "loan_grade": loan_grade,
            "ownership_status": home_ownership.upper(),
            "verification_status": verification,
        }
        #run the data on the pre-trained machine learning model
        application_result = loan_application.invoke(data_dict)
        #build the agent message to feed to our ReAct agent
        agent_message = f"""You are consulting to a loan officer that needs to get more information about a loan applicant. You have the following data dictionary about a user that applied for a loan through our state of the art Machine Learning Model. 
        
        Data Dictionary: {data_dict}

        The Machine Learning Model returned the following result.

        Result: {application_result}

        Summarize the findings to the loan officer. Make sure to pull in relevant information from the user stored in the data dictionary. Make sure to follow these requirements:
        - Summarize the users financial profile (like income, debt, utilization) and how these attributes may have contributed to the result.
        - Summarize the users historical profile (like earliest credit line, public records) and how these attributes may have contributed to the result.
        - Finally summarize the users loan request information like (loan amount, loan term) compare this using the interest rate to the users income and debt utilization in and summarize how they affected the result. Make sure to mention the loan installment amount and how that compares to the users financial profile.
        """
        #append user message
        chatbot.append({"role": "user", "content": "Submitted Loan Application Form!"})
        #call the agent for a response
        response = agent.invoke({
            "messages": [HumanMessage(content=agent_message)]
        }, config)
        #add response to the chatbot interface
        chatbot.append({"role": "assistant", "content": beautify_output(response) + "\n\n---- Machine Learning Model Result ----\n" + application_result + "\n-----------"})
        return chatbot
    
    #build out the interpretation step for the AI agent
    def interpretation_step(chatbot):
        #build out the SHAP values
        shap_values = {
            "Interest Rate": "+0.49",
            "Loan Term": "+0.15",
            "Revolving Utilization": "+0.22",
            "Annual Income": "+0.17",
            "Monthly Installment": "+0.15",
            "Mortgage Account": "+0.27",
            "Earliest Credit Line": "+0.04",
            "Debt To Income Ratio": "+0.15",
            "Previous Bankruptcies": "+0.01",
            "Derogatory Public Records": "+0.03"
        }
        #build the prompt
        agent_message = f"""Based on what you know about the users credit profile and the prediction response from the machine learning model, use your knowledge of the SHAP values of the
        machine learning model to give a summary of what features from the user made the most impact on the prediction. Make sure to reference the SHAP values in your model interpretation. Please meet
        the following requirements:
        - Mention what feature made the most impact on the prediction and how changes in this feature would impact the prediction.
        - Mention how features impact each other.
        - Provide a summary of the user based on your knowledge of the features meeting these requirements: your overall recommendation if a loan should be given, what features about the user give you sense that they will or will not pay the loan back.

        SHAP Values for each feature: {shap_values}

        Keep in mind that the Machine Learning model is classifying if a user based on their features will Fully Pay Off their loan (Class: 1) or Default on their loan (Class: 0).
        """
        #call the agent for a response
        response = agent.invoke({
            "messages": [HumanMessage(content=agent_message)]
        }, config)
        #add response to the chatbot interface
        chatbot.append({"role": "assistant", "content": beautify_output(response)})
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
    def check_input(loan_amount, income, open_accounts, revol_bal, total_acc, chatbot):
        #check for missing data
        if None in [loan_amount, income, open_accounts, revol_bal, total_acc]:
            raise gr.Error("Make there are no missing values in the form!", duration=5)
        #add message to chatbot
        chatbot.append({"role": "user", "content": "Form Submitted"})
        #return values unchanged
        return loan_amount, income, open_accounts, revol_bal, total_acc, chatbot
    
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
        check_input, [loan_amount, income, open_accounts, revol_bal, total_acc, chatbot], [loan_amount, income, open_accounts, revol_bal, total_acc, chatbot]
    ).success(
        loan_request, [loan_amount, loan_term, interest_rate, income, dti, earliest_credit_line, open_accounts, pub_rec, mort_acc, bankruptcies, revol_bal, revol_util, total_acc, loan_grade, home_ownership, verification, chatbot], chatbot
    ).then(
        interpretation_step, chatbot, chatbot
    )

    #handle user clicks apply failure
    apply_button.click(
        check_input, [loan_amount, income, open_accounts, revol_bal, total_acc, chatbot], [loan_amount, income, open_accounts, revol_bal, total_acc, chatbot]
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

    #load the dummies builder pickle file
    with open("models/dummies_builder.pkl", "rb") as f:
            loaded_builder = pickle.load(f)

    demo.launch(show_error=True)