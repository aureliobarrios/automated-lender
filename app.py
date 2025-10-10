import gradio as gr

with gr.Blocks() as demo:
    # ----- Helper Functions -----

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
    def bot(msg, chatbot):

        chatbot.append({"role": "assistant", "content": "Interacted with chatbot!"})

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
        bot, [msg, chatbot], chatbot
    ).then(
        clear_textbox, msg, msg
    )

if __name__ == "__main__":
    demo.launch(show_error=True)