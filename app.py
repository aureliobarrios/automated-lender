import gradio as gr

with gr.Blocks() as demo:
    # ----- Helper Functions -----

    # ----- Components -----
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                graduate = gr.Radio(["Yes", "No"], label="Education", info="Are you a college graduate?")
                
                self_employed = gr.Radio(["Yes", "No"], label="Employment", info="Are you self employed?")
            
            with gr.Row():
                dependents = gr.Radio(["0", "1", "2", "3", "4", "5+"], label="Dependents", info="How many dependents do you have?")
            
            credit_score = gr.Slider(300, 850, precision=0, step=1, value=650, label="Credit Score", info="What is your credit score?", interactive=True)

            income = gr.Number(label="Income", info="Your annual income", minimum=0, maximum=10000000)

            loan_term = gr.Slider(2, 20, precision=0, step=1, value=5, label="Loan Term", info="Loan Length (In Years)", interactive=True)

            loan_amount = gr.Number(label="Amount", info="The requested loan amount", minimum=100, maximum=40000000, value=5000, interactive=True)

            with gr.Row():
                clear_button = gr.Button("Clear", interactive=True)
                apply_button = gr.Button("Apply", interactive=True)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox()
    
    # ----- Functions -----

    # ----- Actions -----

if __name__ == "__main__":
    demo.launch(show_error=True)