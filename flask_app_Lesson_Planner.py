from flask import Flask, render_template, request, jsonify
import openai
import Lesson_Planner
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('chat8.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_message = data['message']
    expand = data.get('expand', False)
    contract = data.get('contract', False)
    paragraph_index = data.get('paragraphIndex', None)

    if expand and paragraph_index is not None:

        expanded_response = send_expansion_request_to_gpt4(user_message)

        return jsonify({'response': expanded_response})
    if contract:

        return jsonify({'response': "Original unexpanded text for paragraph."})


    result = Lesson_Planner.llm_chain.run(user_query= user_message)
    response = result

    return jsonify({'response': response})

def send_expansion_request_to_gpt4(expansion_prompt):

    prompt_template = f"Expand the following text with real life example and incidents: \"{expansion_prompt}\""



    prompt = PromptTemplate.from_template(prompt_template)

    expansion_chain = LLMChain(llm=Lesson_Planner.llm, prompt=prompt)
    response= expansion_chain.predict()

    return response
if __name__ == '__main__':
    app.run(debug=True)