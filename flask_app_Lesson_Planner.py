
from flask import Flask, render_template, request, jsonify
import openai
import Lesson_Planner  
import os
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

    # Normal query processing
    search_results = Lesson_Planner.db.similarity_search(user_message, k=4)
    result = Lesson_Planner.llm_chain.run({"question": user_message, "input_documents": search_results})
    response = result  

    return jsonify({'response': response})

def send_expansion_request_to_gpt4(expansion_prompt):

    prompt = f"Expand the following text with real life example and incidents: \"{expansion_prompt}\""

    response = openai.ChatCompletion.create(
        openai_api_key= os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message['content']
if __name__ == '__main__':
    app.run(debug=True)
