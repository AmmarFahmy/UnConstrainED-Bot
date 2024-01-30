

from flask import Flask, render_template, request, jsonify
import openai
import Assessment_rubrics_generator

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
        # Handle the expansion request
        # expansion_prompt = "Please provide a detailed explanation of the following text: \n\n" + user_message
        expanded_response = send_expansion_request_to_gpt4(user_message)
        # expanded_response = send_expansion_request_to_gpt4(expansion_prompt)
        return jsonify({'response': expanded_response})
    if contract:

        return jsonify({'response': "Original unexpanded text for paragraph."})

    # Normal query processing
    search_results = Assessment_rubrics_generator.db.similarity_search(user_message, k=4)
    result = Assessment_rubrics_generator.llm_chain.run({"question": user_message, "input_documents": search_results})
    response = result  # Adjust according to the actual structure of result

    return jsonify({'response': response})

def send_expansion_request_to_gpt4(expansion_prompt):
    # Implement this function to send the prompt to GPT-4
    # For now, this is a placeholder function
    # You'll need to use the OpenAI API to send the request
    # Return the expanded text as a string
    # return "Expanded content based on GPT-4's response for: " + expansion_prompt
    prompt = f"Please provide a detailed explanation of the following text: \"{expansion_prompt}\""

    # Send this prompt to GPT-4
    # Note: You'll need to use the OpenAI API here
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message['content']
if __name__ == '__main__':
    app.run(debug=True)
