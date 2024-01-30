#
# from flask import Flask, render_template, request, jsonify
# import openai
#
# app = Flask(__name__)
#
# # Replace 'YOUR_API_KEY' with your actual OpenAI API key
# openai.api_key = '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'
#
# @app.route('/')
# def index():
#     return render_template('chat1.html')
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     user_message = request.form['message']
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": user_message}]
#         )
#         return jsonify({'response': response.choices[0].message['content']})
#     except Exception as e:
#         return jsonify({'response': str(e)})
#
# if __name__ == '__main__':
#     app.run(debug=True)




#
# from flask import Flask, render_template, request, jsonify
# import Lesson_Planner  # Import your model and database setup
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return render_template('chat3.html')
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     user_message = request.json['message']
#     expand = request.json.get('expand', False)
#     paragraph_index = request.json.get('paragraphIndex', None)
#
#     if expand and paragraph_index is not None:
#         # Here you should handle the expansion of the specific paragraph
#         # This is a placeholder for the expansion logic
#         expanded_text = "Expanded content for paragraph " + str(paragraph_index)
#         return jsonify({'response': expanded_text})
#
#     # Normal query processing
#     search_results = Lesson_Planner.db.similarity_search(user_message, k=4)
#     result = Lesson_Planner.llm_chain.run({"question": user_message, "input_documents": search_results})
#     response = result  # Adjust according to the actual structure of result
#
#     return jsonify({'response': response})
#
# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import openai
import Lesson_Planner  # Assuming this contains your model setup, including llm_chain and db

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
    search_results = Lesson_Planner.db.similarity_search(user_message, k=4)
    result = Lesson_Planner.llm_chain.run({"question": user_message, "input_documents": search_results})
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
