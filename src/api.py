import openai
from flask import Flask, request, jsonify
import os

# Set your OpenAI API Key from environment variable
openai.api_key = os.getenv('Your_OpenAI_API_Key')

app = Flask(__name__)

@app.route('/get_market_analysis', methods=['GET'])
def get_market_analysis():
    # Get user message from the query parameter
    user_message = request.args.get('message', default="Hello!", type=str)
    
    # Make the API call to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can change this to the model of your choice
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )
    
    # Extract the analysis response from OpenAI's response
    analysis = response['choices'][0]['message']['content']

    # Return the analysis as JSON response
    return jsonify({"analysis": analysis})

if __name__ == '__main__':
    app.run(debug=True)
