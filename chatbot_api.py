pip install flask

from flask import Flask,request,jsonify
from chat import chatbot_response
import requests

app=Flask(__name__)


@app.route('/api')
def get_response():
    
                                                                
    r = requests.patch(url, jsonify(elevator1))
##    if "text" in request.args:
##        s = request.args["text"]
##        return jsonify({
##            'response': chatbot_response(s)
##        })
##
##    else:
##        return jsonify({
##            'response': 'Not a valid text. Please try again.'
##        })


if __name__=="__main__":
    app.run(debug=True)






