from flask import Flask, jsonify, request
from inference import infer

app = Flask(__name__)

@app.route('/translate', methods = ['POST'])
def translate():
    """modify/update the information for <user_id>"""
    # you can use <user_id>, which is a str but could
    # changed to be int or whatever you want, along
    # with your lxml knowledge to make the required
    # changes
    data = request.form # a multidict containing POST data
    src = data['src']
    tgt = data['tgt']
    pred = infer([src], [tgt])[0]
    return jsonify(pred=pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True, use_reloader=False)