from flask import Flask, jsonify, request

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
    print(src, tgt)
    pred = 'Hi, how are you?'


    print(pred)

    return jsonify(pred=pred)