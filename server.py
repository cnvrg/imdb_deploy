from flask import Flask, request, jsonify
import traceback 
import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def run():
    try:
        data = request.get_json(force=True)
        input_params = data['input']
        result =  predict.predict(input_params)
        return jsonify({'prediction': result})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
