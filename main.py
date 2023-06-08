from flask import Flask, request, jsonify
import os
from datetime import datetime
from Controladores.ControladorModelo import ControladorModelo
import json
app = Flask(__name__)

from flask_cors import CORS
cors = CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + file.filename
        file.save(os.path.join('Archivos/', filename))
        return jsonify({'message': 'Archivo guardado exitosamente.'}), 200
    else:
        return jsonify({'error': 'No se recibió ningún archivo.'}), 400


if __name__ == '__main__':
    app.run(debug=False,port=9000)