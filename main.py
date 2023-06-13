from flask import Flask, request, jsonify
import os
from datetime import datetime
from Controladores.ControladorModelo import ControladorModelo
import json
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

app = Flask(__name__)

from flask_cors import CORS
cors = CORS(app)

myClient = pymongo.MongoClient("mongodb+srv://ricardo:admin123@clusterinteligentes.5onsapb.mongodb.net/")
myDb = myClient["Inteligentes"]
myCol = myDb["Datos"]
df = "" 
filename = ""

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        global filename
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + file.filename
        file.save(os.path.join('Archivos/', filename))
        global df
        df = pd.read_excel(file)
        fileJson = df.to_json()
        task = {"documento": fileJson}
        myCol.insert_one(task)
        
        columnas = df.columns
        for columna in columnas:
            if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                df = df.dropna(subset=[columna])
            if df[columna].dtypes == 'int64' or df[columna].dtypes == 'float64':
                print(df[columna].mean())
                df[columna] = df[columna].fillna(df[columna].mean())
        return jsonify({'message': 'Archivo guardado exitosamente.'}), 200
    else:
        return jsonify({'error': 'No se recibió ningún archivo.'}), 400
    
@app.route('/graficar', methods=['POST'])
def graficar():
    try:
        df_numeric = df.select_dtypes(include='number')
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        df_numeric.hist()
        filenameH = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + 'histograma.png'
        plt.savefig('Imagenes/histogramas/' + filenameH)

        colormap = plt.cm.coolwarm
        plt.figure(figsize=(12,12))
        plt.title('Chronic_Kidney_Disease Data Set', y=1.05, size=15)
        sb.heatmap(df_numeric.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
        filenameC = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + 'correlacion.png'
        plt.savefig('Imagenes/correlacion/' + filenameC)
        return jsonify({'histograma': 'Imagenes/histogramas/' + filenameH,
                        'correlacion':'Imagenes/correlacion/' + filenameC}), 200
    except AttributeError :
        return jsonify({'error': 'No se ha cargado ningún archivo.'}), 400
    except Exception:
        return jsonify({'error': 'Ha ocurrido un error.'}), 400

@app.route('/entrenar', methods=['POST'])
def entrenar():
    try:
        body = request.get_json()
        x = df[body['x']]
        y = df[body['y']]
        normalizacion = body['normalizacion']
        tecnica = body['tecnica']
        numero = body['numero']
        X_train, X_test, y_train, y_test = [], [], [], []
        if tecnica == 'hold':
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = numero/100, random_state = 101)
        elif tecnica == 'cross':
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/numero, random_state = 101)

        if normalizacion == 'standard':
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
        elif normalizacion == 'minmax':
            escalar=MinMaxScaler()
            X_train=escalar.fit_transform(X_train)
            X_test=escalar.transform(X_test)

        modelo = ControladorModelo().entrenar(body['modelo'])
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        ruta = "Models/"+ filename + body['modelo']+".h5"
        modelo.save(ruta)

        return jsonify({'message': 'Entrenamiento exitoso.'}), 200
    except AttributeError :
        return jsonify({'error': 'No se ha cargado ningún archivo.'}), 400
    except Exception:
        return jsonify({'error': 'Ha ocurrido un error.'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    print("predict")
    return jsonify({'message': 'Predicción exitosa.'}), 200

if __name__ == '__main__':
    app.run(debug=False,port=9000)