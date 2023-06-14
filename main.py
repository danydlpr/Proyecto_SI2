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
from sklearn.preprocessing import LabelEncoder
import joblib  
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import re
import operator

        
app = Flask(__name__)

from flask_cors import CORS
cors = CORS(app)

myClient = pymongo.MongoClient("mongodb+srv://ricardo:admin123@clusterinteligentes.5onsapb.mongodb.net/")
myDb = myClient["Inteligentes"]
myCol = myDb["Datos"]
myMoldel = myDb["Modelos"]
myCods = myDb["Codigos"]
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
        originales=df.copy()
        for columna in columnas:
            if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                labelencoder_X = LabelEncoder()
            
                df[columna] =  labelencoder_X.fit_transform(df[columna])
        aux=[]
        lista=[]
        dicc={}
        for columna in columnas:
            if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':

            
                for valorCod,valorOg in zip(df[columna],originales[columna]):
                    if valorCod not in aux:
                        new={}
                        new["valorCod"]=valorCod
                        new["valorOg"]=valorOg
                        lista.append(new)
                        aux.append(valorCod)
                aux=[]
                dicc[columna]=lista
                lista=[]
        task = {"documento":filename,"codigos": dicc}
        myCods.insert_one(task)

        return jsonify({'message': 'Archivo guardado exitosamente.','nombre':filename}), 200
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

        ruta = "Models/"+ filename.split(".")[0]+"_" + body['modelo']+".pkl"
        joblib.dump(modelo, ruta)
        acc = accuracy_score(y_pred, y_test)
        pre = precision_score(y_pred, y_test, average='macro')
        rec = recall_score(y_pred, y_test, average='micro')
        f1 = f1_score(y_pred, y_test, average='weighted')
        promedio = (acc + pre + rec + f1)/4
        print(acc, pre, rec, f1)
        task = {"accuracy": acc, 
                "precision": pre, 
                "recall": rec,
                "f1": f1,
                "ruta": ruta, 
                "x": body['x'], 
                "y": body['y'], 
                "normalizacion": normalizacion, 
                "tecnica": tecnica,
                "numero": numero, 
                "nombre": filename,
                "promedio": promedio,
                "modelo": body['modelo'],
                }
        myMoldel.insert_one(task)
        
        return jsonify({'message': 'Entrenamiento exitoso.', 'nombre' : ruta}), 200
    except AttributeError :
        return jsonify({'error': 'No se ha cargado ningún archivo.'}), 400
    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error.'}), 400
    
def convertir_a_cadena(documento):
    documento['_id'] = str(documento['_id'])
    return documento
@app.route('/listar', methods=['POST'])
def listar():
    body = request.get_json()
    nombre = body['nombre']
    
    resultados = myMoldel.find({ 'nombre': nombre })
    lista_resultados = []
    for documento in resultados:
        documento = convertir_a_cadena(documento)
        lista_resultados.append(documento['modelo'])

    json_resultados = json.dumps(lista_resultados)

    return  json_resultados, 200
@app.route('/metricas', methods=['POST'])
def metricas():
    body = request.get_json()
    nombre = body['nombre']
    
    resultados = myMoldel.find({ 'nombre': nombre })
    lista_resultados = []
    for documento in resultados:
        documento = convertir_a_cadena(documento)
        lista_resultados.append(documento)

    json_resultados = json.dumps(lista_resultados)

    return  json_resultados, 200
@app.route('/mejores', methods=['POST'])
def mejores():
    body = request.get_json()
    nombre = body['nombre']
    
    resultados = myMoldel.find({ 'nombre': nombre }).sort('promedio', -1).limit(3)
    lista_resultados = {}
    i=1
    for documento in resultados:
        documento = convertir_a_cadena(documento)
        
        lista_resultados['TOP'+str(i)] = documento
        i+=1
        
    json_resultados = json.dumps(lista_resultados)

    return  json_resultados, 200




@app.route('/predecir', methods=['POST'])
def predict():
    try:
        body = request.get_json() 
        modelo = body['modelo']
        documento= body['documento']
        prediccion = body['prediccion']
        clf_rf = joblib.load('Models/'+modelo+'.pkl')
        
        
        doc = myCods.find({ 'documento': documento })
        aux={}
        for documento in doc:
            documento = convertir_a_cadena(documento)
            aux=documento['codigos']
        datos=[]
        y=[]
        for titulo in aux.keys():
            try:
                valor=prediccion[titulo]
                arr=aux[titulo]

                for auxdicc in arr:
                    if auxdicc['valorOg']==valor:
                        datos.append(auxdicc['valorCod'])

                        break
            except:
                y=aux[titulo]
            

        resultado_prediccion = clf_rf.predict([datos])
        
        for res in y:
                    if res['valorCod']==resultado_prediccion.tolist()[0]:
                        resultado_prediccion=res['valorOg']

                        break

        return jsonify({'prediction': resultado_prediccion}), 200
    except ValueError :
        return jsonify({'error': 'Valor no encontrado.'}), 400
        

if __name__ == '__main__':
    app.run(debug=False,port=9000)