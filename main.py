import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from surprise import dump
import uuid

app = Flask(__name__)
CORS(app)

# Cargar los datos
tabla_user = pd.read_csv('tabla_user.csv')
tabla_track = pd.read_csv('tabla_track.csv')
tabla_artist = pd.read_csv('tabla_artist.csv')

@app.route('/registro', methods=['POST'])
def registrar_usuario():

    global tabla_user

    datos = request.json
    nombre_usuario = datos.get('nombre_usuario') 
    edad = datos.get('edad')  
    genero = datos.get('genero')
    pais = datos.get('pais')
    
    if nombre_usuario not in tabla_user['nombre'].values:
        # Generar un nuevo ID de usuario único
        nuevo_usuario_id = str(uuid.uuid4())
        nuevo_usuario = pd.DataFrame({
            'user-id': [nuevo_usuario_id],
            'nombre': [nombre_usuario],
            'edad': [edad],  # Asegúrate de añadir estos campos al DataFrame si es necesario
            'genero': [genero],
            'pais': [pais],
            'fecha_registro': [datetime.datetime.now().strftime('%b %d, %Y')]
        })
        tabla_user = pd.concat([tabla_user, nuevo_usuario], ignore_index=True)
        return jsonify({"mensaje": "Usuario registrado exitosamente", "user-id": nuevo_usuario_id}), 200
    else:
        return jsonify({"mensaje": "El nombre de usuario ya existe"}), 400

@app.route('/recomendacion', methods=['POST'])
def recomendar_canciones():
    datos = request.json
    nombre_usuario = datos['nombre_usuario']
    
    if nombre_usuario in tabla_user['nombre'].values:
        user_id = tabla_user[tabla_user['nombre'] == nombre_usuario]['user-id'].values[0]
        _, modelo = dump.load('./model')
        
        # Usuario existente: Hacer predicciones
        recomendaciones = []
        for _, track in tabla_track.iterrows():
            track_id = track['track-id']
            prediccion = modelo.predict(user_id, track_id)
            artist_name = tabla_artist[tabla_artist['artist-id'] == track['artist-id']]['artist-name'].values[0]
            recomendaciones.append({
                'track_name': track['track-name'],
                'artist_name': artist_name,
                'count': track['count'],
                'prediccion': prediccion.est
            })
        
        # Ordenar las recomendaciones por la predicción, de mayor a menor
        recomendaciones_ordenadas = sorted(recomendaciones, key=lambda x: x['prediccion'], reverse=True)
    else:
        # Usuario nuevo: Devolver canciones ordenadas por popularidad
        recomendaciones_ordenadas = tabla_track.merge(tabla_artist, on='artist-id').sort_values(by='count', ascending=False).head(10).to_dict('records')
        # Eliminar campos innecesarios como 'artist-id' y 'track-id' si se desea
        for rec in recomendaciones_ordenadas:
            rec.pop('artist-id', None)
            rec.pop('track-id', None)

    return jsonify(recomendaciones_ordenadas)

if __name__ == '__main__':
    app.run(debug=True)