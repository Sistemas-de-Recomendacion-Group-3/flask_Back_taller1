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
tabla_user['es_nuevo'] = False
tabla_track = pd.read_csv('tabla_track.csv')
tabla_artist = pd.read_csv('tabla_artist.csv')
#Interaccioens activas usuario-cancion
historial_calificaciones = pd.DataFrame(columns=['user_id', 'track_id', 'rating'])

@app.route('/verificar_usuario', methods=['POST'])
def verificar_usuario():
    datos = request.json
    nombre_usuario = datos['nombre_usuario']

    if nombre_usuario in tabla_user['nombre'].values:
        return jsonify({"mensaje": "Usuario encontrado"}), 200
    else:
        return jsonify({"mensaje": "Usuario no encontrado"}), 404

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
            'edad': [edad],
            'genero': [genero],
            'pais': [pais],
            'fecha_registro': [datetime.datetime.now().strftime('%b %d, %Y')],
            # Marcar como nuevo usuario
            'es_nuevo': [True]
        })
        tabla_user = pd.concat([tabla_user, nuevo_usuario], ignore_index=True)
        return jsonify({"mensaje": "Usuario registrado exitosamente", "user-id": nuevo_usuario_id}), 200
    else:
        return jsonify({"mensaje": "El nombre de usuario ya existe"}), 400

@app.route('/verificar_usuario_nuevo/<nombre_usuario>', methods=['GET'])
def verificar_usuario_nuevo(nombre_usuario):
    if nombre_usuario in tabla_user['nombre'].values:
        es_nuevo = tabla_user.loc[tabla_user['nombre'] == nombre_usuario, 'es_nuevo'].values[0]
        return str(es_nuevo), 200
    else:
        return "False", 404

@app.route('/guardar_calificacion', methods=['POST'])
def guardar_calificacion():
    datos = request.json
    user_id = datos['user_id']
    track_id = datos['track_id']
    rating = datos['rating']

    if user_id in tabla_user['user-id'].values and track_id in tabla_track['track-id'].values and 1 <= rating <= 5:
        nueva_calificacion = {'user_id': user_id, 'track_id': track_id, 'rating': rating}
        historial_calificaciones = historial_calificaciones.append(nueva_calificacion, ignore_index=True)
        return jsonify({"mensaje": "Calificación guardada exitosamente"}), 200
    else:
        return jsonify({"mensaje": "Error al guardar la calificación. Asegúrate de que el usuario y la canción existan, y la calificación esté entre 1 y 5."}), 400

@app.route('/interacciones_usuario/<nombre_usuario>', methods=['GET'])
def obtener_interacciones_usuario(nombre_usuario):
    if nombre_usuario in tabla_user['nombre'].values:
        user_id = tabla_user.loc[tabla_user['nombre'] == nombre_usuario, 'user-id'].values[0]
        interacciones_usuario = historial_calificaciones[historial_calificaciones['user_id'] == user_id].to_dict('records')
        return jsonify(interacciones_usuario), 200
    else:
        return jsonify({"mensaje": "Usuario no encontrado"}), 404
    
@app.route('/recomendacion', methods=['POST'])
def recomendar_canciones():
    datos = request.json
    nombre_usuario = datos['nombre_usuario']
    
    if nombre_usuario in tabla_user['nombre'].values:
        user_data = tabla_user.loc[tabla_user['nombre'] == nombre_usuario]
        user_id = user_data['user-id'].values[0]
        es_nuevo = user_data['es_nuevo'].values[0]
        _, modelo = dump.load('./model')
        
        # Usuario existente: Hacer predicciones
        if not es_nuevo:
            recomendaciones = []
            for _, track in tabla_track.iterrows():
                track_id = track['track-id']
                prediccion = modelo.predict(user_id, track_id)
                # Escalar la calificación de 1 a 5 a un porcentaje de 0 a 100
                porcentaje_prediccion = round(prediccion.est / 5 * 100, 2) - 0.01
                artist_name = tabla_artist.loc[tabla_artist['artist-id'] == track['artist-id'], 'artist-name'].values[0]
                recomendaciones.append({
                    'track_name': track['track-name'],
                    'artist_name': artist_name,
                    'count': track['count'],
                    'prediccion': porcentaje_prediccion
                })
            
            # Ordenar las recomendaciones por el porcentaje de predicción, de mayor a menor
            recomendaciones_ordenadas = sorted(recomendaciones, key=lambda x: x['prediccion'], reverse=True)
        else:
            recomendaciones_ordenadas = tabla_track.merge(tabla_artist, on='artist-id').sort_values(by='count', ascending=False).head(10).to_dict('records')
            for rec in recomendaciones_ordenadas:
                rec.pop('artist-id', None)
                rec.pop('track-id', None)
    else:
        # Usuario no encontrado, devolver un mensaje de error
        return jsonify({"mensaje": "Usuario no encontrado"}), 404

    return jsonify(recomendaciones_ordenadas)

if __name__ == '__main__':
    app.run(debug=True)