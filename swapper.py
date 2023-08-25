import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__>='0.7'

# Cargar modelos de InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)


# Cargar las imágenes de entrada
image_to_replace = cv2.imread('imagen.jpg')
replacement_face = cv2.imread('cara.jpg')

# Detectar caras en ambas imágenes
faces_to_replace = app.get(image_to_replace)
replacement_faces = app.get(replacement_face)

if len(faces_to_replace) > 0 and len(replacement_faces) > 0:
    # Extraer el primer rostro detectado de cada imagen
    face_to_replace = faces_to_replace[0]
    replacement_face = replacement_faces[0]

    # Realizar el intercambio de rostros
    transformed_face = replacement_face

    # Sustituir el rostro en la imagen original
    res = swapper.get(image_to_replace, face_to_replace, replacement_face, paste_back=True)
    #y1, y2, x1, x2 = face_to_replace.bbox.astype(int)
    #image_to_replace[y1:y2, x1:x2] = transformed_face

    # Guardar la imagen resultante
    cv2.imwrite('imagen_resultante.jpg', res)
else:
    print("No se detectaron rostros en una de las imágenes.")