#DETECTOR DE EXPRESSÕES FACIAIS COM AS PORCENTAGENS DE CADA EMOÇÃO

import cv2
import numpy as np

camera = cv2.VideoCapture(1)

def tirar_foto():
    conectado, imagem = camera.read()
    cv2.imshow("Face", imagem)
    return imagem

cascade_faces = 'Materiall/Material/haarcascade_frontalface_default.xml'
caminho_modelo = 'Materiall/Material/modelo_01_expressoes.h5'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Carrega o modelo
face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile=False)

expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

while (True):
    # Clique na imagem da webcam para tirar uma foto
    imagemOriginal = tirar_foto()
    imagemColorida = imagemOriginal.copy()
    # Inverte a ordem dos canais (utilizar caso a imagem capturada fique com cores invertidas)
    imagemPosProcessada = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2RGB)

    cv2.imwrite("testecaptura.jpg",imagemPosProcessada)

    faces = face_detection.detectMultiScale(imagemPosProcessada, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    cinza = cv2.cvtColor(imagemPosProcessada, cv2.COLOR_BGR2GRAY)

    if len(faces) > 0:
        for (fX, fY, fW, fH) in faces:
            roi = cinza[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classificador_emocoes.predict(roi)[0]
            print(preds)
            emotion_probability = np.max(preds)
            label = expressoes[preds.argmax()]
            cv2.putText(imagemColorida, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(imagemColorida, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
    else:
        print('Nenhuma face detectada')

    probabilidades = imagemColorida.copy()

    # Mostra gráfico apenas se detectou uma face
    if len(faces) == 1:
        for (i, (emotion, prob)) in enumerate(zip(expressoes, preds)):
            # Nome das emoções
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(probabilidades, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (200, 250, 20), -1)
            cv2.putText(probabilidades, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Face',probabilidades)

    cv2.imwrite("captura.jpg", probabilidades)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)

#cv2.destroyAllWindows()