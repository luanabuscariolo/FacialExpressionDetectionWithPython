import cv2
import numpy as np
import keyboard

capturar = True
fechar = False

keyboard.on_press_key("p", lambda _: podeCapturar(False))
def podeCapturar(valor):
    global capturar
    capturar = valor

keyboard.on_press_key("r", lambda _: podeCapturar(True))
def podeCapturar(valor):
    global capturar
    capturar = valor

keyboard.on_press_key("x", lambda _: fecharJanela())
def fecharJanela():
    global fechar
    fechar = True

def reconhecedorFaceExpressao():
    global fechar
    global capturar

    camera = cv2.VideoCapture(1)

    cascade_faces = 'Materiall/Material/haarcascade_frontalface_default.xml'
    caminho_modelo = 'Materiall/Material/modelo_01_expressoes.h5'

    detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read("classificadorLBPH.yml")

    largura, altura = 220, 220
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    def tirar_foto():
        conectado, imagem = camera.read()
        cv2.imshow("Face", imagem)
        return imagem

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array

    # Carrega o modelo
    face_detection = cv2.CascadeClassifier(cascade_faces)
    classificador_emocoes = load_model(caminho_modelo, compile=False)

    expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

    print("Para pausar aperte P")
    print("Para continuar aperte R")
    print("Para fechar aperte X")

    while (True):
          if capturar == True:
                imagemOriginal = tirar_foto()
                cinza = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2GRAY)
                cv2.putText(imagemOriginal, "Para fechar aperte X", org=(30, 470), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
                cv2.imshow('Face', imagemOriginal)
                facesDetectadas = detectorFace.detectMultiScale(cinza, scaleFactor=1.5, minSize=(150, 150))
                #print(str(len(facesDetectadas)) + " faces detectadas")

                for (fX, fY, fW, fH) in facesDetectadas:
                    imagemFace = cv2.resize(cinza[fY:fY + fH, fX:fX + fW], (largura, altura))
                    cv2.rectangle(imagemOriginal, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                    id, confianca = reconhecedor.predict(imagemFace)
                    nome = ""
                    if id == 1:
                        nome = 'Luana'
                    elif id == 2:
                        nome = 'Vitor'
                    cv2.putText(imagemOriginal, str(id) + "-" + nome, (fX, fY + (fH + 30)), font, 2, (0, 0, 255))
                    #cv2.putText(imagemOriginal, str(confianca), (fX, fY + (fH + 50)), font, 1, (0, 0, 255))

                    roi = cinza[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    preds = classificador_emocoes.predict(roi)[0]
                    print(preds)
                    emotion_probability = np.max(preds)
                    label = expressoes[preds.argmax()]
                    cv2.putText(imagemOriginal, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(imagemOriginal, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

                    cv2.putText(imagemOriginal, "Para pausar aperte P", org=(30, 410), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
                    cv2.putText(imagemOriginal, "Para continuar aperte R", org=(30, 440), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)

                    cv2.imshow("Face", imagemOriginal)

                    probabilidades = imagemOriginal.copy()

                    # Mostra gráfico apenas se detectou uma face
                    if len(facesDetectadas) == 1:
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

                    #cv2.imwrite("captura.jpg", probabilidades)

                cv2.waitKey(1)

          if fechar == True:
            cv2.destroyAllWindows()
            break

    #cv2.waitKey(0)

#reconhecedorFaceExpressao()