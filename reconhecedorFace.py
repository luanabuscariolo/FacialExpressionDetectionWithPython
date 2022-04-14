import cv2

camera = cv2.VideoCapture(1)
detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")

#EIGENFACE
#reconhecedor = cv2.face.EigenFaceRecognizer_create()
#reconhecedor.read("classificadorEigen.yml")

#LBPHFACE
#reconhecedor = cv2.face.LBPHFaceRecognizer_create()
#reconhecedor.read("classificadorLBPH.yml")

#MELHOR RECONHECEDOR DOS 3 CONFORME TESTE EM FOTOS TIRADAS
#FISHERFACE
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("classificadorFisher.yml")

largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150, 150))

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x +l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = 'Luana'
        elif id == 2:
            nome = 'Vitor'
        cv2.putText(imagem, nome, (x,y + (a+30)), font, 2, (0,0,255))
        cv2.putText(imagem, str(confianca), (x, y + (a+50)), font, 1, (0,0,255))

    cv2.imshow("Face", imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.waitKey(0)
cv2.destroyAllWindows()