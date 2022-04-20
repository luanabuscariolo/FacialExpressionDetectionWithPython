#AO EXECUTAR, 3 CLASSIFICADORES SERÃO GERADOS PARA O TREINAMENTO
import cv2
import os
import numpy as np

def treinamentoFotosBD():
    eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=0)
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()

    def getImagemComId ():
        caminhos = [os.path.join('DeteccaoFaceExpressao/fotos', f) for f in os.listdir('DeteccaoFaceExpressao/fotos')]
        #print(caminhos)
        faces = []
        ids = []
        for caminhoImagem in caminhos:
            imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
            id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
            #print(id)
            ids.append(id)
            faces.append(imagemFace)
            #cv2.imshow("Face", imagemFace)
            #cv2.waitKey(10)
        return np.array(ids), faces

    ids, faces = getImagemComId()
    #print(faces)

    print("Treinando...")

    eigenface.train(faces, ids)
    eigenface.write('classificadorEigen.yml')

    fisherface.train(faces, ids)
    fisherface.write('classificadorFisher.yml')

    lbph.train(faces, ids)
    lbph.write('classificadorLBPH.yml')

    print("Treinamento realizado")