import cv2
import numpy as np
import argparse
from module_net import MobileNetDetector

class DangerDetector(MobileNetDetector):  # Agora herdamos de MobileNetDetector
    def __init__(self, CONFIDENCE=0.7):
        # Inicialize a classe Pai
        super().__init__(CONFIDENCE=CONFIDENCE)
        # Ajuste o valor de confiança para o valor que você achar melhor
        self.CONFIDENCE = CONFIDENCE

    def separar_caixa_entre_animais(self, img: np.ndarray, resultados: list): #-> (np.ndarray, dict):
        # img (np.ndarray): Imagem de saida - as detecções são desenhadas apenas se "self.draw = True"
        # resultados ( list(dict) ): Lista de dicionários com as detecções (classe, confidence, bbox(x1, y1, x2, y2))
        #{'classe': 'cow', 'confidence': 0.9906375, 'bbox': (379, 131, 181, 120)},
        """Não mude ou renomeie esta função."""
        img = img.copy()
        animais = {'vaca': [], 'lobo': [],'n_lobos':0, 'n_vacas':0}
        n_lobos=0
        n_vacas=0
        xys=[]

        for elemento in resultados:

            if elemento["classe"]=="cow":
                n_vacas+=1
                
                x_esquerda_vaca, y_esquerda_vaca, x_direita_vaca, y_direita_vaca = elemento["bbox"]

                animais["vaca"].append(elemento)
            else:
                x_esquerda_lobo_atual, y_esquerda_lobo_atual, x_direita_lobo_atual, y_direita_lobo_atual = elemento["bbox"]
                if n_lobos == 0:
                    x_esquerda_lobo, y_esquerda_lobo, x_direita_lobo, y_direita_lobo = x_esquerda_lobo_atual, y_esquerda_lobo_atual, x_direita_lobo_atual, y_direita_lobo_atual
                else:
                    x_esquerda_lobo = min(x_esquerda_lobo, x_esquerda_lobo_atual)
                    y_esquerda_lobo = min(y_esquerda_lobo, y_esquerda_lobo_atual)
                    x_direita_lobo = max(x_direita_lobo, x_direita_lobo_atual)
                    y_direita_lobo = max(y_direita_lobo, y_direita_lobo_atual)
                n_lobos+=1
                animais["lobo"].append(elemento)
        animais['n_lobos']=n_lobos
        animais['n_vacas']=n_vacas
        if n_vacas>0:
            img = cv2.rectangle(img, (x_esquerda_vaca, y_esquerda_vaca), (x_direita_vaca, y_direita_vaca), (0, 255, 0), 2)
        if n_lobos>0: 
            img =cv2.rectangle(img, (x_esquerda_lobo, y_esquerda_lobo), (x_direita_lobo, y_direita_lobo), (255, 0,0 ), 2)
            xys = (x_esquerda_lobo, y_esquerda_lobo, x_direita_lobo, y_direita_lobo)
            if n_vacas > 0 and n_lobos > 0:
                if x_esquerda_lobo < x_esquerda_vaca < x_direita_lobo and x_esquerda_lobo < x_direita_vaca < x_direita_lobo and y_esquerda_lobo < y_esquerda_vaca < y_direita_lobo and y_esquerda_lobo < y_direita_vaca < y_direita_lobo:
                    animais['perigo'] = True
                else:
                    animais['perigo'] = False
        else:
            animais['perigo']=False
        
            


        return img, animais

    def calcula_iou(self, boxA: list, boxB: list) -> float:
        """Calcula a intersecção sobre a união (Intersection over Union - IoU) entre duas caixas."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def checar_perigo(self, image: np.ndarray, animais: dict) -> np.ndarray:
        """Não mude ou renomeie esta função."""
        return image

def main():
    import time
    bgr = cv2.imread("img/cow_wolf02.png")

    Detector = DangerDetector()
    image, results = Detector.detect(bgr)
    print (results)

    image, animais = Detector.separar_caixa_entre_animais(image, results)
    image = Detector.checar_perigo(image, animais)
    cv2.imshow("Result_MobileNet", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
