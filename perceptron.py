
'''
Aluno: Gabriel de Souza Nogueira da Silva
Matricula: 398847
'''

import re
from scipy import stats
import numpy as np

quant_neuronio = 3
vet_atributos = []  # vetor de entradas
vet_respostas = []  # vetor de saidas
one_out = 0
valor_tirado_att = np.ones((1, 4))
valor_tirado_resp = np.ones((1, 3))
cont = 0



def normaliza(x):
    vet = list()
    vet1 = list()
    vet2 = list()
    vet3 = list()
    vet4 = list()
    cont1 = 0
    cont2 = 1
    cont3 = 2
    cont4 = 3
    for i in range(0, len(x)):
        if cont1 == i:
            vet1.append(x[i])
            cont1 = cont1 + 4
        if cont2 == i:
            vet2.append(x[i])
            cont2 = cont2 + 4
        if cont3 == i:
            vet3.append(x[i])
            cont3 = cont3 + 4
        if cont4 == i:
            vet4.append(x[i])
            cont4 = cont4 + 4

    vet1 = norm(vet1)
    vet2 = norm(vet2)
    vet3 = norm(vet3)
    vet4 = norm(vet4)

    for i in range(0, len(vet1)):
        vet.append(vet1[i])
        vet.append(vet2[i])
        vet.append(vet3[i])
        vet.append(vet4[i])

    return vet


def norm(x):
    return stats.zscore(x)


dados = open("iris_log.dat", "r")

for line in dados:
    # separando o que é x do que é d
    line = line.strip()  # quebra no \n
    line = re.sub('\s+', ',', line)  # trocando os espaços vazios por virgula
    # quebra nas virgulas e retorna 2 valores
    a1, a2, a3, a4, r1, r2, r3 = line.split(",")
    vet_atributos.append(float(a1))
    vet_atributos.append(float(a2))
    vet_atributos.append(float(a3))
    vet_atributos.append(float(a4))
    vet_respostas.append(float(r1))
    vet_respostas.append(float(r2))
    vet_respostas.append(float(r3))

dados.close()

vet_atributos = normaliza(vet_atributos)


def cria_mat(vet_atributos, vet_respostas):
    # embaralha as minhas amostras para o treinamento
    mat = np.zeros((150, 7))
    k = 0
    for i in range(0, 150):
        for j in range(0, 4):
            mat[i][j] = vet_atributos[k]
            k += 1
    k = 0
    for i in range(0, 150):
        for j in range(4, 7):
            mat[i][j] = vet_respostas[k]
            k += 1

    mat = np.random.permutation(mat)

    k = 0
    for i in range(0, 150):
        for j in range(0, 4):
            vet_atributos[k] = mat[i][j]
            k += 1

    k = 0
    for i in range(0, 150):
        for j in range(4, 7):
            vet_respostas[k] = mat[i][j]
            k += 1


def cria_mat_atributos(vet_atributos):
    # crio a matriz de atributos retirando umas das amostras da base de dados
    # para ser o teste o valor do teste é salvo na variavel valor_tirado_att
    # e é retornado o uma matriz com todas as outras amostras
    global one_out
    k = 0
    vet = np.ones((int(len(vet_atributos)/4), 4))

    vet_1 = np.ones((int(len(vet_atributos)/4)-1, 4))

    for i in range(0, int(len(vet_atributos)/4)):
        for j in range(0, 4):
            vet[i][j] = vet_atributos[k]
            k += 1

    for q in range(0, 4):
        valor_tirado_att[0][q] = vet[one_out][q]

    aux = list()
    for i in range(0, int(len(vet_atributos)/4)):
        for j in range(0, 4):
            if i != one_out:
                aux.append(vet[i][j])
    k = 0
    for i in range(0, int(len(aux)/4)):
        for j in range(0, 4):
            vet_1[i][j] = aux[k]
            k += 1

    return vet_1


def cria_mat_atributos_peso_bias(mat_atributos):
    # coloca o peso do bias na matriz de atributos
    vet = np.ones((5, 149))
    aux = np.transpose(mat_atributos)

    for i in range(1, 5):
        for j in range(0, 149):
            vet[i][j] = aux[i-1][j]
    return vet


def cria_mat_resposta(vet_resposta):
    # crio a matriz de respostas retirando uma das amostras da base de dados
    # para ser a resposta do valor do teste é salvo na variavel valor_tirado_resp
    # e é retornado o uma matriz com todas as outras respostas das amostras
    global one_out
    k = 0
    vet = np.ones((int(len(vet_resposta)/3), 3))

    vet_1 = np.ones((int(len(vet_resposta)/3)-1, 3))

    for i in range(0, int(len(vet_resposta)/3)):
        for j in range(0, 3):
            vet[i][j] = vet_resposta[k]
            k += 1

    for q in range(0, 3):
        valor_tirado_resp[0][q] = vet[one_out][q]

    aux = list()
    for i in range(0, int(len(vet_resposta)/3)):
        for j in range(0, 3):
            if i != one_out:
                aux.append(vet[i][j])
    k = 0
    for i in range(0, int(len(aux)/3)):
        for j in range(0, 3):
            vet_1[i][j] = aux[k]
            k += 1

    return vet_1


def cria_mat_w():
    vet = np.ones((quant_neuronio, 5))
    for i in range(0, quant_neuronio):
        for j in range(0, 5):
            # valores aleatorios em uma distribuiçao normal
            vet[i][j] = 0  
    return vet


def treinamento(atributos_com_peso_bias, resposta):
    # treina a rede atualizando os valores da matriz de pesos com base no erro

    global W
    mat = np.zeros((5, 1))
    resp = np.zeros((3, 1))

    for k in range(0, 149):
        for i in range(0, 5):
            mat[i][0] = atributos_com_peso_bias[i][k]

        for i in range(0, 3):
            resp[i][0] = resposta[k][i]

        u = np.dot(W, mat)
        y = np.zeros((3, 1))

        for i in range(0, 3):
            if u[i][0] > 0:
                y[i][0] = 1
            else:
                y[i][0] = 0

        # calc erro
        e = np.zeros((3, 1))
        for i in range(0, 3):
            e[i][0] = resp[i][0] - y[i][0]
        for i in range(0, 3):
            for j in range(0, 5):
                # atualiza a matriz de pesos
                W[i][j] = W[i][j] + 0.5*e[i][0]*mat[j][0]


def testar():
    # testa o valor retirado das amostras na rede treinada e retorna se acertou ou nao
    global cont, W
    teste_com_peso_bias = np.ones((1, 5))
    for i in range(1, 5):
        teste_com_peso_bias[0][i] = valor_tirado_att[0][i-1]

    u = np.dot(W, np.transpose(teste_com_peso_bias))
    y = np.zeros((3, 1))

    for i in range(0, 3):
        if u[i][0] > 0:
            y[i][0] = 1
        else:
            y[i][0] = 0

    aux = 0

    for i in range(0, 3):

        a = y[i][0]
        b = valor_tirado_resp[0][i]

        if a == b:
            aux += 1

    if aux == 3:
        # print("Acertou!")
        cont += 1
    else:
        pass
        # print("Errou!")

W = cria_mat_w()

while one_out < 150:
    # quantidade de vezes que a base é mostrada para a rede
    for i in range(0, 5):

        cria_mat(vet_atributos, vet_respostas)

        atributo = cria_mat_atributos(vet_atributos)

        respostas = cria_mat_resposta(vet_respostas)

        atributos_com_peso_bias = cria_mat_atributos_peso_bias(atributo)

        treinamento(atributos_com_peso_bias, respostas)

    testar()

    one_out += 1

print("Acuracia " + str((cont/150)*100) + "% " +
      "Quant. de amostras acertadas: " + str(cont))
