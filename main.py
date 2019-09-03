#!/usr/bin/python
#-*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import time, sys


inicio_seq = time.time()
# Seedando os numeros aleatorios para garantir que as listas são iguais
np.random.seed(333)

# Inicializando MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
######################

# Parâmetros
n_pontos = 500000
n_classes = 3
classes = ['r', 'b', 'g']
k = int(round(np.log2(n_pontos))) # log2 do numero total de pontos
#################################

# Lista de pontos iniciada aleatóriamente
pontos = [[np.random.rand(), np.random.rand(), np.random.choice(classes)] for _ in range(n_pontos)]

# Ponto de teste inserido aleatóriamente
ponto_teste = [np.random.rand(), np.random.rand()]

# Mestre espera os resultados
if rank == 0:
	print("K =", k)

	start_time = time.time()

	# Antigo esquema de envio (desnecessário pois os intervalos são baseados no ranking)
	'''for i in range(1, size):
		#data = pontos[(i - 1)*n_pontos/(size - 1): i*n_pontos/(size - 1)]
		data = [(i - 1)*n_pontos/(size - 1), i*n_pontos/(size - 1)]
		comm.send(data, dest=i)'''

	# Recuperando os resultados e armazenando na lista "vizinhos"
	vizinhos = []
	for i in range(1, size):
		data = comm.recv(source=i)
		for d in data:
			vizinhos.append(d)

	# Ordenando a lista que no momento possui k * n_escravos elementos
	vizinhos.sort()

	# Pegando os k primeiros itens da lista e então temos os k vizinhos mais próximos
	k_vizinhos = vizinhos[0:k]

	# Criamos um dicionário no formato {classe : [n_ocorrencias, distancia_acumulada]}
	knn_dict = {x : [0, 0] for x in classes}
	for v in k_vizinhos:
		knn_dict[v[1]][0] += 1
		knn_dict[v[1]][1] += v[0]
	##################################################################################


	# Isolamos as informações de acordo com o o maior numero de ocorrencias e inversamente para a distancia acumulada (+0.001 para não haver divisão por 0)
	classe, dados = max(iter(knn_dict.items()), key=(lambda k: (k[1][0], 1/(k[1][1] + 0.001))))
	fim = time.time() - start_time
	tempo_seq = time.time() - inicio_seq
	# Imprimindo informações
	sys.stdout.flush()
	print("\nO ponto", [round(elem, 4) for elem in ponto_teste], "foi classificado", classe, "com", dados[0], "vizinhos.\n")
	print("Tempo:", time.time() - start_time)
	print("Speedup: ", tempo_seq/fim)
	#######################################################################################################################

	# Plotando os resultados
	'''print("Deseja plotar os resultados? [s/N]")
	sys.stdout.flush()
	if raw_input().strip()[0].lower() == 's':
		import matplotlib.pyplot as plt
		for v in k_vizinhos:
			plt.plot([ponto_teste[0], v[2]], [ponto_teste[1], v[3]], c='k', linewidth=0.5)
		plt.scatter([x[0] for x in pontos if x[2] == 'r'], [x[1] for x in pontos if x[2] == 'r'], c='r', label='[r] Vermelho')
		plt.scatter([x[0] for x in pontos if x[2] == 'b'], [x[1] for x in pontos if x[2] == 'b'], c='b', label='[b] Azul')
		plt.scatter([x[0] for x in pontos if x[2] == 'g'], [x[1] for x in pontos if x[2] == 'g'], c='g', label='[g] Verde')
		plt.scatter(ponto_teste[0], ponto_teste[1], marker='^', s=50, label='Teste', c=classe)
		plt.legend()
		plt.show()'''
	################################################################################################################################

else:
	# Antigo esquema de recebimento de dados (desnecessário
	'''data = comm.recv(source=0)
	print "Escravo", rank, "trabalhando com", data[1] - data[0], "pontos.'''

	# Calculando cada distancia e armazenando as informações [distancia, classe, x, y]
	dists = []
	for p in pontos[int((rank - 1)*n_pontos/(size - 1)) : int(rank*n_pontos/(size - 1))]:
		d = np.sqrt((ponto_teste[0] - p[0])**2 + (ponto_teste[1] - p[1])**2)
		dists.append([d, p[2], p[0], p[1]])


	# Ordenando a lista
	dists.sort()
	# Enviando os k primeiros itens da lista ordenada para o mestre
	comm.send(dists[0:k], dest=0)
