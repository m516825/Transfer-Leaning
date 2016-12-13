import argparse 
import time 
import os
import sys
import numpy as np
import random
from scipy import sparse
import subprocess

source_file = 'source.txt'
train_file = 'train.txt'
test_file = 'test.txt'

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default='./test1', type=str)
	parser.add_argument('--predict_file', default='output.txt', type=str)
	parser.add_argument('--CV', default=0, type=int)
	parser.add_argument('--iter', default=100, type=int)
	parser.add_argument('--hidden', default=100, type=int)
	parser.add_argument('--target_mpath', default="./t_model.txt", type=str)
	parser.add_argument('--source_mpath', default="./s_model.txt", type=str)
	parser.add_argument('--new_target_path', default="./g_target.txt", type=str)
	parser.add_argument('--new_target_mpath', default="./g_model.txt", type=str)
	parser.add_argument('--valid', default=0, type=int)
	parser.add_argument('--auto_valid', default=0, type=int)
	args = parser.parse_args()

	return args

class MF(object):
	def __init__(self, args):
		self.args = args
		self.cv = args.CV
		self.source_data = []
		self.target_data = []
		self.test_data = []
		self.nmf_model = None

		self.load_data()

	def load_data(self):	
		self.source_path = os.path.join(self.args.data_dir, source_file)
		self.target_path = os.path.join(self.args.data_dir, train_file)
		self.test_path = os.path.join(self.args.data_dir, test_file)

		with open(self.source_path, 'r') as f:
			for line in f.readlines():
				line = line.strip().split()
				self.source_data.append([int(line[0]), int(line[1]), float(line[2])])
		with open(self.target_path, 'r') as f:
			for line in f.readlines():
				line = line.strip().split()
				self.target_data.append([int(line[0]), int(line[1]), float(line[2])])

		if self.args.auto_valid == 1:

			self.test_path = os.path.join('./', 'v_'+test_file)
			self.target_path = os.path.join('./', 'v_'+train_file)
			random.shuffle(self.target_data)
			cvlen = len(self.target_data)/5
			self.test_data = self.target_data[:cvlen]
			self.target_data = self.target_data[cvlen:]
			with open(self.target_path, 'w') as f:
				for u, i, v in self.target_data:
					f.write(str(u)+' '+str(i)+' '+str(v)+'\n')
			with open(self.test_path, 'w') as f:
				for u, i, v in self.test_data:
					f.write(str(u)+' '+str(i)+' '+str(v)+'\n')
		else:	
			with open(self.test_path, 'r') as f:
				for line in f.readlines():
					line = line.strip().split()
					self.test_data.append([int(line[0]), int(line[1])])
			print >> sys.stderr, 'done loading data'

	def cross_validation(self):

		hidden = 100
		iteration = 20
		cv = self.cv
		train_path = self.target_path
		model_path = "./cv_model.txt"

		exec_str = "./libmf-2.01/mf-train -f 0 -l2 0.05 -k {} -t {} -v {} {} {}".format(hidden, iteration, \
																				cv, train_path, model_path)
		subprocess.call(exec_str.split())

	def train(self, train_path=None, model_path=None):
		print >> sys.stderr, 'start training'
		if train_path == None:
			train_path = self.target_path
		if model_path == None:
			model_path = self.args.target_mpath

		hidden = self.args.hidden
		iteration = self.args.iter

		exec_str = "./libmf-2.01/mf-train -f 0 -l2 0.05 -k {} -t {} {} {}".format(hidden, iteration, \
																				train_path, model_path)
		subprocess.call(exec_str.split())

	def predict(self, test_path=None, model_path=None):
		print >> sys.stderr, 'start predicting'
		if test_path == None:
			test_path = self.test_path
		if model_path == None:
			model_path = self.args.target_mpath

		reformat_path = "./my_test.txt"
		fout = open(reformat_path, 'w')
		for line in open(test_path, 'r'):
			line = line.strip().split()
			line[2] = "0.0"
			fout.write(' '.join(line)+'\n')
		fout.close()
		if self.args.valid == 1:
			reformat_path = test_path 

		exec_str = "./libmf-2.01//mf-predict -e 0 {} {} {}".format(reformat_path, model_path, self.args.predict_file)

		subprocess.call(exec_str.split())


	def eval(self, pred, ans):
		pred = np.array(pred)
		ans = np.array(ans)
		RMSE = np.sqrt(((pred - ans)**2).mean())
		return RMSE

	def get_decompose_matrix(self, model_path=None):
		print >> sys.stderr, 'loading decompose matrice P, Q'
		if model_path == None:
			model_path = self.args.target_mpath
		P = []
		Q = []
		for i, line in enumerate(open(model_path, 'r')):
			if i > 4:
				info = line.strip().split()
				value = map(float, info[2:])
				if info[0][0] == 'p':
					P.append(value)
				if info[0][0] == 'q':
					Q.append(value)
		P = np.array(P, dtype='float32')
		Q = np.array(Q, dtype='float32')

		print >> sys.stderr, 'P shape: {}, Q shape: {}'.format(P.shape, Q.shape)

		return P, Q

	def MF_2_SVD(self, P, Q, K=50, K2=10, Iter=None):
		print >> sys.stderr, 'start transfroming MF to SVD format'
		import scipy.sparse.linalg as ssl
		svd_p = ssl.svds(P, k=K, maxiter=Iter)
		svd_q = ssl.svds(Q, k=K, maxiter=Iter)
		###########sorted in ascending order############
		svd_p = [np.fliplr(svd_p[0]), np.flipud(svd_p[1]), np.flipud(svd_p[2])]
		svd_q = [np.fliplr(svd_q[0]), np.flipud(svd_q[1]), np.flipud(svd_q[2])]
		print >> sys.stderr, 'svd p: {}, {}, {}'.format(svd_p[0].shape, svd_p[1].shape, svd_p[2].shape)
		print >> sys.stderr, 'svd q: {}, {}, {}'.format(svd_q[0].shape, svd_q[1].shape, svd_q[2].shape)
		
		singular = np.zeros((K, K))
		np.fill_diagonal(singular, svd_p[1])
		svd_p[1] = singular
		singular = np.zeros((K, K))
		np.fill_diagonal(singular, svd_q[1])
		svd_q[1] = singular
		del singular

		Up = svd_p[0]
		Uqh = svd_q[0].T
		matrix_list = [svd_p[1], svd_p[2], svd_q[2].T, svd_q[1].T]
		com_matrix = reduce(np.dot, matrix_list) 
		del svd_p, svd_q, matrix_list

		print >> sys.stderr, 'transfrom shape: {}, {}, {}'.format(Up.shape, com_matrix.shape, Uqh.shape )

		svd_com = ssl.svds(com_matrix, k=K2, maxiter=Iter)
		svd_com = [np.fliplr(svd_com[0]), np.flipud(svd_com[1]), np.flipud(svd_com[2])]
		U = np.dot(Up, svd_com[0])
		D = svd_com[1]
		Vh = np.dot(svd_com[2], Uqh)
		del Up, Uqh, svd_com

		singular = np.zeros((len(D), len(D)))
		np.fill_diagonal(singular, D)
		D = singular
		print >> sys.stderr, 'final shape: {}, {}, {}'.format(U.shape, D.shape, Vh.shape)

		return U, D, Vh

	def cal_G_user_item(self, U1, D1, Vh1, U2, D2, Vh2, K=None):
		print >> sys.stderr, 'start calculating G user and G item'
		from sklearn.neighbors import NearestNeighbors

		def G_error(z1, G, z2):
			error = 0.
			for i, z1_v in enumerate(z1):
				# print z1_v, z2[G[i]]
				error += ((z1_v - z2[G[i]])**2).sum()
			return error

		# G user
		S_user = []
		G_uesr = []
		z1 = np.dot(U1, np.sqrt(D1))
		z2_s = np.dot(U2, np.sqrt(D2))
		
		for k in range(K):
			k += 1
			# s +1
			tmp_s = S_user[:]
			tmp_s.append(1.)	
			singular_s = np.zeros((k, k))
			np.fill_diagonal(singular_s, np.array(tmp_s))

			z2_k_p = np.dot(z2_s[:, :k], singular_s)
			# K-NN
			neigh = NearestNeighbors(n_neighbors=1)
			neigh.fit(z2_k_p)

			G_p = []
			n_s = neigh.kneighbors(z1[:,:k], 1, return_distance=False)
			for i, index in enumerate(n_s):
				G_p.append(int(index))	
			# s -1
			tmp_s = S_user[:]
			tmp_s.append(-1.)	
			singular_s = np.zeros((k, k))
			np.fill_diagonal(singular_s, np.array(tmp_s))

			z2_k_m = np.dot(z2_s[:, :k], singular_s)
			# K-NN
			neigh = NearestNeighbors(n_neighbors=1)
			neigh.fit(z2_k_m)

			G_m = []
			n_s = neigh.kneighbors(z1[:, :k], 1, return_distance=False)
			for i, index in enumerate(n_s):
				G_m.append(int(index))

			alpha_p = G_error(z1[:, :k], G_p, z2_k_p)
			alpha_m = G_error(z1[:, :k], G_m, z2_k_m)
			print alpha_p, alpha_m
			if alpha_p > alpha_m:
				G_uesr.append(G_m)
				S_user.append(-1)
			else:
				G_uesr.append(G_p)
				S_user.append(1)
		
		# G item
		S_item = []
		G_item = []
		z1 = np.dot(Vh1.T, np.sqrt(D1))
		z2_s = np.dot(Vh2.T, np.sqrt(D2))
		
		for k in range(K):
			k += 1
			# s +1
			tmp_s = S_item[:]
			tmp_s.append(1.)	
			singular_s = np.zeros((k, k))
			np.fill_diagonal(singular_s, np.array(tmp_s))

			z2_k_p = np.dot(z2_s[:, :k], singular_s)
			# K-NN
			neigh = NearestNeighbors(n_neighbors=1)
			neigh.fit(z2_k_p)

			G_p = []
			n_s = neigh.kneighbors(z1[:,:k], 1, return_distance=False)
			for i, index in enumerate(n_s):
				G_p.append(int(index))	
			# s -1
			tmp_s = S_item[:]
			tmp_s.append(-1.)	
			singular_s = np.zeros((k, k))
			np.fill_diagonal(singular_s, np.array(tmp_s))

			z2_k_m = np.dot(z2_s[:, :k], singular_s)
			# K-NN
			neigh = NearestNeighbors(n_neighbors=1)
			neigh.fit(z2_k_m)

			G_m = []
			n_s = neigh.kneighbors(z1[:, :k], 1, return_distance=False)
			for i, index in enumerate(n_s):
				G_m.append(int(index))

			alpha_p = G_error(z1[:, :k], G_p, z2_k_p)
			alpha_m = G_error(z1[:, :k], G_m, z2_k_m)
			print alpha_p, alpha_m
			if alpha_p > alpha_m:
				G_item.append(G_m)
				S_item.append(-1)
			else:
				G_item.append(G_p)
				S_item.append(1)

		return G_uesr, G_item

	def matching(self):
		print >> sys.stderr, 'start transfer two matrix'

		def min_G_user_item(R1, G_user, R2_, G_item, length):
			item = np.zeros((len(G_item), length), dtype='float32')
			for i, index in enumerate(G_item):
				item[i, index] = 1.
			R2_ = np.dot(R2_, item.T)

			R2 = []
			for index in G_user:
				R2.append(R2_[index])
			del R2_

			R2 = np.array(R2, dtype='float32')

			error = 0.
			for u, i, v in R1:
				error += (v - R2[u, i])**2
			error /= float(len(R1))
			error = np.sqrt(error)

			del R2

			return error


		self.train(train_path=self.target_path, model_path=self.args.target_mpath)
		self.train(train_path=self.source_path, model_path=self.args.source_mpath)

		P1, Q1 = self.get_decompose_matrix(model_path=self.args.target_mpath)
		P2, Q2 = self.get_decompose_matrix(model_path=self.args.source_mpath)

		U1, D1, Vh1 = self.MF_2_SVD(P=P1, Q=Q1)
		U2, D2, Vh2 = self.MF_2_SVD(P=P2, Q=Q2)

		s = time.time()
		G_user, G_item = self.cal_G_user_item(U1=U1, D1=D1, Vh1=Vh1, U2=U2, D2=D2, Vh2=Vh2, K=10)
		print >> sys.stderr, 'take {} s to calculate G user and G item'.format(time.time() - s)

		for gu in G_user:
			tmp = {}
			Unique = 0
			for i in gu:
				tmp[i] = tmp.get(i, 0) + 1
			for k, v in tmp.iteritems():
				if v == 1:
					Unique += 1
			print 'Unique: {}'.format(Unique)
		del tmp

		s = time.time()
		error = []
		R2_ = np.dot(P2, Q2.T).astype('float32')
		for Gu, Gi in zip(G_user, G_item):
			e = min_G_user_item(self.target_data, Gu, R2_, Gi, len(Vh2.T))
			error.append(e)
			print e
		min_g = np.argmin(np.array(error))
		print >> sys.stderr, 'take {} s to get best G user and G item'.format(time.time() - s)

		with open('./g_user.txt', 'w') as f:
			for gu in G_user[min_g]:
				f.write(str(gu)+'\n')

		with open('./g_item.txt', 'w') as f:
			for gi in G_item[min_g]:
				f.write(str(gi)+'\n')
		return 0

		# R2_ = np.zeros((len(P2), len(Q2)), dtype='float32')
		# for u, i, v in self.source_data:
		# 	R2_[u, i] = v

		item = np.zeros((len(G_item[min_g]), len(Vh2.T)), dtype='float32')
		for i, index in enumerate(G_item[min_g]):
				item[i, index] = 1.
		R2_ = np.dot(R2_, item.T)

		R1 = []
		for index in G_user[min_g]:
			R1.append(R2_[index])

		# del R2_, item

		R1 = np.array(R1)

		error_user = {}
		error_num = {}
		for u, i, v in self.target_data:
			try:
				error_user[u].append((v-R1[u, i])**2)
			except:
				error_user[u] = []
				error_user[u].append((v-R1[u, i])**2)
		for k, v in error_user.iteritems():
			error_user[k] = np.sqrt((np.array(v)).mean())
			error_num[k] = len(v)
		for k, v in sorted(error_user.iteritems(), key=lambda x:x[1]):
			print k, v
		##############################################################
		# R2_ = np.zeros((len(P2), len(Q2)), dtype='float32')
		# for u, i, v in self.source_data:
		# 	R2_[u, i] = v

		# item = np.zeros((len(G_item[min_g]), len(Vh2.T)), dtype='float32')
		# for i, index in enumerate(G_item[min_g]):
		# 		item[i, index] = 1.
		# R2_ = np.dot(R2_, item.T)

		R1 = []
		for index in G_user[min_g]:
			if error_user.get(index, 0) < 1.0:
				R1.append(R2_[index])
			else:
				R1.append(np.array([0.]*len(R2_[index])))
		R1 = np.array(R1)

		##############################################################
		# pred_path = os.path.join(self.args.data_dir, 'out.txt')
		# fo = open(pred_path, 'w')
		# for line in open(self.test_path, 'r'):
		# 	line = line.strip().split()
		# 	value = R1[int(line[0]), int(line[1])]
		# 	fo.write(str(value)+'\n')
		# sys.exit(0)

		for u, i, v in self.target_data:
			R1[u, i] = v

		with open(self.args.new_target_path, 'w') as f:
			for u, arr in enumerate(R1):
				for i, value in enumerate(arr):
					if value > 0.:
						out = str(u)+' '+str(i)+' '+str(value)+'\n'
						f.write(out)

		self.train(train_path=self.args.new_target_path, model_path=self.args.new_target_mpath)
		self.predict(model_path=self.args.new_target_mpath)
		

def main():
	start = time.time()

	args = arg_parse()

	model = MF(args)

	if args.CV > 0:
		model.cross_validation()
	else:
		model.matching()
		# model.train()
		model.predict()
		# model.train(train_path=args.new_target_path, model_path=args.new_target_mpath)
		# model.predict(model_path=args.new_target_mpath)


	print >> sys.stderr, 'time cost: {}'.format(time.time()-start)

if __name__ == '__main__':
	main()