import numpy as np 
import os 
import sys
import tensorflow as tf
import argparse
import progressbar as pb
g_user = []
g_item = []

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default='./test1', type=str)
	parser.add_argument('--predict_file', default='./pred.txt', type=str)
	parser.add_argument('--iter', default=100, type=int)
	parser.add_argument('--source_mpath', default="./s_model.txt", type=str)
	parser.add_argument('--g_user', default='./g_user.txt', type=str)
	parser.add_argument('--g_item', default='./g_item.txt', type=str)
	parser.add_argument('--valid', default=0, type=int)
	args = parser.parse_args()
	args.R1 = os.path.join(args.data_dir, 'train.txt')
	args.R2 = os.path.join(args.data_dir, 'source.txt')
	args.test = os.path.join(args.data_dir, 'test.txt')

	return args

class Data(object):
	def __init__(self, R1_info, R2_info, test_data, sample, avg):
		self.R1 = R1_info[0]
		self.R1_length = len(R1_info[0])
		self.R1_current = 0
		self.R1_u = R1_info[1]
		self.R1_i = R1_info[2]
		self.R2_u = R2_info[2]
		self.R2_i = R2_info[3]
		self.P2 = R2_info[0]
		self.Q2 = R2_info[1]
		self.hidden = len(self.P2[0])
		self.valid_data = test_data
		self.sample = sample
		self.avg = avg

	def next_batch(self, size, sample_size):
		
		if self.R1_current == 0:
			np.random.shuffle(self.R1)

		index_u = np.random.choice(self.R1_u, sample_size, replace=True)
		index_i = np.random.choice(self.R1_i, sample_size, replace=True)
		sample = []
		for u, i in zip(index_u, index_i):
			if self.sample.get((int(u), int(i)), 0.) == 0.:
				sample.append([u, i, self.avg, g_user[u], g_item[i], 0.75])
		sample = np.array(sample)

		if self.R1_current + size < self.R1_length:
			train = self.R1[self.R1_current:self.R1_current+size, :]
			train = np.concatenate((train, sample), axis=0)
			np.random.shuffle(train)
			u, i, v, mu, mi, w = train[:, 0], train[:, 1], train[:, 2], train[:, 3], train[:, 4], train[:, 5]
			
			self.R1_current += size

			return u, i, v, mu, mi, w

		else:
			train = self.R1[self.R1_current:, :]
			train = np.concatenate((train, sample), axis=0)
			np.random.shuffle(train)
			u, i, v, mu, mi, w = train[:, 0], train[:, 1], train[:, 2], train[:, 3], train[:, 4], train[:, 5]
			
			self.R1_current = 0
			
			return u, i, v, mu, mi, w

def format_R1(R1):
	'''
		user_id, item_id, value, mapped_R2_user_id, mapped_R2_item_id
	'''
	for i in range(len(R1)):
		user = R1[i][0]
		item = R1[i][1]
		R1[i].append(g_user[user])
		R1[i].append(g_item[item])
		R1[i].append(1.)

	return R1

def get_decompose_matrix(args):
	print >> sys.stderr, 'loading decompose matrice P, Q'
	model_path = args.source_mpath
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

def load_data(args):

	R1_u = R1_i = R2_u = R2_i = 0
	R1 = []
	R2 = []
	test = []
	avg = 0.

	for line in open(args.R1, 'r'):
		line = line.strip().split()
		R1.append([int(line[0]), int(line[1]), float(line[2])])
		avg += float(line[2])
		if int(line[0]) >= R1_u:
			R1_u = int(line[0])
		if int(line[1]) >= R1_i:
			R1_i = int(line[1])
	R1_i += 1
	R1_u += 1
	avg /= float(len(R1))
	print >> sys.stderr, 'avg: {}'.format(avg)

	P2, Q2 = get_decompose_matrix(args)
	R2_i = len(Q2)
	R2_u = len(P2)

	for line in open(args.test, 'r'):
		line = line.strip().split()
		test.append([int(line[0]), int(line[1]), line[2]])

	for line in open(args.g_user, 'r'):
		line = line.strip()
		g_user.append(int(line))

	for line in open(args.g_item, 'r'):
		line = line.strip()
		g_item.append(int(line))

	R1 = format_R1(R1)
	R1 = np.array(R1)

	sample = {}
	for i in R1:
		sample[(int(i[0]), int(i[1]))] = 1

	print >> sys.stderr, 'R1: user {}, item {}'.format(R1_u, R1_i)
	print >> sys.stderr, 'R2: user {}, item {}'.format(R2_u, R2_i)

	return (R1, R1_u, R1_i), (P2, Q2, R2_u, R2_i), test, sample, avg

def prediction_error(R1, P1, Q1):
	error = 0.
	for info in R1:
		error += (float(info[2]) - np.dot(P1[info[0]], Q1[info[1]]))**2
	error /= float(len(R1))
	error = np.sqrt(error)

	return error

def predict(output_path, test, P1, Q1):
	fo = open(output_path, 'w')
	for pair in test:
		out = str(int(pair[0]))+' '+str(int(pair[1]))+' '+str(float(np.dot(P1[pair[0]], Q1[pair[1]])))+'\n'
		fo.write(out)

def model(args, data, iterations, learning_rate, batch_size, beta=0.3, l2=0.05):

	t_u = tf.placeholder(tf.int32, [None])
	t_i = tf.placeholder(tf.int32, [None])
	t_v = tf.placeholder(tf.float32, [None])
	t_mu = tf.placeholder(tf.int32, [None])
	t_mi = tf.placeholder(tf.int32, [None])
	t_w = tf.placeholder(tf.float32, [None])

	P2 = tf.constant(data.P2)
	Q2 = tf.constant(data.Q2)

	with tf.device('/cpu:0'):
		P1 = tf.Variable(tf.random_uniform([data.R1_u, data.hidden], -.1, .1))
		Q1 = tf.Variable(tf.random_uniform([data.R1_i, data.hidden], -.1, .1))
		t_p = tf.nn.embedding_lookup(P1, t_u)
		t_q = tf.nn.embedding_lookup(Q1, t_i)
		t_p2 = tf.nn.embedding_lookup(P2, t_mu)
		t_q2 = tf.nn.embedding_lookup(Q2, t_mi)

	R1_loss = tf.reduce_sum(tf.pow(t_v - tf.reduce_sum(tf.mul(t_p, t_q), reduction_indices=1), 2) * t_w)

	normalize = l2 * (tf.nn.l2_loss(t_p) + tf.nn.l2_loss(t_q))

	P_loss = beta * tf.reduce_sum(tf.atan(tf.pow(t_p - t_p2, 2)))

	Q_loss = beta * tf.reduce_sum(tf.atan(tf.pow(t_q - t_q2, 2)))

	cost = R1_loss + normalize + P_loss + Q_loss

	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

	with tf.Session() as sess:

		init = tf.initialize_all_variables()
		sess.run(init)

		batch_number = data.R1_length/batch_size
		batch_number += 1 if data.R1_length % batch_size > 0 else 0

		for ite in range(iterations):
			print >> sys.stderr, 'Iterations ',ite+1,':'
			v_RMSE = None
			avg_cost = 0.
			pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_number).start()
			for b in range(batch_number):
				
				u, i, v, mu, mi, w = data.next_batch(batch_size, batch_size)
				
				_, c = sess.run([optimizer, cost], feed_dict={t_u:u, t_i:i, t_v:v, t_mu:mu, t_mi:mi, t_w:w})
				
				avg_cost += c/data.R1_length*2

				pbar.update(b+1)
			pbar.finish()

			RMSE = prediction_error(data.R1, P1.eval(), Q1.eval())

			if args.valid == 1:
				v_RMSE = prediction_error(data.valid_data, P1.eval(), Q1.eval())
				print >> sys.stderr, 'cost: {}, RMSE={}, v_RMSE={}'.format(avg_cost, RMSE, v_RMSE)
			else:
				print >> sys.stderr, 'cost: {}, RMSE={}'.format(avg_cost, RMSE)

		predict(args.predict_file, data.valid_data, P1.eval(), Q1.eval())

def main():

	args = arg_parse()

	R1_info, R2_info, test_data, sample, avg = load_data(args)

	data = Data(R1_info, R2_info, test_data, sample, avg)

	model(args=args, data=data, iterations=args.iter, learning_rate=0.1, batch_size=200)

if __name__ == '__main__':
	main()
