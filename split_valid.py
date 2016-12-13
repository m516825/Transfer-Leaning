import sys
import os 
import subprocess
import random

if __name__ == '__main__':

	data_dir = sys.argv[1]
	cv = sys.argv[2]
	valid_dir = './valid'

	if not os.path.exists(valid_dir):
		os.makedirs(valid_dir)

	t_test_path = os.path.join(valid_dir, 'test.txt')
	t_train_path = os.path.join(valid_dir, 'train.txt')
	s_source_path = os.path.join(data_dir, 'source.txt')
	s_train_path = os.path.join(data_dir, 'train.txt')

	command = 'cp '+s_source_path+' '+valid_dir
	subprocess.call(command.split())

	ftest = open(t_test_path, 'w')
	ftrain = open(t_train_path, 'w')

	train = []
	for line in open(s_train_path, 'r'):
		line = line.strip()
		train.append(line)
	random.shuffle(train)

	size = len(train)/int(cv)
	test = train[:size]
	train = train[size:]
	for s in test:
		ftest.write(s+'\n')

	for s in train:
		ftrain.write(s+'\n')




