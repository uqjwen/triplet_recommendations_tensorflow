from keras.layers import Embedding
import tensorflow as tf 
import argparse
import numpy as np 
import sys
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp 
class Model():
	def __init__(self, args):
		
		self.user = tf.placeholder(tf.int32, shape=[args.batch_size,1])
		self.pitem = tf.placeholder(tf.int32, shape=[args.batch_size,1])
		self.nitem = tf.placeholder(tf.int32, shape=[args.batch_size,1])
		self.pgenre = tf.placeholder(tf.float32, shape=[args.batch_size, args.genre_feat_size])
		self.ngenre = tf.placeholder(tf.float32, shape=[args.batch_size, args.genre_feat_size])


		u_embed = Embedding(input_dim = args.n_user, output_dim = args.embedding_size, input_length = 1)
		i_embed = Embedding(input_dim = args.n_item, output_dim = args.embedding_size, input_length = 1)

		user_latent = tf.reshape(u_embed(self.user), [args.batch_size, args.embedding_size])
		pitem_latent = tf.reshape(i_embed(self.pitem), [args.batch_size, args.embedding_size])
		nitem_latent = tf.reshape(i_embed(self.nitem), [args.batch_size, args.embedding_size])



		# self.pscore = tf.reduce_sum(tf.mul(user_latent, pitem_latent),-1)
		# self.nscore = tf.reduce_sum(tf.mul(user_latent, nitem_latent),-1)
		self.pscore = self.scoring(user_latent, pitem_latent, self.pgenre)
		self.nscore = self.scoring(user_latent, nitem_latent, self.ngenre, reuse = True)

		self.loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.pscore - self.nscore)))
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

		# print self.pscore.get_shape()
		for v in tf.trainable_variables():
			print v, v.name
	def scoring(self, user, item, feature, reuse = False):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		vec = tf.concat(1, [user, item, tf.mul(user,item),feature])
		vec_size = vec.get_shape().as_list()[-1]

		with tf.variable_scope("hiden1"):
			w1 = tf.get_variable('w1',[vec_size,64], initializer = tf.random_normal_initializer(stddev = 0.02))
			b1 = tf.get_variable('b1', [64], initializer = tf.constant_initializer(0.0))
			hidden = tf.matmul(vec,w1)+b1

		with tf.variable_scope('score'):
			w2 = tf.get_variable('w2',[32,1], initializer = tf.random_normal_initializer(stddev=0.02))
			b2 = tf.get_variable('b2', [1], initializer = tf.constant_initializer(0.0))
			output = tf.matmul(hidden, w2)+b2
		return output
filepath = '/home/uqjwen/Downloads/ml-100k/'
class DataLoader():
	def __init__(self,batch_size):
		self.batch_size = batch_size

		train = np.genfromtxt(filepath+'ua.base')
		test = np.genfromtxt(filepath+'ua.test')
		data = np.concatenate([train,test], axis=0).astype(np.int32)

		self.n_user = len(np.unique(data[:,0]))
		self.n_item = len(np.unique(data[:,1]))

		data = data[:len(train)]
		self.matrix = sp.lil_matrix((self.n_user, self.n_item))
		for record in data:
			self.matrix[record[0]-1, record[1]-1] = int(record[2]>=4.)
		self.matcoo = self.matrix.tocoo()
		self.train_size = len(self.matcoo.row)

		#######################read genre####################
		fr = open(filepath+'u.item')
		data = fr.readlines()
		fr.close()
		genre_mat = []
		for line in data:
			line = line.strip()
			listfromline = line.split('|')
			genre_mat.append(map(int,listfromline[-19:]))
		self.genre_mat = np.array(genre_mat)
		self.genre_feat_size = self.genre_mat.shape[1]
		# self.matrix = np.zeros((self.n_user, self.n_item))
		# for record in data:
		# 	self.matrix[record[0]-1,record[1]-1] = record[2]
		# print self.n_user, self.n_item

	def next_batch(self):
		begin = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size
		self.pointer+=1
		nitem = np.random.randint(0,self.n_item,(self.batch_size))
		return self.matcoo.row[begin:end].reshape(-1,1), \
				self.matcoo.col[begin:end].reshape(-1,1), self.genre_mat[self.matcoo.col[begin:end]], \
				nitem.reshape(-1,1), self.genre_mat[nitem]
				# np.random.randint(0,self.n_item, (self.batch_size, 1))
		# user = []
		# pitem = []
		# nitem = []
		# for i in range(self.n_user):
		# 	user.append(i)
		# 	a,b = np.random.choice(range(self.n_item), 2, replace = False)
		# 	while self.matrix[i,a]==self.matrix[i,b]:
		# 		a,b = np.random.choice(range(self.n_item),2,replace = False)
		# 	if self.matrix[i,a]>self.matrix[i,b]:
		# 		pitem.append(a)
		# 		nitem.append(b)
		# 	else:
		# 		pitem.append(b)
		# 		nitem.append(a)
		# return np.array(user).reshape(-1,1), np.array(pitem).reshape(-1,1), np.array(nitem).reshape(-1,1)

	def val_data(self):
		test = np.genfromtxt(filepath+'ua.test').astype(np.int32)
		# matrix = sp.lil_matrix((self.n_user, self.n_item))
		# for record in test:
		# 	matrix[record[0]-1, record[1]-1] = int(record[2]>=4)
		matrix = np.zeros((self.n_user, self.n_item))
		for record in test:
			matrix[record[0]-1,record[1]-1] = int(record[2]>=4.0)
		users = []
		items = []
		ratings = []
		for i in range(self.n_user):
			users.extend([i]*self.n_item)
			items.extend(range(self.n_item))
			ratings.extend(list(matrix[i]))
		return np.array(users).reshape(-1,1), np.array(items).reshape(-1,1), ratings
		# return test[:,0].reshape(-1,1)-1, test[:,1].reshape(-1,1)-1, test[:,2].reshape(-1,1)
	def reset_pointer(self):
		self.pointer = 0

def validation(model, data_loader, sess, batch_size):
	val_user, val_item, val_score = data_loader.val_data()
	total_batch = int(len(val_user)/batch_size)
	val_score = val_score[:total_batch*batch_size]
	y_preds = []
	for i in range(total_batch):
		begin = i*batch_size
		end = (i+1)*batch_size
		temp_val_user = val_user[begin:end]
		temp_val_item = val_item[begin:end]
		temp_val_pfeats = data_loader.genre_mat[temp_val_item]
		temp_val_pfeats = temp_val_pfeats.reshape(batch_size, data_loader.genre_feat_size)

		temp_val_score = val_score[begin:end]
		pred_score = sess.run(model.pscore, feed_dict = {model.user: temp_val_user,
														model.pitem:temp_val_item,
														model.pgenre: temp_val_pfeats})
		# temp_val_score =  map(int,temp_val_score>=3)
		# print pred_score.dtype, temp_val_score.dtype
		# for i in range(len(temp_val_score)):
		# 	print pred_score[i], temp_val_score[i]
		# pred_score
		y_preds.extend(list(pred_score.reshape(batch_size)))
		# res.append(roc_auc_score(temp_val_score, pred_score))
	# pred_score = sess.run(model.pscore, feed_dict = {model.user:val_user, model.pitem:val_item})
	# val_auc_score = roc_auc_score(val_score, pred_score)
	print ("\tvalidation roc_auc_score:{}".format(roc_auc_score(val_score, y_preds)))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_user", type=int, default = 943, help="number of users")
	parser.add_argument("--n_item", type=int, default = 1682, help = 'number of items')
	parser.add_argument("--batch_size", type=int, default = 64, help = 'batch size for training')
	parser.add_argument("--nb_epochs", type = int, default = 100, help = 'number of epochs')
	parser.add_argument("--nb_batches", type=int, default = 100, help='number of batches')
	parser.add_argument('--embedding_size', type=int, default=64, help = 'embedding size of latent factors')
	parser.add_argument("--genre_feat_size", type=int, default = 19, help = 'size of movie genre features')
	args = parser.parse_args()
	print args.n_user

	data_loader = DataLoader(args.batch_size)
	args.n_user = data_loader.n_user
	args.n_item = data_loader.n_item
	args.genre_feat_size = data_loader.genre_feat_size
	# args.batch_size = data_loader.n_user
	model = Model(args)

	saver = tf.train.Saver(tf.all_variables())
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		ckpt = tf.train.get_checkpoint_state("./checkpoint/")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (" [!] Load parameters success!!!")
		for e in range(args.nb_epochs):
			data_loader.reset_pointer()
			validation(model, data_loader, sess, args.batch_size)
			total_batch = int(data_loader.train_size/args.batch_size)
			for b in range(total_batch):
				user,pitem, pfeats, nitem, nfeats = data_loader.next_batch()
				loss,_ = sess.run([model.loss, model.train_op], feed_dict = {model.user:user,
																			model.pitem:pitem, model.pgenre:pfeats,
																			model.nitem:nitem, model.ngenre:nfeats})


				sys.stdout.write("\r {}/{}, {}/{}. loss:{}".format(e,args.nb_epochs,b,total_batch,loss))
				sys.stdout.flush()

				if (e*args.nb_batches+b)%1000 == 0 or (e == args.nb_epochs-1 and b == args.nb_batches-1):
					saver.save(sess, './checkpoint/'+"model.ckpt", global_step = e*args.nb_batches+b)

	# model = Model(args)

if __name__ == "__main__":
	main()