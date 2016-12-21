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

		u_embed = Embedding(input_dim = args.n_user, output_dim = args.embedding_size, input_length = 1)
		i_embed = Embedding(input_dim = args.n_item, output_dim = args.embedding_size, input_length = 1)

		user_latent = u_embed(self.user)
		pitem_latent = i_embed(self.pitem)
		nitem_latent = i_embed(self.nitem)

		self.pscore = tf.reduce_sum(tf.mul(user_latent, pitem_latent),-1)
		self.nscore = tf.reduce_sum(tf.mul(user_latent, nitem_latent),-1)

		self.loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.pscore - self.nscore)))
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

		print self.pscore.get_shape()
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
		# self.matrix = np.zeros((self.n_user, self.n_item))
		# for record in data:
		# 	self.matrix[record[0]-1,record[1]-1] = record[2]
		# print self.n_user, self.n_item

	def next_batch(self):
		begin = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size
		self.pointer+=1
		return self.matcoo.row[begin:end].reshape(-1,1),\
				self.matcoo.col[begin:end].reshape(-1,1),\
				np.random.randint(0,self.n_item, (self.batch_size, 1))
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
		temp_val_score = val_score[begin:end]
		pred_score = sess.run(model.pscore, feed_dict = {model.user: temp_val_user, model.pitem:temp_val_item})
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
	parser.add_argument("--nb_epochs", type = int, default = 20, help = 'number of epochs')
	parser.add_argument("--nb_batches", type=int, default = 100, help='number of batches')
	parser.add_argument('--embedding_size', type=int, default=64, help = 'embedding size of latent factors')
	args = parser.parse_args()
	print args.n_user

	data_loader = DataLoader(args.batch_size)
	args.n_user = data_loader.n_user
	args.n_item = data_loader.n_item
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
				user,pitem,nitem = data_loader.next_batch()
				loss,_ = sess.run([model.loss, model.train_op], feed_dict = {model.user:user, model.pitem:pitem, model.nitem:nitem})


				sys.stdout.write("\r {}/{}, {}/{}. loss:{}".format(e,100,b,total_batch,loss))
				sys.stdout.flush()

				if (e*args.nb_batches+b)%1000 == 0 or (e == args.nb_epochs-1 and b == args.nb_batches-1):
					saver.save(sess, './checkpoint/'+"model.ckpt", global_step = e*args.nb_batches+b)
			# val_user, val_item, val_score = data_loader.val_data()
			# pred_score = sess.run(model.pscore, feed_dict = {model.user:val_user, model.pitem:val_item})
			# val_auc_score = roc_auc_score(val_score, pred_score)
			# print ("validation roc_auc_score:{}".format(val_auc_score))

	# model = Model(args)

if __name__ == "__main__":
	main()