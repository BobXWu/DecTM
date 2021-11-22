import numpy as np
import tensorflow as tf


class DecTM(object):
    def __init__(self, config, act_func=tf.nn.softplus):
        self.config = config
        self.act_func = act_func

        self.x = tf.placeholder(tf.float32, [None, config.vocab_size])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.h_dim = float(config.num_topic)
        self.a = 1*np.ones((1 , int(self.h_dim))).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)
        self.var2 = tf.constant( ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +
                                ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T )

        self.beta = tf.get_variable(shape=(self.config.vocab_size, self.config.num_topic), initializer=tf.contrib.layers.xavier_initializer(), name='beta')

        self._create_network()
        self._create_loss_optimizer()

        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        self.z_mean, self.z_log_sigma_sq = self.encode()
        eps = tf.random_normal((self.config.batch_size, self.config.num_topic), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.sigma = tf.exp(self.z_log_sigma_sq)
        self.theta = tf.nn.softmax(self.z)
        self.recon_x = self.decode(self.theta)

    def encode(self,):
        e = tf.layers.dense(self.x, units=self.config.en1_units, activation=self.act_func)
        e = tf.layers.dense(e, units=self.config.en2_units, activation=self.act_func)
        e = tf.nn.dropout(e, self.keep_prob)

        z_mean = tf.contrib.layers.batch_norm(tf.layers.dense(e, units=self.config.num_topic))
        z_log_sigma_sq = tf.contrib.layers.batch_norm(tf.layers.dense(e, units=self.config.num_topic))

        return z_mean, z_log_sigma_sq

    def decode(self, theta):
        recon_x = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.matmul(self.l2_norm(theta), tf.transpose(self.l2_norm(self.beta)))))
        return recon_x

    def l2_norm(self, x):
        return tf.nn.l2_normalize(x, dim=1)

    def _create_loss_optimizer(self):
        recon_loss = -tf.reduce_sum(self.x * tf.log(self.recon_x + 1e-10), 1)
        latent_loss = 0.5 * ( tf.reduce_sum(tf.div(self.sigma, self.var2), 1) + \
        tf.reduce_sum( tf.multiply(tf.div((self.mu2 - self.z_mean), self.var2),
                  (self.mu2 - self.z_mean)), 1) - self.h_dim + tf.reduce_sum(tf.log(self.var2), 1) - tf.reduce_sum(self.z_log_sigma_sq, 1) )

        self.loss = recon_loss + latent_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.99).minimize(self.loss)
