import numpy as np
from DecTM import DecTM


class Runner(object):
    def __init__(self, config, init_embeddings=None):
        self.config = config
        self.model = DecTM(self.config)

    def train(self, X):
        feed_dict = dict()

        data_size = X.shape[0]
        batch_size = self.config.batch_size
        total_batch = int(data_size / batch_size)

        for epoch in range(1, self.config.num_epoch + 1):
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            loss = np.zeros((data_size,))
            for i in range(total_batch):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_input = X[idx[start:end]]
                feed_dict[self.model.x] = batch_input
                _, batch_loss = self.model.sess.run((self.model.optimizer, self.model.loss), feed_dict=feed_dict)
                loss[start:end] = batch_loss

            # the incompleted batch
            feed_dict[self.model.x] = X[idx[-batch_size:]]
            _, batch_loss = self.model.sess.run((self.model.optimizer, self.model.loss), feed_dict=feed_dict)
            loss[-batch_size:] = batch_loss

            if epoch % 5 == 0:
                print("Epoch: {:03d} loss={:.3f}".format(epoch, np.mean(loss)))

        beta = self.model.sess.run((self.model.beta)).T

        return beta

    def test(self, X):
        data_size = X.shape[0]
        batch_size = self.config.batch_size

        theta = np.zeros((data_size, self.config.num_topic))
        loss = np.zeros((data_size,))
        var_tuple = (self.model.loss, self.model.theta)
        for i in range(int(data_size / batch_size)):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_input = X[start:end]
            feed_dict = {self.model.x: batch_input}
            batch_loss, batch_theta = self.model.sess.run(var_tuple, feed_dict=feed_dict)
            loss[start:end] = batch_loss
            theta[start:end] = batch_theta

        batch_input = X[-batch_size:]
        feed_dict = {self.model.x: batch_input}
        batch_loss, batch_theta = self.model.sess.run(var_tuple, feed_dict=feed_dict)
        loss[-batch_size:] = batch_loss
        theta[-batch_size:] = batch_theta

        loss = np.mean(loss)
        return theta
