import numpy as np
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Plug-and-play Deep locally temporal embedding.')
parser.add_argument('config_file', type=str, help='A .config file for DeepLTE.')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use.')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import theano
from theano import tensor as T
import neuralnet as nn
from DeepLTE_UCF import DeepLLENet


class DataManager(nn.utils.DataManager):
    def __init__(self, config_file, placeholders):
        super(DataManager, self).__init__(config_file, placeholders)
        self.num_frames = self.config['model']['num_frames']
        self.load_data()

    def load_data(self):
        import pickle as pkl
        dataset = pkl.load(open(self.path, 'rb'))
        dataset = self.normalize(dataset)
        self.dataset = np.transpose(dataset, (0, 1, 4, 2, 3)).astype('float32')[0]
        self.num_train_data = self.dataset.shape[0]

    def normalize(self, data):
        return 2. * (data / 255. - .5)

    def unnormalize(self, data):
        data = (data / 2. + .5) * 255.99
        return np.uint8(data)


def interpolate(config_file, *args):
    net = DeepLLENet(config_file)
    mon = nn.monitor.Monitor(config_file)
    mon.dump_model(net)

    X = T.tensor4('input', theano.config.floatX)
    X_nodes = X[net.nodes]
    X_targets = X[net.targets]
    X_recon = T.concatenate((X_nodes, X_targets))
    X_ = theano.shared(np.zeros((net.num_frames,) + net.input_tensor_shape[1:], 'float32'), 'input_placeholder')
    dm = DataManager(config_file, X_)

    # training costs
    net.set_training_status(True)
    lr_ = theano.shared(net.learning_rate, 'learning rate')
    updates, cost, psnr = net.learn(X_nodes, X_recon, lambda1=1., lambda2=1e-3, lambda3=1e-4, learning_rate=lr_)
    optimize = nn.compile([], [cost, psnr], updates=updates, givens={X: X_}, name='optimize deepLTE')

    # testing cost
    net.set_training_status(False)
    W = T.scalar('pos', 'float32')
    new_frame1 = net(X_nodes, W)
    test = nn.compile([W], new_frame1, givens={X: X_}, name='test deepLTE')

    # training scheme
    last_seq_cost = -np.inf
    print('Interpolating sequence...')
    dm.update_input(dm.dataset)
    iteration = 0
    cost = 1e10
    start_time = time.time()
    while iteration < net.n_epochs:
        iteration += 1

        _recon_cost, _psnr = optimize()
        if np.isnan(_recon_cost) or np.isinf(_recon_cost):
            raise ValueError('Training failed.')

        if iteration % net.validation_frequency == 0:
            mon.plot('reconstruction train cost', _recon_cost)
            mon.plot('train psnr', _psnr)
            mon.plot('time elapsed', (time.time() - start_time) / 60.)
            mon.flush()
        mon.tick()
        if cost < last_seq_cost:
            break

    r = np.concatenate([test(i) for i in np.arange(net.order+1).astype('float32')])
    tf = np.concatenate((test(.5), test(2.5)))
    f = np.concatenate([test(pos) for pos in net.interps])
    mon.save_image('train frame', r, callback=dm.unnormalize)
    mon.save_image('train interp frame', tf, callback=dm.unnormalize)
    for i in sorted(net.nodes+net.targets):
        img = dm.dataset[i]
        img = img[None]
        mon.save_image('%d' % i, img, callback=dm.unnormalize)

    img = dm.dataset[3]
    img = img[None]
    mon.save_image('gt 3', img, callback=dm.unnormalize)
    mon.save_image('3', f, callback=dm.unnormalize)
    mon.flush()


if __name__ == '__main__':
    interpolate(args.config_file)
