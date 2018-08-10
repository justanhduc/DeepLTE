import numpy as np
import time
from scipy import misc
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
from DeepLTE import DeepLTE


def crop_center(image, resize=256, crop=(224, 224)):
    h, w = image.shape[:2]
    scale = resize * 1.0 / min(h, w)
    if h < w:
        newh, neww = resize, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), resize
    image = misc.imresize(image, (newh, neww))

    orig_shape = image.shape
    h0 = int((orig_shape[0] - crop[0]) * 0.5)
    w0 = int((orig_shape[1] - crop[1]) * 0.5)
    image = image[h0:h0 + crop[0], w0:w0 + crop[1]]
    return image.astype('uint8')


def preprocess_video(folder, num_frames, resolution, from_frame=0, to_frame=None, stride=None):
    stride = num_frames - 1 if stride is None else stride
    img_list = os.listdir(folder)
    file_type = os.path.splitext(img_list[0])[1]
    img_list = [os.path.splitext(e)[0] for e in img_list]
    img_list.sort(key=int)
    img_list = [e + file_type for e in img_list]
    to_frame = len(img_list) if to_frame is None or to_frame > len(img_list) else to_frame
    print('Processing %s' % folder)
    video = []
    for idx in range(from_frame-1, to_frame, stride):
        if idx + num_frames > to_frame-1:
            break
        image_list = img_list[idx:idx+num_frames]
        if len(image_list) < num_frames:
            continue
        image_list = [misc.imread(folder + '/' + f) for f in image_list]
        video.append(np.array([crop_center(image, resolution[0], resolution) for image in image_list]))
    video = np.stack(video)
    return video


class VideoManager(nn.utils.DataManager):
    def __init__(self, config_file, placeholders):
        super(VideoManager, self).__init__(config_file, placeholders)
        self.num_frames = self.config['model']['num_frames']
        self.input_shape = self.config['model']['input_shape']
        self.load_data()

    def load_data(self):
        video = preprocess_video(self.path, self.num_frames, self.input_shape[:2])
        self.dataset = np.float32(self.normalize(np.transpose(video, (0, 1, 4, 2, 3))))
        self.data_size = self.dataset.shape[0]

    def normalize(self, data):
        return 2. * (data / 255. - .5)

    def unnormalize(self, data):
        data = (data / 2. + .5) * 255.99
        return np.uint8(data)


def interpolate(config_file, *args):
    net = DeepLTE(config_file)
    mon = nn.monitor.Monitor(config_file)
    mon.dump_model(net)

    X__ = T.tensor5('input', theano.config.floatX)
    X = X__[0]
    X_nodes = X[net.nodes]
    X_targets = X[net.targets]
    X_recon = T.concatenate((X_nodes, X_targets))
    X_ = theano.shared(np.zeros((1, net.num_frames,) + net.input_tensor_shape[1:], 'float32'), 'input_placeholder')
    dm = VideoManager(config_file, X_)

    # training costs
    net.set_training_status(True)
    lr_ = theano.shared(net.learning_rate, 'learning rate')
    updates, cost, psnr = net.learn(X_nodes, X_recon, lambda1=1., lambda2=1e-3, lambda3=1e-4, learning_rate=lr_)
    optimize = nn.compile([], [cost, psnr], updates=updates, givens={X__: X_}, name='optimize deepLTE')

    # testing cost
    net.set_training_status(False)
    W = T.scalar('pos', 'float32')
    new_frame = net(X_nodes, W)
    test = nn.compile([W], new_frame, givens={X__: X_}, name='test deepLTE')

    # training scheme
    last_seq_cost = -np.inf
    for seq, _ in enumerate(dm.get_batches()):
        print('Interpolating sequence %d' % (seq + 1))
        iteration = 0
        cost = 1e10
        start_time = time.time()
        while iteration < net.n_epochs:
            iteration += 1

            _recon_cost, _psnr = optimize()
            if np.isnan(_recon_cost) or np.isinf(_recon_cost):
                raise ValueError('Training failed.')
            cost = _recon_cost

            if iteration % net.validation_frequency == 0:
                mon.plot('reconstruction train cost %d' % (seq+1), _recon_cost)
                mon.plot('train psnr %d' % (seq+1), _psnr)
                mon.plot('time elapsed %d' % (seq+1), (time.time() - start_time) / 60.)
                mon.flush()
            mon.tick()
            if cost < last_seq_cost:
                break

        for idx, pos in enumerate(np.array(net.interps, dtype='float32')):
            img = test(pos)
            mon.save_image('%d %d' % (seq+1, idx), img, dm.unnormalize)
        mon.flush()
        if seq == 0:
            last_seq_cost = cost


if __name__ == '__main__':
    interpolate(args.config_file)
