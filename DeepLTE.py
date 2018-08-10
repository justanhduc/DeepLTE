import numpy as np
from theano import tensor as T

import neuralnet as nn
from neuralnet import utils
from neuralnet import Model


class DeepLTE(Model):
    def __init__(self, config_file, **kwargs):
        super(DeepLTE, self).__init__(config_file, **kwargs)
        self.num_frames = self.config['model']['num_frames']
        self.order = self.config['model']['order']
        self.nodes = self.config['model']['nodes']
        self.targets = self.config['model']['targets']
        self.interps = self.config['model']['interps']
        self.alpha = self.config['model']['alpha']
        self.dropout = self.config['model']['dropout']
        self.perceptual_cost = self.config['model']['perceptual_cost']
        self.vgg_weight_file = self.config['model']['vgg_weight_file']
        self.input_tensor_shape = (None,) + self.input_shape[1:]

        enc = nn.model_zoo.resnet34(self.input_tensor_shape, 64, 'lrelu', False, False, name='encoder', alpha=self.alpha)
        self.model.append(enc)

        subnet = 'decoder'
        dec = nn.Sequential(input_shape=enc.output_shape, layer_name='decoder')
        dec.append(nn.ResizingLayer(dec.input_shape, 2, layer_name=subnet + '_up1'))
        dec.append(nn.StackingConv(dec.output_shape, 3, 256, 5, batch_norm=False, layer_name=subnet + '_block5',
                                   He_init='normal', stride=(1, 1), activation='lrelu', alpha=self.alpha))

        dec.append(nn.ResizingLayer(dec.output_shape, 2, layer_name=subnet + '_up2'))
        dec.append(nn.StackingConv(dec.output_shape, 5, 128, 5, batch_norm=False, layer_name=subnet + '_block6',
                                   He_init='normal', stride=(1, 1), activation='lrelu', alpha=self.alpha))

        dec.append(nn.ResizingLayer(dec.output_shape, 2, layer_name=subnet + '_up3'))
        dec.append(nn.StackingConv(dec.output_shape, 6, 128, 5, batch_norm=False, layer_name=subnet + '_block7',
                                   He_init='normal', stride=(1, 1), activation='lrelu', alpha=self.alpha))
        dec.append(nn.ConvolutionalLayer(dec.output_shape, 128, 5, activation='linear', layer_name=subnet+'_conv7'))
        if self.dropout:
            dec.append(nn.DropoutLayer(dec.output_shape, drop_prob=.5, layer_name=subnet + '_dropout7'))
        dec.append(nn.ActivationLayer(dec.output_shape, 'lrelu', subnet+'_act7', alpha=self.alpha))

        dec.append(nn.ResizingLayer(dec.output_shape, 2, layer_name=subnet + '_up4'))
        dec.append(nn.StackingConv(dec.output_shape, 8, 64, 5, batch_norm=False, layer_name=subnet + '_block8',
                                   He_init='normal', stride=(1, 1), activation='lrelu', alpha=self.alpha))
        dec.append(nn.ConvolutionalLayer(dec.output_shape, 64, 5, activation='linear', layer_name=subnet + '_conv8'))
        if self.dropout:
            dec.append(nn.DropoutLayer(dec.output_shape, drop_prob=.5, layer_name=subnet + '_dropout8'))
        dec.append(nn.ActivationLayer(dec.output_shape, 'lrelu', subnet + '_act8', alpha=self.alpha))

        dec.append(nn.ConvolutionalLayer(dec.output_shape, 3, 5, activation='tanh', no_bias=False,
                                         layer_name=subnet + '_output'))

        self.model.append(dec)

    def inference_encoder(self, input, W=False):
        output = self.model['encoder'](input)
        if W:
            if isinstance(W, float):
                W = T.constant(W, dtype='float32')
            b, c, h, w = output.shape
            output = output.flatten(2)
            x = T.tile(np.arange(self.order+1, dtype='float32'), (c*h*w, 1)).T
            u = T.tile(W, c * h * w)
            output = utils.lagrange_interpolation(x, output, u, self.order)
            output = output.reshape((1, c, h, w))
        return output

    def inference_decoder(self, input):
        return self.model['decoder'](input)

    def inference(self, input, W):
        output = self.inference_decoder(self.inference_encoder(input, W))
        return output

    def unnormalize(self, X):
        return X / 2. + .5

    def get_cost(self, input, gt, **kwargs):
        lambda1 = kwargs.get('lambda1', 1)
        lambda2 = kwargs.get('lambda2', 1e-3)
        lambda3 = kwargs.get('lambda2', 1e-4)
        recon = self(input, False)
        interp = T.concatenate([self(input, .5 + i) for i in range(self.order)])
        out = T.concatenate((recon, interp))
        inter_fr1 = self.unnormalize(out) * 255.99

        cost = nn.huberloss(inter_fr1, self.unnormalize(gt) * 255.) * lambda1 + \
               nn.dog_loss(inter_fr1, self.unnormalize(gt) * 255., 3) * lambda2
        if self.perceptual_cost:
            cost += nn.vgg16_loss(inter_fr1, self.unnormalize(gt) * 255., self.vgg_weight_file) * lambda3
        psnr = nn.psnr(self.unnormalize(out), self.unnormalize(gt))
        return cost, psnr

    def learn(self, input, gt, **kwargs):
        cost, psnr = self.get_cost(input, gt, **kwargs)
        updates = self.build_updates(cost, self.trainable, **kwargs)
        return updates, cost, psnr
