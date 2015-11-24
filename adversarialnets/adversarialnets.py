import theano.tensor as T
import theano
import numpy as np
# from theano.tensor.shared_randomstreams import RandomStreams


class Generator:

    def __init__(self, num_vis, num_output,
                 noise_type='gaussian', randomstate=np.random.RandomState()):
        self.noise_type = noise_type
        self.num_vis = num_vis
        self.num_output = num_output
        self.randomstate = randomstate
        self.W = theano.shared(value=np.asarray(
            self.randomstate.uniform(low=-0.001, high=0.001,
                                     size=(num_vis, num_output,)
                                     )), name='WG')

        self.b = theano.shared(
            np.asarray(np.zeros(num_output), dtype=theano.config.floatX),
            name='b')
        if randomstate is None:
            self.randomstate = T.raw_random.RandomStreams()
        self.params = [self.W, self.b]

    def get_noise(self):
        """ Output simply random noise
            Generate m examples by nxk size
        """
        size = (1, self.num_vis)
        if self.noise_type == 'gaussian':
            return theano.shared(name='N', value=np.asarray(
                self.randomstate.uniform(low=-0.001, high=0.001, size=(size)
                                         )))
        if self.noise_type == 'binomial':
            return self.radomstate.binomial(n=self.num_vis, p=0.1)

    def output(self, X):
        return T.nnet.sigmoid(T.dot(X, self.W) + self.b)

    def get_output_samples(self, num_samples):
        pass

    def update(self, cost, lrate=.001):
        grads = T.grad(cost, self.params)
        return [(oldparam, oldparam + lrate * newparam)
                for (oldparam, newparam) in zip(self.params, grads)]


class Discriminator:

    """ Output single scalar """

    def __init__(self, num_vis,rng=np.random.RandomState()):
        self.rng = rng
        self.W = theano.shared(value=np.asarray(
            self.rng.uniform(low=-0.001, high=0.001, size=(num_vis, 1,)
                             )), name='WD')
        self.b = theano.shared(np.zeros((1)))

        self.params = [self.W]

    def output(self, X):
        """ This module implement MLP"""
        # inpres = T.nnet.sigmoid(T.dot(X, self.W))
        # noiseres = T.nnet.sigmoid(T.dot(Z, self.W))
        return T.nnet.sigmoid(T.dot(X, self.W))

    def update(self, cost):
        grads = T.grad(cost, self.params)
        return [(oldparam, oldparam + 0.001 * newparam) for (oldparam, newparam) in zip(self.params, grads)]


class Layer:

    def __init__(self, data, num_vis, num_hid):
        self.x = T.matrix('x')
        self.data = data
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.params = [self.W, self.b]

    def forward(self):
        return T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)


class Model:

    def __init__(self, data, num_vis, num_hid):
        self.x = T.matrix('x')
        self.data = data
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.num_samples = self.data.shape[0]

    def _sample(self, num_examples):
        return self.x[:num_examples]

    def compute(self, minibatch=1, steps=5, lrate=0.01):
        G = Generator(self.num_vis, self.num_hid)
        D = Discriminator(self.num_vis)
        for i in range(steps):
            # Sample m noise examples from Generator
            noise_samples = G.get_noise()
            # Sample m examples from data distribution
            data_examples = self._sample(minibatch)
            # Get real examples
            realX = D.output(data_examples)
            # Get generated examples
            genX = D.output(noise_samples)
            drealcost = T.mean(T.nnet.binary_crossentropy(realX, T.ones(realX.shape)))
            dgencost = T.mean(T.nnet.binary_crossentropy(noise_samples, T.zeros(genX.shape)))
            gencost = T.mean(T.nnet.binary_crossentropy(genX, T.ones(genX.shape)))
            cost = drealcost + dgencost
            updates = D.update(cost.mean())
            func = theano.function([], (realX, genX), updates=updates, givens={self.x: self.data})
            print("Discriminator cost {0}: ".format(func()))
        noise_samples = G.get_noise()
        allparams = []
        for param in G.params:
            allparams.append(param)
        '''for param in D.params:
            allparams.append(param)'''
        #gencost = 1 / self.num_samples * \
        #    T.sum(T.log(1 - D.output(G.output(noise_samples))))
        grads = T.grad(T.mean(gencost), allparams)
        return gencost, [(oldparam, oldparam - lrate * newparam) for (oldparam, newparam) in zip(allparams, grads)]

    def train(self, minibatch=10, steps=5, iters=100):
        cost, updates = self.compute(minibatch=minibatch, steps=steps)
        func = theano.function([], cost, updates=updates)
        samples = []
        for i in range(iters):
            disc_cost = func()
            print("Generator cost cost: {0}".format(disc_cost))
        return samples
