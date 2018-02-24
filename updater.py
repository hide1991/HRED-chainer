# Custom updater for truncated BackProp Through Time (BPTT)
from chainer import training

import chainer

class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, eos, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len
        self.eos = eos

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        ### training loop
        loss = 0    # init loss
        # Get the next batch
        batch = train_iter.__next__()

        # Concatenate the word IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        x = self.converter(batch, self.device).transpose(1,2,0)
        #print(x)
        xp = chainer.cuda.get_array_module(x.data)
        for i in range(14):
            loss += optimizer.target(chainer.Variable(x[i]), chainer.Variable(x[i+1]), self.eos)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
        chainer.report({'loss': loss}, optimizer.target)
        optimizer.target.reset_all_state()


