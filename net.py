import chainer

import chainer.functions as F
import chainer.links as L
from chainer import Variable
import numpy as np
import cupy as cp


class Encoder(chainer.Chain):
    ### bi-directional lstm encoder
    def __init__(self, n_vocab, n_hidden):
        super(Encoder, self).__init__(
            emb = L.EmbedID(n_vocab, n_hidden, ignore_label=-1),
            lstm_forward = L.LSTM(None, n_hidden),
            lstm_backward = L.LSTM(None, n_hidden),
        )

    def reset_state(self):
        self.lstm_forward.reset_state()
        self.lstm_backward.reset_state()

    def __call__(self, x):
        ### this function is called in training
        # wordid = np.array([0], dtype = np.int32) # word id
        for word in x:
            h_forward = self.emb(word)
            h_forward = self.lstm_forward(h_forward)
        for word in reversed(x):
            h_backward = self.emb(word)
            h_backward = self.lstm_backward(h_backward)
        h = F.concat((h_forward, h_backward))
        return h

    def encode(self, x, vocab):
        ### this function is called in predict
        ### unknown word is 'unk'
        #print(x)
        #print(x.shape)
        for word in x:
            try:
                h_forward = self.emb(word)
                h_forward = self.lstm_forward(h_forward)
            except:
                h_forward = self.emb(Variable(xp.array([vocab['unk']])))
                h_forward = self.lstm_forward(h_forward)
        for word in reversed(x):
            try:
                h_backward = self.emb(word)
                h_backward = self.lstm_backward(h_backward)
            except:
                h_forward = self.emb(Variable(xp.array([vocab['unk']])))
                h_forward = self.lstm_backward(h_forward)
        h = F.concat((h_forward, h_backward))
        return h


class Decoder(chainer.Chain):
    ### lstm decoder
    def __init__(self, n_vocab, n_hidden, batch, eos):
        self.n_hidden = n_hidden
        super(Decoder, self).__init__(
            emb0 = L.EmbedID(n_vocab, n_hidden, ignore_label=-1),
            lstm1 = L.StatelessLSTM(None, n_hidden),
            l2 = L.Linear(n_hidden, n_vocab)
        )
        #self.c_init = np.zeros((batch, self.n_hidden), dtype = np.float32) # hidden cell initialize         
        #self.w_eos = Variable(np.full(batch, eos, dtype = np.int32))

    def __call__(self, x, h_input, eos):
        ### this function is called in training
        #print(x.shape) 
        
        xp = chainer.cuda.get_array_module(h_input.data)
        c_init = xp.zeros((x.shape[1], self.n_hidden), dtype = xp.float32) # hidden cell initialize         
        w_eos = Variable(xp.full(x.shape[1], eos, dtype = xp.int32))
        count = 0   # word counter
        loss = 0  # loss initialize
        h_old = h_input # hidden variable initialize
        w = self.emb0(w_eos)
        c_new, h_new = self.lstm1(c_init, h_old, w)
        ww = self.l2(h_new)           # predict word
        c_old = c_new           # update cell
        h_old = h_new
                  # update hidden
        for word in x:
            #print(word)
            loss += F.softmax_cross_entropy(ww, word)
            w = self.emb0(word)
            c_new, h_new = self.lstm1(c_old, h_old, w)
            ww = self.l2(h_new)
            c_old = c_new           # update cell
            h_old = h_new           # update hidden
            count += 1
        #wordid = Variable(xp.full(x.shape[1],eos, dtype = xp.int32)) # word id
        #loss += F.softmax_cross_entropy(ww, w_eos)
        return loss/count
    
    def predict(self, h_input, vocab, id2wd):
        xp = chainer.cuda.get_array_module(h_input.data)
        h_old = h_input
        #print(h_input.shape)
        c_old = xp.zeros((1, self.n_hidden), dtype = xp.float32) # hidden cell initialize
        wordid = Variable(xp.array([vocab['eos']], dtype = xp.int32)) # word id
        w = self.emb0(wordid)
        talk = []
        print("output: ",end='')
        for i in range(50):
            try:
                c_new, h_new = self.lstm1(c_old, h_old, w)
                #print(c_old)
                ww = self.l2(h_new)
                c_old = c_new           # update cell
                h_old = h_new           # update hidden
                wordid = Variable(xp.array([F.argmax(F.softmax(ww)).data], dtype = xp.int32)) # word id
                w = self.emb0(wordid)
                #print(wordid)
                talk.append(wordid.data[0])
                print(id2wd[wordid.data[0]], " ", end='')
            except:
                pass
                print("error")
            if id2wd[wordid.data[0]] == 'eos':
                break
        print("")
        return talk

class Context(chainer.Chain):
    def __init__(self, n_hidden):
        super(Context, self).__init__(
            lstm1 = L.LSTM(None, n_hidden),
        )

    def reset_state(self):
        self.lstm1.reset_state()              

    def __call__(self, h):
        h = self.lstm1(h)
        return h

class Hred(chainer.Chain):
    def __init__(self,n_vocab,n_hidden, batch, eos):
        super(Hred,self).__init__(
            enc = Encoder(n_vocab, n_hidden),
            dec = Decoder(n_vocab, n_hidden, batch, eos),
            context = Context(n_hidden),
        )

    def __call__(self, x, y, eos):
        loss = 0  # loss initialize
        ### encode ###
        h = self.enc(x)       # h size is 2 * n_hidden
        self.enc.reset_state()

        ### context ###
        h = self.context(h)

        ### decode ###
        loss += self.dec(y, h, eos)
        
        return loss

    def reset_all_state(self):
        self.enc.reset_state()
        self.context.reset_state()

    def test(self, data, vocab, id2wd):
        xp = chainer.cuda.get_array_module(data)
        #print("input: ", data)
        x = []
        for word in data.split(" "):
            #print(word)
            try:
                x.append(vocab[word])
            except:
                x.append(vocab['unk'])
        x.pop(len(x)-1)
        x.append(vocab['eos'])
        print(x)
        x = xp.array([x], dtype=xp.int32)
        h = self.enc.encode(chainer.Variable(x.T), vocab)
        self.enc.reset_state
        h = self.context(h)
        x = self.dec.predict(h, vocab, id2wd)
        print(x)
        # context update
        x = xp.array([x], dtype=xp.int32)
        h = self.enc.encode(chainer.Variable(x.T), vocab)
        self.enc.reset_state
        self.context(h)
        