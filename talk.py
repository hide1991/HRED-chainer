import MeCab
import net
import chainer
import numpy as np
import os

path = "dialogs"    # directry of dialogs
vocab = {}
id2wd = {}
dialogs = []
c=0
print("loading dialog data ...")
### dialogファイルを再帰的にオープンして、データセットを作成する
for (root, dirs, files) in os.walk(path):
    for file in files:
        if os.path.splitext(file)[1] == u'.txt':
            with open(os.path.join(root, file),encoding='utf-8') as txt:
                sentences = []   # init sentence
                for line in txt:
                    line_s = line.replace(' \n', '')
                    sentences.append(line_s.split())
                    for word in line.split():
                        if word not in vocab:
                            ind = len(vocab)
                            vocab[word] = ind
                            id2wd[ind] = word
                dialogs.append(sentences)
# end of data load
ind = len(vocab)
vocab['eos'] = ind
eos = ind
id2wd[ind] = 'eos'

# unknown word
ind = len(vocab)
vocab['unk'] = ind
id2wd[ind] = 'unk'

# padding word
vocab['pad'] = -1
id2wd[-1] = 'pad'

n_vocab=len(vocab)
print(n_vocab)
print("finish")

model = net.Hred(n_vocab,400,1,eos)
chainer.serializers.load_npz(r'result/gen_iter_33000.npz', model)
print("load model ok")
#
mecab = MeCab.Tagger("-Owakati")
print("Mecab setup ok")
xp=np
while(1):
    text = input("入力->")
    data=mecab.parse(text)
    #print(data)
    print("出力->", end='')
    model.test(data, vocab, id2wd)