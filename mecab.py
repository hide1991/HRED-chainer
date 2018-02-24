# -*- coding: utf-8 -*-
#テキストファイルから分かち書きのテキストファイルを作るプログラム
import MeCab

mc = MeCab.Tagger("-Owakati")
count = 0
with open('origin.txt',"r") as text:
    f= open(f'dialogs/dialog{count}.txt',"w",encoding='utf-8')
    for line in text:
        try:
            line_text=mc.parse(line)
            if line_text != '\n':
                f.writelines(line_text)
            else:
                f.close
                count += 1
                f= open(f'dialogs/dialog{count}.txt',"w",encoding='utf-8')
                print('this line is null')
        except:
            print("変換に失敗したテキストがあります")
            print(line_text)
    f.close()