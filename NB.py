# 数据预处理程序，将Ham和Spam邮件分类
import re
import jieba
import numpy as np
from collections import Counter
# tqdm库用来画进度条
from tqdm import tqdm
# logging库用来组织jieba库打印日志
import logging

jieba.setLogLevel(logging.INFO)

# 读取停止词
stopword = open('./stopWord.txt', encoding='utf-8').read().split('\n')
# 包含词语的文件数
HamWords = Counter()
SpamWords = Counter()
# 词语数
HamWordsNum = Counter()
SpamWordsNum = Counter()


def countwords(label, path):
    wordtimes = Counter()
    with open(path, 'rb') as f:
        content = f.read()
        content = content.decode('gbk', 'ignore')
    # 用utf-8格式读取邮件
    text = content.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba库进行分词
    text = jieba.cut(text, cut_all=False)
    for word in text:
        if len(word) > 1 and word not in stopword:
            wordtimes[word] += 1
            if label == 'ham':
                HamWordsNum[word] += 1
            if label == 'spam':
                SpamWordsNum[word] += 1
    # 将分词组成新的数组
    for word in wordtimes:
        if wordtimes[word] >= 1:
            if label == 'ham':
                HamWords[word] += 1
            if label == 'spam':
                SpamWords[word] += 1
    f.close()


def reademails():
    HamNum = 0
    SpamNum = 0
    index_path = './trec06c/full/index'
    with open(index_path) as f:
        lines = f.readlines()
    # 读取前80%的数据作为训练集
    lines = lines[:int(0.8 * len(lines))]
    with tqdm(total=len(lines)) as pbar:
        pbar.set_description("Computing term frequency")
        for line in lines:
            label = line.split(' ')[0]
            path = './trec06c' + line.split(' ')[1].replace('\n', '')[2:]
            if label == 'ham':
                HamNum += 1
            if label == 'spam':
                SpamNum += 1
            countwords(label, path)
            pbar.update()
    return HamNum, SpamNum


def train(HamNum, SpamNum):
    PHam = HamNum / (HamNum + SpamNum)
    PSpam = 1 - PHam
    # 生成每个词汇的概率字典
    with tqdm(total=len(HamWords) + len(SpamWords)) as pbar:
        pbar.set_description("Training")
        for word in HamWords:
            HamWords[word] = np.log(HamWords[word]) - np.log(HamNum)
            pbar.update()
        for word in SpamWords:
            SpamWords[word] = np.log(SpamWords[word]) - np.log(SpamNum)
            pbar.update()
    return PHam, PSpam


def test(path, PHam, PSpam):
    # 判断为Ham和Spam的概率
    PH = 1
    PS = 1
    with open(path, 'rb') as f:
        content = f.read()
        content = content.decode('gbk', 'ignore')
    # 用utf-8格式读取邮件
    text = content.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba库进行分词
    text = jieba.lcut(text, cut_all=False)
    PH = np.log(PH)
    PS = np.log(PS)
    for word in text:
        if word not in stopword:
            if word in HamWords and word in SpamWords:
                PH = PH + HamWords[word]
                PS = PS + SpamWords[word]

    PH = PH + np.log(PHam)
    PS = PS + np.log(PSpam)
    # 如果判断为Ham则返回True
    if PH > PS:
        return 'ham'
    else:
        return 'spam'


# 检测模型准确性
def detection(PHam, PSpam):
    # True Positive
    TP = 0
    # True Negative
    TN = 0
    # False Positive
    FP = 0
    # False Negative
    FN = 0
    index_path = './trec06c/full/index'
    with open(index_path) as f:
        lines = f.readlines()
    # 读取后20%的数据作为测试集
    lines = lines[int(0.8 * len(lines)):]
    with tqdm(total=len(lines)) as pbar:
        pbar.set_description("Testing")
        for line in lines:
            label = line.split(' ')[0]
            path = './trec06c' + line.split(' ')[1].replace('\n', '')[2:]
            result = test(path, PHam, PSpam)
            if result == 'ham' and label == 'ham':
                TP += 1
            if result == 'spam' and label == 'spam':
                TN += 1
            if result == 'ham' and label == 'spam':
                FP += 1
            if result == 'spam' and label == 'ham':
                FN += 1
            pbar.update()
    # 准确率
    accuracy = (TP + TN) / len(lines) * 100
    # 查准率
    precision = TP / (TP + FP) * 100
    # 召回率
    recall = TP / (TP + FN) * 100
    print('Test finish')
    print('Accuracy:{}%'.format(accuracy))
    print('Precision:{}%'.format(precision))
    print('Recall:{}%'.format(recall))


if __name__ == '__main__':
    HamNum, SpamNum = reademails()
    PHam, PSpam = train(HamNum, SpamNum)
    detection(PHam, PSpam)
    with open('./HamWords.txt', 'w') as f1:
        f1.write(str(HamWords))
    with open('./SpamWords.txt', 'w') as f2:
        f2.write(str(SpamWords))
