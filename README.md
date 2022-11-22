# Naive_Bayesian_Chinese_Spam_Filter
## 朴素贝叶斯垃圾邮件分类器
项目使用python3编写
数据集使用trec06c，下载连接:<https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo06>  
文件结构:  
├─NB.py  
├─stopWord.txt  
└─trec06c  
程序会自动生成每个词出现概率的对数，分别为HamWords.txt和SpamWords.txt  
模型使用前80%数据作为训练集，后20%作为测试集  
准确率:96.93593314763231%  
精确率:97.78657968313141%  
召回率:93.30813694975545%  
