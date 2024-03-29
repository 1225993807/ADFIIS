ADFIIS
===
唐宇皓，袁钟*，彭德中.面向不完备混合数据的模糊多粒度异常检测


摘要
===
针对现有的异常检测方法大多无法有效处理不完备混合数据的问题，提出一种面向不完备混合数据的模糊多粒
度异常检测算法，所提算法能够处理混合属性数据，分别考虑在标称属性和在数值属性上出现缺失值的情况。首先，对属性
之间的模糊相似度进行定义；其次，计算每个属性的模糊熵，基于熵的大小使用多粒度的思想构建多个属性序列；再次，计
算每一个样本的一个异常值并与阈值进行比较，如果异常值大于阈值则认为此样本异常；最后，基于所提方法提出一种面向
不完备混合数据的模糊多粒度异常检测算法(ADFIIS)。在公开数据集上进行实验，并与 MFGAD(Multi-Fuzzy Granules Anomaly
Detection)等主流离群点检测算法进行对比。实验结果表明，ADFIIS 算法在不完备混合数据集上的受试者操作曲线(ROC)效果
更好，并在 12 个实验数据集中的 6 个数据集上拥有更大的曲线下面积(AUC)。平均来看，其 AUC 值优于 90%的对比方法。所
提算法使用模型扩展法在不改变原始数据集的情况下对不完备数据集进行异常检测，拓展了异常检测的适用范围。

使用
===
直接运行AFDIIS.py文件即可得到demo数据的结果
demo数据如下：
```
    trandata =  'c',   4.0000,  0.7000,
                'a',   "*",     0.4000,
                'c',   1.0000,  0.6000,
                '*',   2.0000,  0.3000,
                'a',   8.0000,  0.5000,
                'b',   10.0000, "*"
```


结果如下：
```
out_scores=
            0.77784424
            0.68225438
            0.76327036
            0.75369334
            0.81681734
            0.76119826
            
```
相关数据集位于https://github.com/BElloney/Outlier-detection
