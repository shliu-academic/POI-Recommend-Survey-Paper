# Spatio Temporal Network For Next POI Recommendation
paper：基于深度学习时空网络的poi推荐。同时考虑时间因素和空间因素的深度学习时空网络模型的的论文进行调研。主要调研近三年：2020年到2022年，以下是粗略调研一共76篇论文。
google中检索论文标题、摘要、关键字中带有时空和poi的论文


Table of Contents
=================

  <!-- * [Task](#Task) -->
  * [Survey](#Survey)
  * [Dataset](#Dataset)
  * [Models](#Models_)
    * [RNN Based Models](#RNN_Based_Models)
    * [LSTM Models](#LSTM_Models)
    * [GRU Models](#GRU_Models)
    * [Graph Embedding Models](#Graph_Embedding_Models)
    * [GAN Models](#GAN_Models)
    * [Encoder-Decoder Models](#Encoder-Decoder_Models)
    * [Hybrid Model](#Hybrid_Model)
  * [Capture The Time Factor Component](#Consider_Only_Time)
  * [Capture The Geographic Factor Component](#Consider_Only_Geography)
  * [Others](#Others_)

<!-- 在这个括号内的论文，是只考虑时间或地理因素的论文  -->
## Survey
1. Md. Ashraful Islam, Mir Mahathir Mohammad, Sarkar Snigdha Sarathi Das, Mohammed Eunus Ali. **A survey on deep learning based Point-of-Interest (POI) recommendations**. Neurocomputing 2022. [[Paper]](https://doi.org/10.1016/j.neucom.2021.05.114)



## Dataset
1. Foursquare Dataset. [[Download]](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
2. Gowalla Dataset. [[Download]](http://snap.stanford.edu/data/loc-Gowalla.html)
3. Brightkite Dataset. [[Download]](http://snap.stanford.edu/data/loc-Brightkite.html)
4. Weeplaces Dataset. [[Download]](https://www.yongliu.org/datasets.html)
5. Yelp Dataset. [[Download]](https://www.yelp.com/dataset/)


## Models_


### RNN_Based_Models
1. Dingqi Yang, Benjamin Fankhauser, Paolo Rosso, Philippe Cudre-Mauroux. **Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States!**. IJCAI 2020. [[paper]](https://www.ijcai.org/proceedings/2020/302)
2. Kangzhi Zhao, Yong Zhang, Hongzhi Yin, Jin Wang, Kai Zheng, Xiaofang Zhou, Chunxiao Xing. **Discovering Subsequence Patterns for Next POI Recommendation**. IJCAI 2020. [[paper]](https://doi.org/10.24963/ijcai.2020/445)
3. Zhenhua Huang, Xiaolong Lin, Hai Liu, Bo Zhang, Yunwen Chen, Yong Tang. **Deep Representation Learning for Location-Based Recommendation**. IEEE Transactions on Computational Social Systems 2020. [[paper]](https://ieeexplore.ieee.org/document/9024121)




### LSTM_Models
#### Basic_LSTM
1. Yuxia Wu, Ke Li, Guoshuai Zhao, Xueming Qian. **Personalized Long- and Short-term Preference Learning for Next POI Recommendation**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9117156)
2. Ke Sun, Tieyun Qian, Tong Chen, Yile Liang, Quoc Viet Hung Nguyen, Hongzhi Yin. **Where to go next: modeling long-and short-term user preferences for point-of-interest recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5353)
3. Fuqiang Yu, Lizhen Cui, Wei Guo, Xudong Lu, Qingzhong Li, Hua Lu. **A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data**. Proceedings of The Web Conference 2020. [[paper]](https://doi.org/10.1145/3366423.3380202)
4. Qing Guo, Zhu Sun, Jie Zhang, Yin-Leng Theng. **An attentional recurrent neural network for personalized next location recommendation**. Proceedings of the AAAI
Conference on Artificial Intelligence 2020. [[paper]](https://doi.org/10.1609/aaai.v34i01.5337)
5. Lu Zhang, Zhu Sun, Jie Zhang, Yu Lei, Chen Li, Ziqing Wu, Horst Kloeden and Felix Klanner. **An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins**. IJCAI 2020. [[paper]](http://184pc128.csie.ntnu.edu.tw/presentation/21-04-12/An%20Interactive%20Multi-Task%20Learning%20Framework%20for%20Next%20POI%20Recommendation%20with%20Uncertain%20Check-ins.pdf)
6. Honglian Wang, Peiyan Li, Yang Liu, Junming Shao. **Towards real-time demand-aware sequential POI recommendation**. Information Sciences 2021. [[paper]](https://doi.org/10.1016/j.ins.2020.08.088) DSPR
7. Yihao Zhang; Pengxiang Lan; Yuhao Wang; Haoran Xiang. **Spatio-Temporal Mogrifier LSTM and Attention Network for Next POI Recommendation**.ICWS 2022. [[paper]](https://ieeexplore.ieee.org/abstract/document/9885757)
 


#### Bi_LSTM
1. Tongcun Liu, Jianxin Liao, Zhigen Wu, Yulong Wang, Jingyu Wang. **Exploiting geographical-temporal awareness attention for next point-of-interest recommendation**. Neurocomputing 2020. [[paper]](https://doi.org/10.1016/j.neucom.2019.12.122)
2. Chi Harold Liu, Yu Wang, Chengzhe Piao, Zipeng Dai, Ye Yuan, Guoren Wang, Dapeng Wu. **Time-aware location prediction by convolutional area-of-interest modeling and memory-augmented attentive lstm**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9128016)


#### Modified_LSTM
1.  P.Zhao, A.Luo, Y.Liu, F.Zhuang, J.Xu, Z.Li, V.S.Sheng, X.Zhou. **Where to go next: A spatio-temporal gated network for next poi recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ieeexplore.ieee.org/document/9133505)
2.  Meihui Shi， Derong Shen，Yue Kou，Tiezheng Nie，Ge Yu. **Attentional Memory Network with Correlation-based Embedding for time-aware POI recommendation**. Knowledge-Based Systems 2021. [[paper]](https://doi.org/10.1016/j.knosys.2021.106747)




#### Self_Attention
1. Defu Lian, Yongji Wu, Yong Ge, Xing Xie, Enhong Chen. **Geography-Aware Sequential Location Recommendation**. KDD 2020. [[paper]](https://doi.org/10.1145/3394486.3403252)
2. Qianyu Guo, Jianzhong Qi. **Sanst: a self-attentive network for next point-of-interest recommendation**. arXiv preprint arXiv:2001.10379 (2020). [[paper]](https://arxiv.org/abs/2001.10379)



### GRU_Models
1. Yuwen Liu,Aixiang Pei,Fan Wang,Yihong Yang,Xuyun Zhang,Hao Wang,Hongning Dai,Lianyong Qi,Rui Ma. **An attention-based category-aware GRU model for the next POI recommendation**. International Journal of INTELLIGENT SYSTEMS 2021. [[paper]]( https://doi.org/10.1002/int.22412)
2. Jinfeng Fang， Xiangfu Meng. **URPI-GRU: An approach of next POI recommendation based on user relationship and preference information**. Knowledge-Based Systems 2022. [[paper]](https://jeit.ac.cn/en/article/doi/10.11999/JEIT200368?viewType=HTML)
3. Xixi Li, Ruimin Hu, Zheng Wang. **Next-point-of-interest recommendation based on joint mining of regularity and randomness**. Knowledge-Based Systems 2022. [[paper]](https://doi.org/10.1016/j.knosys.2022.109848)


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


### Graph_Embedding_Models
1. Xi Xiong, Fei Xiong, Jun Zhao, Shaojie Qiao, Yuanyuan Lie, Ying Zhao. **Dynamic discovery of favorite locations in spatio-temporal social networks**. Information Processing and Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2020.102337) 
2. Nicholas Lim. **STP-UDGAT: Spatial-Temporal-Preference User Dimensional Graph Attention Network for Next POI Recommendation**. CIKM 2020. [[Paper]](https://doi.org/10.1145/3340531.3411876)
3. Zixuan Yuan. **Spatio-Temporal Dual Graph Attention Network for Query-POI Matching**. SIGIR 2020. [[Paper]](https://doi.org/10.1145/3397271.3401159) 
4. Haoyu Han. **STGCN: A Spatial-Temporal Aware Graph Learning Method for POI Recommendation**. ICDM 2020. [[Paper]](https://ieeexplore.ieee.org/document/9338281) 
5. Yaqiong Qiao, Xiangyang Luo, Chenliang Li, Hechan Tian, Jiangtao Ma. **Heterogeneous graph-based joint representation learning for users and POIs in location-based social network**. Information Processing & Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2019.102151) 
6. Meng Chen, Yan Zhao, Yang Liu, Xiaohui Yu, Kai Zheng. **Modeling spatial trajectories with attribute representation learning**. IEEE Transactions on Knowledge and Data Engineering 2020. [[Paper]](https://ieeexplore.ieee.org/document/9112685) 
7. Xiaojiao Hu, Jiajie Xu, Weiqing Wang, Zhixu Li,An Liu. **A graph embedding based model for fine-grained POI recommendation**. Neurocomputing 2021. [[Paper]](https://doi.org/10.1016/j.neucom.2020.01.118)
8. Xin Wang; Xiao Liu; Li Li; Xiao Chen; Jin Liu; Hao Wu . **Time-aware User Modeling with Check-in Time Prediction for Next POI Recommendation**. ICWS 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9590322) 
9. Junbeom Kim; Sihyun Jeong; Goeon Park; Kihoon Cha; Ilhyun Suh; Byungkook Oh. **DynaPosGNN: Dynamic-Positional GNN for Next POI Recommendation**. ICDMW 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9680032) 
10. Zixuan Yuan. **Incremental Spatio-Temporal Graph Learning for Online Query-POI Matching**. WWW 2021. [[Paper]](https://doi.org/10.1145/3442381.3449810)
11. Wei Chen,....... **Building and exploiting spatial–temporal knowledge graph for next POI recommendation**. Knowledge-Based Systems 2022. [[Paper]](https://doi.org/10.1016/j.knosys.2022.109951) 
12. Zhaobo Wang, Yanmin Zhu, Haobing Liu, Chunyang Wang. **Learning Graph-based Disentangled Representations for Next POI Recommendation**. SIGIR 2022. [[Paper]](https://doi.org/10.1145/3477495.3532012) 
13. Quan Li; Xinhua Xu; Xinghong Liu; Qi Chen. **An Attention-Based Spatiotemporal GGNN for Next POI Recommendation**. IEEE Access 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9727181) 
14. Xiaolin Wang, Guohao Sun, Xiu Fang, Jian Yang and Shoujin Wang. **Modeling Spatio-temporal Neighbourhood for Personalized Point-of-interest Recommendation**. IJCAI 2022. [[Paper]](https://www.ijcai.org/proceedings/2022/0490.pdf) 
15. Xuan Rao...... **Graph-Flashback Network for Next Location Recommendation**. KDD 2022. [[Paper]](https://doi.org/10.1145/3534678.3539383) 
16. Yuwen Liu; Huiping Wu; Khosro Rezaee; ...... **Interaction-Enhanced and Time-Aware Graph Convolutional Network for Successive Point-of-Interest Recommendation in Traveling Enterprises**.  IEEE Transactions on Industrial Informatics 2022. [[Paper]](https://ieeexplore.ieee.org/document/9863644) 



### GAN_Models
1. Qiang Gao, Fengli Zhang, Fuming Yao, Ailing Li, Lin Mei, Fan Zhou. **Adversarial Mobility Learning for Human Trajectory Classification**. IEEE Access 2020. [[Paper]](https://ieeexplore.ieee.org/document/8967063)



### Encoder-Decoder_Models
1. Song Yang, Jiamou Liu, Kaiqi Zhao. **GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation**. SIGIR 2022. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3477495.3531983)
2. Xinfeng Wang, Fumiyo Fukumoto, Jiyi Li, Dongjin Yu & Xiaoxiao Sun . **STaTRL: Spatial-temporal and text representation learning for POI recommendation**. Applied Intelligence 2022. [[Paper]](https://link.springer.com/article/10.1007/s10489-022-03858-w)
3. En Wang; Yiheng Jiang; Yuanbo Xu; Liang Wang; Yongjian Yang. **Spatial-Temporal Interval Aware Sequential POI Recommendation**.ICDE 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9835452)
4. Jiayi Xie, Zhenzhong Chen. **Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation**.arXiv 2022. [[Paper]](https://doi.org/10.48550/arXiv.2209.01559)



### Hybrid_Model
1. Yanan Zhang, Guanfeng Liu, An Liu, Yifan Zhang, Zhixu Li, Xiangliang Zhang, Qing Li. **Personalized Geographical Influence Modeling for POI Recommendation**.  IEEE Intelligent Systems 2020. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9102414)
2. Meng Chen, Yixuan Zuo, Xiaoyi Jia, Yang Liu, Xiaohui Yu, Kai Zheng. **Cem: a convolutional embedding model for predicting next locations**. IEEE Transactions on Intelligent Transportation Systems 2020. [[Paper]](https://ieeexplore.ieee.org/document/9064808)
3. Shaojie Dai, Yanwei Yu, Hao Fan & Junyu Dong . **Personalized POI Recommendation: Spatio-Temporal Representation Learning with Social Tie**. DASFAA 2021. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_37)
4. Yanyan Zhao; Jingyi Liu; Daren Zha; Kai Liu . **Hierarchical and Multi-Resolution Preference Modeling for Next POI Recommendation**. IJCNN 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9533980)
5. Jiyong Zhang， Xin Liu， Xiaofei Zhou， Xiaowen Chu. **Leveraging graph neural networks for point-of-interest recommendations**. Neurocomputing 2021. [[Paper]](https://doi.org/10.1016/j.neucom.2021.07.063)
6. Jianxin Liao, Tongcun Liu, Hongzhi Yin, Tong Chen, Jingyu Wang & Yulong Wang . **An integrated model based on deep multimodal and rank learning for point-of-interest recommendation**.World Wide Web (2021). [[paper]](https://link.springer.com/article/10.1007/s11280-021-00865-8)
7. Qiang Cui, Chenrui Zhang, Yafeng Zhang, Jinpeng Wang, Mingchen Cai. **ST-PIL: Spatial-Temporal Periodic Interest Learning for Next Point-of-Interest Recommendation**. CIKM 2021. [[Paper]](https://doi.org/10.1145/3459637.3482189)
8. Yingtao Luo, Qiang Liu, Zhaocheng Liu. **STAN: Spatio-Temporal Attention Network for Next Location Recommendation**.WWW 2022. [[paper]](https://dl.acm.org/doi/abs/10.1145/3442381.3449998)
9. Zheng Huang, Jing Ma, Yushun Dong, Natasha Zhang Foutz, Jundong Li. **Empowering Next POI Recommendation with Multi-Relational Modeling**.arXiv 2022. [[paper]](https://doi.org/10.48550/arXiv.2204.12288)
10. Gehua Ma, Jingyuan Zhao, Huajin Tang. **Successive POI Recommendation via Brain-inspired Spatiotemporal Aware Representation**.ICLR 2022. [[paper]](https://openreview.net/forum?id=9W2KnHqm_xN)
11. Yuhe Zhou, Guangfei Yang, BingYan, Yuanfeng Cai, Zhiguo Zhu. **Point-of-interest recommendation model considering strength of user relationship for location-based social networks**.Expert Systems with Applications 2022. [[paper]](https://doi.org/10.1016/j.eswa.2022.117147)



### Consider_Only_Geography
1. Shanshan Feng, Lucas Vinh Tran, Gao Cong, Lisi Chen, Jing Li, Fan Li. **HME: A Hyperbolic Metric Embedding Approach for Next-POI Recommendation**.SIGIR 2022. [[paper]](https://doi.org/10.1145/3397271.3401049)
2. Buru Chang. **Learning Graph-Based Geographical Latent Representation for Point-of-Interest Recommendation**.CIKM 2022. [[paper]](https://doi.org/10.1145/3340531.3411905)


### Consider_Only_Time

### Others_
1. Peng Han. **Contextualized Point-of-Interest Recommendation**.IJCAI 2020. [[paper]](https://www.ijcai.org/proceedings/2020/0344.pdf)


2020:
1. https://people.eng.unimelb.edu.au/jianzhongq/papers/IJCNN2020_TimeSAN.pdf
2. http://www.joca.cn/EN/10.11772/j.issn.1001-9081.2019071289


2021:
1. Huimin Sun. **MFNP: A Meta-optimized Model for Few-shot Next POI Recommendation**.IJCAI 2021. [[paper]](https://www.ijcai.org/proceedings/2021/415)
2. Miao Fan. **Meta-Learned Spatial-Temporal POI Auto-Completion for the Search Engine at Baidu Maps**.KDD 2021. [[paper]](https://doi.org/10.1145/3447548.3467058)
3. Yang Li. **Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation**. IJCAI 2021. [[paper]](https://www.ijcai.org/proceedings/2021/206)
4. Xiao Liu. **Geo-BERT Pre-training Model for Query Rewriting in POI Search**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.findings-emnlp.190/)
5. Zeyu Li. **You Are What and Where You Are: Graph Enhanced Attention Network for Explainable POI Recommendation**.CIKM 2021. [[paper]](https://doi.org/10.1145/3459637.3481962)
6. Mingwei Zhang. **SNPR: A Serendipity-Oriented Next POI Recommendation Model**. CIKM 2021. [[paper]](https://doi.org/10.1145/3459637.3482394)
7. https://www.semanticscholar.org/paper/Location-Prediction-via-Bi-direction-Speculation-Li-Hu/398e15e41cbe123432ed7281fa30be792e25d0f3

2022
1. https://dl.acm.org/doi/pdf/10.1145/3477495.3531905
3. https://dl.acm.org/doi/10.1145/3477495.3531905
4. https://dl.acm.org/doi/10.1145/3477495.3531801
5. https://dl.acm.org/doi/10.1145/3523227.3551481
6. https://www.ijcai.org/proceedings/2022/521
8. https://dl.acm.org/doi/10.1145/3511808.3557642
11. https://dl.acm.org/doi/10.1145/3437963.3441797

updated in 2022.11.14
