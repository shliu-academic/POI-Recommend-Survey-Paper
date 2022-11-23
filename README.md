# Spatio Temporal Network For Next POI Recommendation
在深度学习邻域的下，对下一个POI进行推荐时，同时考虑时间因素和空间因素的时空网络模型的的论文进行调研。主要调研近三年：2020年到2022年。

Table of Contents
=================

  <!-- * [Task](#Task) -->
  * [Survey](#Survey)
  * [Dataset](#Dataset)
  * [RNN Based Models](#RNN_Based_Models)
  * [LSTM Models](#LSTM_Models)
  * [GRU Models](#GRU_Models)
  * [Graph Embedding Models](#Graph_Embedding_Models)
  * [GAN Models](#GAN_Models)
  * [Other Models](#Other_Models)


## Survey
1. Md. Ashraful Islam, Mir Mahathir Mohammad, Sarkar Snigdha Sarathi Das, Mohammed Eunus Ali. **A survey on deep learning based Point-of-Interest (POI) recommendations**. Neurocomputing 2022. [[Paper]](https://doi.org/10.1016/j.neucom.2021.05.114)



## Dataset
1. Foursquare Dataset. [[Download]](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
2. Gowalla Dataset. [[Download]](http://snap.stanford.edu/data/loc-Gowalla.html)
3. Brightkite Dataset. [[Download]](http://snap.stanford.edu/data/loc-Brightkite.html)
4. Weeplaces Dataset. [[Download]](https://www.yongliu.org/datasets.html)
5. Yelp Dataset. [[Download]](https://www.yelp.com/dataset/)



## RNN_Based_Models
1. Dingqi Yang, Benjamin Fankhauser, Paolo Rosso, Philippe Cudre-Mauroux. **Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States!**. IJCAI 2020. [[paper]](https://www.ijcai.org/proceedings/2020/302)
2. Kangzhi Zhao, Yong Zhang, Hongzhi Yin, Jin Wang, Kai Zheng, Xiaofang Zhou, Chunxiao Xing. **Discovering Subsequence Patterns for Next POI Recommendation**. IJCAI 2020. [[paper]](https://doi.org/10.24963/ijcai.2020/445)
3. Qinyong Wang, Hongzhi Yin, Tong Chen, Zi Huang, Hao Wang, Yanchang Zhao, Nguyen Quoc Viet Hung. **Next Point-of-Interest Recommendation on Resource-Constrained Mobile Devices**. Proceedings of The Web Conference 2020. [[paper]](https://dl.acm.org/doi/10.1145/3366423.3380170)
4. Zhenhua Huang, Xiaolong Lin, Hai Liu, Bo Zhang, Yunwen Chen, Yong Tang. **Deep Representation Learning for Location-Based Recommendation**. IEEE Transactions on Computational Social Systems 2020. [[paper]](https://ieeexplore.ieee.org/document/9024121)



## LSTM_Models
### Basic_LSTM
1. Yuxia Wu, Ke Li, Guoshuai Zhao, Xueming Qian. **Personalized Long- and Short-term Preference Learning for Next POI Recommendation**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9117156)
2. Ke Sun, Tieyun Qian, Tong Chen, Yile Liang, Quoc Viet Hung Nguyen, Hongzhi Yin. **Where to go next: modeling long-and short-term user preferences for point-of-interest recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5353)
3. Lu Zhang, Zhu Sun, Jie Zhang, Yu Lei, Chen Li, Ziqing Wu, Horst Kloeden, Felix Klanner. **An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins**. IJCAI 2020. [[paper]](http://184pc128.csie.ntnu.edu.tw/presentation/21-04-12/An%20Interactive%20Multi-Task%20Learning%20Framework%20for%20Next%20POI%20Recommendation%20with%20Uncertain%20Check-ins.pdf)
4. Fuqiang Yu, Lizhen Cui, Wei Guo, Xudong Lu, Qingzhong Li, Hua Lu. **A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data**. Proceedings of The Web Conference 2020. [[paper]](https://doi.org/10.1145/3366423.3380202)
5. Qing Guo, Zhu Sun, Jie Zhang, Yin-Leng Theng. **An attentional recurrent neural network for personalized next location recommendation**. Proceedings of the AAAI
Conference on Artificial Intelligence 2020. [[paper]](https://doi.org/10.1609/aaai.v34i01.5337)
6. Hao Wang, Huawei Shen, Xueqi Cheng. **Modeling POI-Specific Spatial-Temporal Context for Point-of-Interest Recommendation**. PAKDD 2020. [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_11)
 


### Bi_LSTM
1. Tongcun Liu, Jianxin Liao, Zhigen Wu, Yulong Wang, Jingyu Wang. **Exploiting geographical-temporal awareness attention for next point-of-interest recommendation**. Neurocomputing 2020. [[paper]](https://doi.org/10.1016/j.neucom.2019.12.122)
2. Chi Harold Liu, Yu Wang, Chengzhe Piao, Zipeng Dai, Ye Yuan, Guoren Wang, Dapeng Wu. **Time-aware location prediction by convolutional area-of-interest modeling and memory-augmented attentive lstm**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9128016)
3. Buru Chang, Yookyung Koh, Donghyeon Park, Jaewoo Kang. **Content-Aware Successive Point-of-Interest Recommendation**. SIAM 2020. [[paper]](https://doi.org/10.1137/1.9781611976236.12)------G



### Modified_LSTM
1.  P.Zhao, A.Luo, Y.Liu, F.Zhuang, J.Xu, Z.Li, V.S.Sheng, X.Zhou. **Where to go next: A spatio-temporal gated network for next poi recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ieeexplore.ieee.org/document/9133505)



### Self-attention
1. Defu Lian, Yongji Wu, Yong Ge, Xing Xie, Enhong Chen. **Geography-Aware Sequential Location Recommendation**. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2020. [[paper]](https://www.ijcai.org/proceedings/2022/715)-----------------G
2. Qianyu Guo, Jianzhong Qi. **Sanst: a self-attentive network for next point-of-interest recommendation**. arXiv preprint arXiv:2001.10379 (2020). [[paper]](https://arxiv.org/abs/2001.10379)



## GRU_Models


## Graph_Embedding_Models
1. Xi Xiong, Fei Xiong, Jun Zhao, Shaojie Qiao, Yuanyuan Lie, Ying Zhao. **Dynamic discovery of favorite locations in spatio-temporal social networks**. Information Processing and Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2020.102337) 
2. Lu Zhang, Zhu Sun, Jie Zhang, Horst Kloeden, Felix Klanner. **Modeling hierarchical category transition for next POI recommendation with uncertain check-ins**. Information Sciences 2020. [[Paper]](https://doi.org/10.1016/j.ins.2019.12.006) -----------------------------G
3. Yaqiong Qiao, Xiangyang Luo, Chenliang Li, Hechan Tian, Jiangtao Ma. **Heterogeneous graph-based joint representation learning for users and POIs in location-based social network**. Information Processing & Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2019.102151) 
4. Meng Chen, Yan Zhao, Yang Liu, Xiaohui Yu, Kai Zheng. **Modeling spatial trajectories with attribute representation learning**. IEEE Transactions on Knowledge and Data Engineering 2020. [[Paper]](https://ieeexplore.ieee.org/document/9112685) 


## GAN_Models
1. Qiang Gao, Fengli Zhang, Fuming Yao, Ailing Li, Lin Mei, Fan Zhou. **Adversarial Mobility Learning for Human Trajectory Classification**. IEEE Access 2020. [[Paper]](https://ieeexplore.ieee.org/document/8967063)


## Other_Models
1. Shanshan Feng, Lucas Vinh Tran, Gao Cong, Lisi Chen, Jing Li, Fan Li. **Hme: A hyperbolic metric embedding approach for next-poi recommendation**. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval 2020. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3397271.3401049)----------G
2. Hui Luo, Jingbo Zhou, Zhifeng Bao, Shuangli Li, J.Shane Culpepper, Haochao Ying, Hao Liu, Hui Xiong. **Spatial Object Recommendation with Hints: When Spatial Granularity Matters**. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval 2020. [[Paper]](https://dl.acm.org/doi/10.1145/3397271.3401090)
3. Yanan Zhang, Guanfeng Liu, An Liu, Yifan Zhang, Zhixu Li, Xiangliang Zhang, Qing Li. **Personalized Geographical Influence Modeling for POI Recommendation**.  IEEE Intelligent Systems 2020. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9102414)
4. Meng Chen, Yixuan Zuo, Xiaoyi Jia, Yang Liu, Xiaohui Yu, Kai Zheng. **Cem: a convolutional embedding model for predicting next locations**. IEEE Transactions on Intelligent Transportation Systems 2020. [[Paper]](https://ieeexplore.ieee.org/document/9064808)




updated in 2022.11.14
