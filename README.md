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
  * [Encoder-Decoder Models](#Encoder-Decoder_Models)
  * [Multi-module combination model](#Multi-module_Combination_Model)
  * [Other Models](#Other_Models)

<!-- 在这个括号内的论文，是只考虑时间或地理因素的论文  -->
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
3. Zhenhua Huang, Xiaolong Lin, Hai Liu, Bo Zhang, Yunwen Chen, Yong Tang. **Deep Representation Learning for Location-Based Recommendation**. IEEE Transactions on Computational Social Systems 2020. [[paper]](https://ieeexplore.ieee.org/document/9024121)



## LSTM_Models
### Basic_LSTM
1. Yuxia Wu, Ke Li, Guoshuai Zhao, Xueming Qian. **Personalized Long- and Short-term Preference Learning for Next POI Recommendation**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9117156)
2. Ke Sun, Tieyun Qian, Tong Chen, Yile Liang, Quoc Viet Hung Nguyen, Hongzhi Yin. **Where to go next: modeling long-and short-term user preferences for point-of-interest recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5353)
3. Fuqiang Yu, Lizhen Cui, Wei Guo, Xudong Lu, Qingzhong Li, Hua Lu. **A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data**. Proceedings of The Web Conference 2020. [[paper]](https://doi.org/10.1145/3366423.3380202)
4. Qing Guo, Zhu Sun, Jie Zhang, Yin-Leng Theng. **An attentional recurrent neural network for personalized next location recommendation**. Proceedings of the AAAI
Conference on Artificial Intelligence 2020. [[paper]](https://doi.org/10.1609/aaai.v34i01.5337)
5. Hao Wang, Huawei Shen, Xueqi Cheng. **Modeling POI-Specific Spatial-Temporal Context for Point-of-Interest Recommendation**. PAKDD 2020. [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_11)
6. Honglian Wang, Peiyan Li, Yang Liu, Junming Shao. **Towards real-time demand-aware sequential POI recommendation**. Information Sciences 2021. [[paper]](https://doi.org/10.1016/j.ins.2020.08.088)
7. Kun Wang, Xiaofeng Wang & Xuan Lu . **POI recommendation method using LSTM-attention in LBSN considering privacy protection**. Complex & Intelligent Systems (2021). [[paper]](https://link.springer.com/article/10.1007/s40747-021-00440-8)
 


### Bi_LSTM
1. Tongcun Liu, Jianxin Liao, Zhigen Wu, Yulong Wang, Jingyu Wang. **Exploiting geographical-temporal awareness attention for next point-of-interest recommendation**. Neurocomputing 2020. [[paper]](https://doi.org/10.1016/j.neucom.2019.12.122)
2. Chi Harold Liu, Yu Wang, Chengzhe Piao, Zipeng Dai, Ye Yuan, Guoren Wang, Dapeng Wu. **Time-aware location prediction by convolutional area-of-interest modeling and memory-augmented attentive lstm**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9128016)
3. 


### Modified_LSTM
1.  P.Zhao, A.Luo, Y.Liu, F.Zhuang, J.Xu, Z.Li, V.S.Sheng, X.Zhou. **Where to go next: A spatio-temporal gated network for next poi recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ieeexplore.ieee.org/document/9133505)
2.  Meihui Shi， Derong Shen，Yue Kou，Tiezheng Nie，Ge Yu. **Attentional Memory Network with Correlation-based Embedding for time-aware POI recommendation**. Knowledge-Based Systems 2021. [[paper]](https://doi.org/10.1016/j.knosys.2021.106747)



### Self_Attention
1. Defu Lian, Yongji Wu, Yong Ge, Xing Xie, Enhong Chen. **Geography-Aware Sequential Location Recommendation**. KDD 2020. [[paper]](https://doi.org/10.1145/3394486.3403252)
2. Qianyu Guo, Jianzhong Qi. **Sanst: a self-attentive network for next point-of-interest recommendation**. arXiv preprint arXiv:2001.10379 (2020). [[paper]](https://arxiv.org/abs/2001.10379)



## GRU_Models
1. Yuwen Liu,Aixiang Pei,Fan Wang,Yihong Yang,Xuyun Zhang,Hao Wang,Hongning Dai,Lianyong Qi,Rui Ma. **An attention-based category-aware GRU model for the next POI recommendation**. International Journal of INTELLIGENT SYSTEMS 2021. [[paper]]( https://doi.org/10.1002/int.22412)

## Graph_Embedding_Models
1. Xi Xiong, Fei Xiong, Jun Zhao, Shaojie Qiao, Yuanyuan Lie, Ying Zhao. **Dynamic discovery of favorite locations in spatio-temporal social networks**. Information Processing and Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2020.102337) 
2. Yaqiong Qiao, Xiangyang Luo, Chenliang Li, Hechan Tian, Jiangtao Ma. **Heterogeneous graph-based joint representation learning for users and POIs in location-based social network**. Information Processing & Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2019.102151) 
3. Meng Chen, Yan Zhao, Yang Liu, Xiaohui Yu, Kai Zheng. **Modeling spatial trajectories with attribute representation learning**. IEEE Transactions on Knowledge and Data Engineering 2020. [[Paper]](https://ieeexplore.ieee.org/document/9112685) 
4. Miao Li; Wenguang Zheng; Yingyuan Xiao; Ke Zhu; Wei Huang. **Exploring Temporal and Spatial Features for Next POI Recommendation in LBSNs**.  IEEE Access 2021. [[Paper]](https://ieeexplore.ieee.org/document/9360823) 
5. Xiaojiao Hu, Jiajie Xu, Weiqing Wang, Zhixu Li,An Liu. **A graph embedding based model for fine-grained POI recommendation**. Neurocomputing 2021. [[Paper]](https://doi.org/10.1016/j.neucom.2020.01.118) 
6. Yu Wang, An Liu, Junhua Fang, Jianfeng Qu, Lei Zhao. **ADQ-GNN: Next POI Recommendation by Fusing GNN and Area Division with Quadtree**. WISE 2021. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-91560-5_13) 
7. Xin Wang; Xiao Liu; Li Li; Xiao Chen; Jin Liu; Hao Wu . **Time-aware User Modeling with Check-in Time Prediction for Next POI Recommendation**. ICWS 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9590322) 
8. Sadaf Safavi and Mehrdad Jalali . **RecPOID: POI Recommendation with Friendship Aware and Deep CNN**.  Social Networks Analysis and Mining 2021. [[Paper]](https://doi.org/10.3390/fi13030079) 


## GAN_Models
1. Qiang Gao, Fengli Zhang, Fuming Yao, Ailing Li, Lin Mei, Fan Zhou. **Adversarial Mobility Learning for Human Trajectory Classification**. IEEE Access 2020. [[Paper]](https://ieeexplore.ieee.org/document/8967063)



## Encoder-Decoder_Models
1. Lu Zhang, Zhu Sun, Jie Zhang, Yu Lei, Chen Li, Ziqing Wu, Horst Kloeden and Felix Klanner. **An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins**. (IJCAI-20). [[Paper]](http://184pc128.csie.ntnu.edu.tw/presentation/21-04-12/An%20Interactive%20Multi-Task%20Learning%20Framework%20for%20Next%20POI%20Recommendation%20with%20Uncertain%20Check-ins.pdf)
2. Sajal Halder, Kwan Hui Lim, Jeffrey Chan & Xiuzhen Zhang . **Transformer-Based Multi-task Learning for Queuing Time Aware Next POI Recommendation**. PAKDD 2021. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-75765-6_41)
3. Yepeng Li, Xuefeng Xian, Pengpeng Zhao, Yanchi Liu & Victor S. Sheng . **MGSAN: A Multi-granularity Self-attention Network for Next POI Recommendation**. WISE 2021. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-91560-5_14)


## Multi-module_Combination_Model
1. Yue Cui, Hao Sun, Yan Zhao, Hongzhi Yin, Kai Zheng. **Sequential-Knowledge-Aware Next POI Recommendation: A Meta-Learning Approach**. ACM Transactions on Information Systems 2021. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3460198)
2. Hongyu Zang, Dongcheng Han, Xin Li, Zhifeng Wan, Mingzhong Wang. **CHA: Categorical Hierarchy-based Attention for Next POI Recommendation**. ACM Transactions on Information Systems 2021. [[Paper]](https://doi.org/10.1145/3464300)
3. Ling Chen, Yuankai Ying, Dandan Lyu, Shanshan Yu & Gencai Chen . **A multi-task embedding based personalized POI recommendation method**. CCF Transactions on Pervasive Computing and Interaction 2021. [[Paper]](https://link.springer.com/article/10.1007/s42486-021-00069-z)
4. Shaojie Dai, Yanwei Yu, Hao Fan & Junyu Dong . **Personalized POI Recommendation: Spatio-Temporal Representation Learning with Social Tie**. DASFAA 2021. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_37)
5. Yanyan Zhao; Jingyi Liu; Daren Zha; Kai Liu . **Hierarchical and Multi-Resolution Preference Modeling for Next POI Recommendation**. IJCNN 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9533980)
6. Jiyong Zhang， Xin Liu， Xiaofei Zhou， Xiaowen Chu. **Leveraging graph neural networks for point-of-interest recommendations**. Neurocomputing 2021. [[Paper]](https://doi.org/10.1016/j.neucom.2021.07.063)
7. Giannis Christoforidis, Pavlos Kefalas, Apostolos N. Papadopoulos & Yannis Manolopoulos . **RELINE: point-of-interest recommendations using multiple network embeddings**. Knowledge and Information Systems 2021. [[Paper]](https://link.springer.com/article/10.1007/s10115-020-01541-5)



## Other_Models
1. Hui Luo, Jingbo Zhou, Zhifeng Bao, Shuangli Li, J.Shane Culpepper, Haochao Ying, Hao Liu, Hui Xiong. **Spatial Object Recommendation with Hints: When Spatial Granularity Matters**. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval 2020. [[Paper]](https://dl.acm.org/doi/10.1145/3397271.3401090)
3. Yanan Zhang, Guanfeng Liu, An Liu, Yifan Zhang, Zhixu Li, Xiangliang Zhang, Qing Li. **Personalized Geographical Influence Modeling for POI Recommendation**.  IEEE Intelligent Systems 2020. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9102414)
4. Meng Chen, Yixuan Zuo, Xiaoyi Jia, Yang Liu, Xiaohui Yu, Kai Zheng. **Cem: a convolutional embedding model for predicting next locations**. IEEE Transactions on Intelligent Transportation Systems 2020. [[Paper]](https://ieeexplore.ieee.org/document/9064808)
5. Liang Chang, Wei Chen, Jianbo Huang, Chenzhong Bin & Wenkai Wang  . **Exploiting multi-attention network with contextual influence for point-of-interest recommendation**.Applied Intelligence (2021). [[paper]](https://link.springer.com/article/10.1007/s10489-020-01868-0)




updated in 2022.11.14
