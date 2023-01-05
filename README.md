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
    * [Transformer Models](#Transformer_Models)
    * [Graph Learning Based Models](#Graph_Learning_Based_Models)
    * [GAN Models](#GAN_Models)
    * [Mixture Models](#Mixture_Models)

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

2020
1. **Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States!**. IJCAI 2020. [[paper]](https://www.ijcai.org/proceedings/2020/302)
2. **Discovering Subsequence Patterns for Next POI Recommendation**. IJCAI 2020. [[paper]](https://doi.org/10.24963/ijcai.2020/445)
3. **Deep Representation Learning for Location-Based Recommendation**. IEEE Transactions on Computational Social Systems 2020. [[paper]](https://ieeexplore.ieee.org/document/9024121)
4. **Personalized Long- and Short-term Preference Learning for Next POI Recommendation**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9117156)
5. **Where to go next: modeling long-and short-term user preferences for point-of-interest recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5353)
6. **A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data**. Proceedings of The Web Conference 2020. [[paper]](https://doi.org/10.1145/3366423.3380202)
7. **An attentional recurrent neural network for personalized next location recommendation**. Proceedings of the AAAI
Conference on Artificial Intelligence 2020. [[paper]](https://doi.org/10.1609/aaai.v34i01.5337)
8. **An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins**. IJCAI 2020. [[paper]](http://184pc128.csie.ntnu.edu.tw/presentation/21-04-12/An%20Interactive%20Multi-Task%20Learning%20Framework%20for%20Next%20POI%20Recommendation%20with%20Uncertain%20Check-ins.pdf)
9. **Exploiting geographical-temporal awareness attention for next point-of-interest recommendation**. Neurocomputing 2020. [[paper]](https://doi.org/10.1016/j.neucom.2019.12.122)
10. **Time-aware location prediction by convolutional area-of-interest modeling and memory-augmented attentive lstm**. IEEE Transactions on Knowledge and Data Engineering 2020. [[paper]](https://ieeexplore.ieee.org/document/9128016)
11. **Where to go next: A spatio-temporal gated network for next poi recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence 2020. [[paper]](https://ieeexplore.ieee.org/document/9133505)
12. **TimeSAN: A Time-Modulated Self-Attentive Network for Next Point-of-Interest Recommendation**. IJCNN 2020. [[paper]](https://people.eng.unimelb.edu.au/jianzhongq/papers/IJCNN2020_TimeSAN.pdf)
13. **Geography-Aware Sequential Location Recommendation**. KDD 2020. [[paper]](https://doi.org/10.1145/3394486.3403252)
14. **Sanst: a self-attentive network for next point-of-interest recommendation**. arXiv preprint arXiv:2001.10379 (2020). [[paper]](https://arxiv.org/abs/2001.10379)


2021
1. **Towards real-time demand-aware sequential POI recommendation**. Information Sciences 2021. [[paper]](https://doi.org/10.1016/j.ins.2020.08.088) DSPR
2. **Spatio-Temporal Mogrifier LSTM and Attention Network for Next POI Recommendation**.ICWS 2022. [[paper]](https://ieeexplore.ieee.org/abstract/document/9885757)
3. **Attentional Memory Network with Correlation-based Embedding for time-aware POI recommendation**. Knowledge-Based Systems 2021. [[paper]](https://doi.org/10.1016/j.knosys.2021.106747)
4. **An attention-based category-aware GRU model for the next POI recommendation**. International Journal of INTELLIGENT SYSTEMS 2021. [[paper]]( https://doi.org/10.1002/int.22412)
5. **MFNP: A Meta-optimized Model for Few-shot Next POI Recommendation**.IJCAI 2021. [[paper]](https://www.ijcai.org/proceedings/2021/415)


2022
1. **URPI-GRU: An approach of next POI recommendation based on user relationship and preference information**. Knowledge-Based Systems 2022. [[paper]](https://jeit.ac.cn/en/article/doi/10.11999/JEIT200368?viewType=HTML)
2. **Next-point-of-interest recommendation based on joint mining of regularity and randomness**. Knowledge-Based Systems 2022. [[paper]](https://doi.org/10.1016/j.knosys.2022.109848)





### Transformer_Models

2021
1. **SNPR: A Serendipity-Oriented Next POI Recommendation Model**. CIKM 2021. [[paper]](https://doi.org/10.1145/3459637.3482394)
2. **Origin-Aware Next Destination Recommendation with Personalized Preference Attention**[[Paper]](https://dl.acm.org/doi/10.1145/3437963.3441797)

2022
2. **GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation**. SIGIR 2022. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3477495.3531983)
3. **STaTRL: Spatial-temporal and text representation learning for POI recommendation**. Applied Intelligence 2022. [[Paper]](https://link.springer.com/article/10.1007/s10489-022-03858-w)
4. **Spatial-Temporal Interval Aware Sequential POI Recommendation**.ICDE 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9835452)
5. **Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation**.arXiv 2022. [[Paper]](https://doi.org/10.48550/arXiv.2209.01559)
6. **Next Point-of-Interest Recommendation with Auto-Correlation Enhanced Multi-Modal Transformer Network**. [[Paper]](https://dl.acm.org/doi/10.1145/3477495.3531905)
7. **Next Point-of-Interest Recommendation with Inferring Multi-step Future Preferences**.[[Paper]](https://www.ijcai.org/proceedings/2022/521)





-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


### Graph_Learning_Based_Models

2020
1. **Dynamic discovery of favorite locations in spatio-temporal social networks**. Information Processing and Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2020.102337) 
2. **STP-UDGAT: Spatial-Temporal-Preference User Dimensional Graph Attention Network for Next POI Recommendation**. CIKM 2020. [[Paper]](https://doi.org/10.1145/3340531.3411876)
3. **STGCN: A Spatial-Temporal Aware Graph Learning Method for POI Recommendation**. ICDM 2020. [[Paper]](https://ieeexplore.ieee.org/document/9338281) 
4. **Heterogeneous graph-based joint representation learning for users and POIs in location-based social network**. Information Processing & Management 2020. [[Paper]](https://doi.org/10.1016/j.ipm.2019.102151) 
5. **Modeling spatial trajectories with attribute representation learning**. IEEE Transactions on Knowledge and Data Engineering 2020. [[Paper]](https://ieeexplore.ieee.org/document/9112685) 

2021
1. **A graph embedding based model for fine-grained POI recommendation**. Neurocomputing 2021. [[Paper]](https://doi.org/10.1016/j.neucom.2020.01.118)
2. **Time-aware User Modeling with Check-in Time Prediction for Next POI Recommendation**. ICWS 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9590322) 
3. **DynaPosGNN: Dynamic-Positional GNN for Next POI Recommendation**. ICDMW 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9680032) 
4. **Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation**. IJCAI 2021. [[paper]](https://www.ijcai.org/proceedings/2021/206)
5. **You Are What and Where You Are: Graph Enhanced Attention Network for Explainable POI Recommendation**.CIKM 2021. [[paper]](https://doi.org/10.1145/3459637.3481962)

2022
1. **Incremental Spatio-Temporal Graph Learning for Online Query-POI Matching**. WWW 2021. [[Paper]](https://doi.org/10.1145/3442381.3449810)
2. **Building and exploiting spatial–temporal knowledge graph for next POI recommendation**. Knowledge-Based Systems 2022. [[Paper]](https://doi.org/10.1016/j.knosys.2022.109951) 
3. **Learning Graph-based Disentangled Representations for Next POI Recommendation**. SIGIR 2022. [[Paper]](https://doi.org/10.1145/3477495.3532012) 
4. **An Attention-Based Spatiotemporal GGNN for Next POI Recommendation**. IEEE Access 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9727181) 
5. **Modeling Spatio-temporal Neighbourhood for Personalized Point-of-interest Recommendation**. IJCAI 2022. [[Paper]](https://www.ijcai.org/proceedings/2022/0490.pdf) 
6. **Graph-Flashback Network for Next Location Recommendation**. KDD 2022. [[Paper]](https://doi.org/10.1145/3534678.3539383) 
7. **Interaction-Enhanced and Time-Aware Graph Convolutional Network for Successive Point-of-Interest Recommendation in Traveling Enterprises**.  IEEE Transactions on Industrial Informatics 2022. [[Paper]](https://ieeexplore.ieee.org/document/9863644) 
8. **Learning Graph-Based Geographical Latent Representation for Point-of-Interest Recommendation**.CIKM 2022. [[paper]](https://doi.org/10.1145/3340531.3411905)


### GAN_Models
1. **Adversarial Mobility Learning for Human Trajectory Classification**. IEEE Access 2020. [[Paper]](https://ieeexplore.ieee.org/document/8967063)




### Mixture_Models

2020
1. **Personalized Geographical Influence Modeling for POI Recommendation**.  IEEE Intelligent Systems 2020. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9102414)
2. **Cem: a convolutional embedding model for predicting next locations**. IEEE Transactions on Intelligent Transportation Systems 2020. [[Paper]](https://ieeexplore.ieee.org/document/9064808)

2021
1. **Personalized POI Recommendation: Spatio-Temporal Representation Learning with Social Tie**. DASFAA 2021. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_37)
2. **Hierarchical and Multi-Resolution Preference Modeling for Next POI Recommendation**. IJCNN 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9533980)
3. **Leveraging graph neural networks for point-of-interest recommendations**. Neurocomputing 2021. [[Paper]](https://doi.org/10.1016/j.neucom.2021.07.063)
4. **An integrated model based on deep multimodal and rank learning for point-of-interest recommendation**.World Wide Web (2021). [[paper]](https://link.springer.com/article/10.1007/s11280-021-00865-8)
5. **ST-PIL: Spatial-Temporal Periodic Interest Learning for Next Point-of-Interest Recommendation**. CIKM 2021. [[Paper]](https://doi.org/10.1145/3459637.3482189)

2022
1. **STAN: Spatio-Temporal Attention Network for Next Location Recommendation**.WWW 2022. [[paper]](https://dl.acm.org/doi/abs/10.1145/3442381.3449998)
2. **Empowering Next POI Recommendation with Multi-Relational Modeling**.arXiv 2022. [[paper]](https://doi.org/10.48550/arXiv.2204.12288)
3. **Successive POI Recommendation via Brain-inspired Spatiotemporal Aware Representation**.ICLR 2022. [[paper]](https://openreview.net/forum?id=9W2KnHqm_xN)
4. **Point-of-interest recommendation model considering strength of user relationship for location-based social networks**.Expert Systems with Applications 2022. [[paper]](https://doi.org/10.1016/j.eswa.2022.117147)




updated in 2022.11.14
