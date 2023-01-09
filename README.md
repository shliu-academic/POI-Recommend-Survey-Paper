# Spatio Temporal Network For Next POI Recommendation
paper：基于深度学习时空网络的poi推荐；

调研近三年：2020-2022；

通过IEEE Xplore、ScienceDirect、ACM Digital Library、Springer和AI-Paper-Search进行检索；

检索词包括："time"、"Geographic"、"Spatio"、"Temporal"、"Network"、"POI"、"location"、"Recommendation"；

只筛选涉及到深度学习领域的paper；

以下是通过论文内的摘要进行粗略地分类。

Table of Contents
=================

  <!-- * [Task](#Task) -->
  * [Survey](#Survey)
  * [Dataset](#Dataset)
  * [Models](#Models_)
    * [RNN Based Models](#RNN_Based_Models)
    * [CNN Based Models](#CNN_Based_Models)
    * [Transformer Models](#Transformer_Models)
    * [Graph Learning Based Models](#Graph_Learning_Based_Models)
    * [GAN Models](#GAN_Models)
    * [Mixture Models](#Mixture_Models)
    * [Other Models](#Other_Models)

## Survey
### 2019
* Location prediction in large-scale social networks: an in-depth benchmarking study. 
<br>The VLDB Journal　　　　　　　　　　ACM
<br>[[Paper]](https://doi.org/10.1007/s00778-019-00553-0)

### 2020
* A Survey on Point-of-Interest Recommendation in Location-based Social Networks. 
<br>WebMedia　　　　　　　　　　ACM
<br>[[Paper]](https://doi.org/10.1145/3428658.3430970)

* A comprehensive survey on trajectory-based location prediction
<br>Iran Journal of Computer Science 　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/article/10.1007/s42044-019-00052-z)

* Survey on user location prediction based on geo-social networking data
<br>World Wide Web  　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/article/10.1007/s11280-019-00777-8)


### 2021
* Key Research on Recommendation Algorithms Based on Spatio-temporal Relationships in Location Social Networks.
<br>2021 International Conference on Artificial Intelligence, Big Data and Algorithms (CAIBDA)　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9545948)

* A survey on deep learning based Point-of-Interest (POI) recommendations. 
<br>Neurocomputing　　　　　　　　　　ScienceDirect
<br>[[Paper]](https://doi.org/10.1016/j.neucom.2021.05.114)

* Real-time dynamic network learning for location inference modelling and computing. 
<br>Neurocomputing　　　　　　　　　　ScienceDirect
<br>[[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231221016143)

### 2022
* Individual mobility prediction review: Data, problem, method and application. 
<br>Multimodal Transportation　　　　　　　　　　ScienceDirect
<br>[[Paper]](https://www.sciencedirect.com/science/article/pii/S2772586322000028)

* Point-of-Interest Recommender Systems Based on Location-Based Social Networks: A Survey from an Experimental Perspective
<br>ACM Computing Surveys　　　　　　　　　　ACM
<br>[[Paper]](https://doi.org/10.1145/3510409)

* A survey of location-based social networks: problems, methods, and future research directions
<br>GeoInformatica　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/article/10.1007/s10707-021-00450-1)

* A survey on next location prediction techniques, applications, and challenges
<br>EURASIP Journal on Wireless Communications and Networking　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/article/10.1186/s13638-022-02114-6)


## Dataset
* Foursquare Dataset. [[Download]](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
* Gowalla Dataset. [[Download]](http://snap.stanford.edu/data/loc-Gowalla.html)
* Brightkite Dataset. [[Download]](http://snap.stanford.edu/data/loc-Brightkite.html)
* Weeplaces Dataset. [[Download]](https://www.yongliu.org/datasets.html)
* Yelp Dataset. [[Download]](https://www.yelp.com/dataset/)


## Models_


### RNN_Based_Models

#### 2020
* Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States!
<br>IJCAI. 
<br>[[paper]](https://www.ijcai.org/proceedings/2020/302)

* Discovering Subsequence Patterns for Next POI Recommendation. 
<br>IJCAI. 
<br>[[paper]](https://doi.org/10.24963/ijcai.2020/445)

* Deep Representation Learning for Location-Based Recommendation. 
<br>IEEE Transactions on Computational Social Systems. 
<br>[[paper]](https://ieeexplore.ieee.org/document/9024121)

* Personalized Long- and Short-term Preference Learning for Next POI Recommendation. 
<br>IEEE Transactions on Knowledge and Data Engineering. 
<br>[[paper]](https://ieeexplore.ieee.org/document/9117156)

* Where to go next: modeling long-and short-term user preferences for point-of-interest recommendation. 
<br>Proceedings of the AAAI Conference on Artificial Intelligence. 
<br>[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5353)

* A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data. 
<br>Proceedings of The Web Conference 2020. 
<br>[[paper]](https://doi.org/10.1145/3366423.3380202)

* An attentional recurrent neural network for personalized next location recommendation. 
<br>Proceedings of the AAAI Conference on Artificial Intelligence. 
<br>[[paper]](https://doi.org/10.1609/aaai.v34i01.5337)

* An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins. 
<br>IJCAI. 
<br>[[paper]](http://184pc128.csie.ntnu.edu.tw/presentation/21-04-12/An%20Interactive%20Multi-Task%20Learning%20Framework%20for%20Next%20POI%20Recommendation%20with%20Uncertain%20Check-ins.pdf)

* Exploiting geographical-temporal awareness attention for next point-of-interest recommendation. 
<br>Neurocomputing　　　　　　　　　　　ScienceDirect
<br>[[paper]](https://doi.org/10.1016/j.neucom.2019.12.122)

* Time-aware location prediction by convolutional area-of-interest modeling and memory-augmented attentive lstm. 
<br>IEEE Transactions on Knowledge and Data Engineering 2020. 
<br>[[paper]](https://ieeexplore.ieee.org/document/9128016)

* Where to go next: A spatio-temporal gated network for next poi recommendation. 
<br>Proceedings of the AAAI Conference on Artificial Intelligence 2020. 
<br>[[paper]](https://ieeexplore.ieee.org/document/9133505)

* TimeSAN: A Time-Modulated Self-Attentive Network for Next Point-of-Interest Recommendation. 
<br>IJCNN 2020. 
<br>[[paper]](https://people.eng.unimelb.edu.au/jianzhongq/papers/IJCNN2020_TimeSAN.pdf)

* Geography-Aware Sequential Location Recommendation. 
<br>KDD 2020. 
<br>[[paper]](https://doi.org/10.1145/3394486.3403252)

* Sanst: a self-attentive network for next point-of-interest recommendation. 
<br>arXiv preprint arXiv:2001.10379 (2020). 
<br>[[paper]](https://arxiv.org/abs/2001.10379)

* Where to Go Next: A Spatio-Temporal Gated Network for Next POI Recommendation.
<br>IEEE Transactions on Knowledge and Data Engineering 　　　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9133505)

* DualSIN: Dual Sequential Interaction Network for Human Intentional Mobility Prediction.
<br>SIGSPATIAL  　　　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3397536.3422221)


#### 2021
* Towards real-time demand-aware sequential POI recommendation.
<br>Information Sciences 2021. 
<br>[[paper]](https://doi.org/10.1016/j.ins.2020.08.088)

* Attentional Memory Network with Correlation-based Embedding for time-aware POI recommendation. 
<br>Knowledge-Based Systems　　　　　　　　　　ScienceDirect
<br>[[paper]](https://doi.org/10.1016/j.knosys.2021.106747)

* An attention-based category-aware GRU model for the next POI recommendation**. 
<br>International Journal of INTELLIGENT SYSTEMS 2021. 
<br>[[paper]]( https://doi.org/10.1002/int.22412)

* MFNP: A Meta-optimized Model for Few-shot Next POI Recommendation.
<br>IJCAI 2021. 
<br>[[paper]](https://www.ijcai.org/proceedings/2021/415)

* A Context-Aware POI Recommendation.
<br>TENCON 2021 - 2021 IEEE Region 10 Conference (TENCON)　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9707376)

* PS-LSTM:Popularity Analysis And Social Network For Point-Of-Interest Recommendation In Previously Unvisited Locations.
<br>CNIOT 　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3468691.3468720)



#### 2022
* Spatio-Temporal Mogrifier LSTM and Attention Network for Next POI Recommendation.
<br>IEEE International Conference on Web Services(ICWS)      　　　　 IEEE
<br>[[paper]](https://ieeexplore.ieee.org/abstract/document/9885757)

* URPI-GRU: An approach of next POI recommendation based on user relationship and preference information. 
<br>Knowledge-Based Systems 2022. 
<br>[[paper]](https://jeit.ac.cn/en/article/doi/10.11999/JEIT200368?viewType=HTML)

* Next-point-of-interest recommendation based on joint mining of regularity and randomness. 
<br>Knowledge-Based Systems 2022. 
<br>[[paper]](https://doi.org/10.1016/j.knosys.2022.109848)

* Real-time POI recommendation via modeling long- and short-term user preferences.
<br>Neurocomputing　　　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S092523122101434X)



### CNN_Based_Models
#### 2020
* Personalized tourism route recommendation based on user’s active interests
<br>2020 21st IEEE International Conference on Mobile Data Management (MDM).      　　　　 IEEE 
<br>[[paper]](https://ieeexplore.ieee.org/document/9162322)

### 2021
* Tell Me Where to Go Next: Improving POI Recommendation via Conversation
<br>DASFAA　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-73200-4_14)



### Transformer_Models

#### 2021
* SNPR: A Serendipity-Oriented Next POI Recommendation Model. 
<br>CIKM　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3459637.3482394)

* Origin-Aware Next Destination Recommendation with Personalized Preference Attention
<br>WSDM.
<br>[[Paper]](https://dl.acm.org/doi/10.1145/3437963.3441797)

#### 2022
* GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation. 
<br>SIGIR　　　　　　　　　ACM
<br>[[Paper]](https://dl.acm.org/doi/abs/10.1145/3477495.3531983)

* STaTRL: Spatial-temporal and text representation learning for POI recommendation. 
<br>Applied Intelligence　　　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/article/10.1007/s10489-022-03858-w)

* Spatial-Temporal Interval Aware Sequential POI Recommendation.
<br>ICDE. 
<br>[[Paper]](https://ieeexplore.ieee.org/abstract/document/9835452)

* Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation.
<br>arXiv. 
<br>[[Paper]](https://doi.org/10.48550/arXiv.2209.01559)

* Next Point-of-Interest Recommendation with Auto-Correlation Enhanced Multi-Modal Transformer Network. 
<br>SIGIR　　　　　　　　　ACM Digital Library
<br>[[Paper]](https://dl.acm.org/doi/10.1145/3477495.3531905)

* Next Point-of-Interest Recommendation with Inferring Multi-step Future Preferences.
<br> IJCAI
<br>[[Paper]](https://www.ijcai.org/proceedings/2022/521)

* Long- and Short-term Preference Learning with Enhanced Spatial Transformer for Next POI Recommendation
<br>2022 5th International Conference on Data Science and Information Technology (DSIT)　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9943896)

* POI recommendation with queuing time and user interest awareness
<br>Data Mining and Knowledge Discovery　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s10618-022-00865-w)

### Graph_Learning_Based_Models

#### 2020
* Dynamic discovery of favorite locations in spatio-temporal social networks. 
<br>Information Processing and Management.　　　　　　　　　　　ScienceDirect
<br>[[Paper]](https://doi.org/10.1016/j.ipm.2020.102337) 

* STP-UDGAT: Spatial-Temporal-Preference User Dimensional Graph Attention Network for Next POI Recommendation. 
<br>CIKM. 
<br>[[Paper]](https://doi.org/10.1145/3340531.3411876)

* STGCN: A Spatial-Temporal Aware Graph Learning Method for POI Recommendation. 
<br>ICDM. 
<br>[[Paper]](https://ieeexplore.ieee.org/document/9338281) 

* Heterogeneous graph-based joint representation learning for users and POIs in location-based social network. 
<br>Information Processing & Management　　　　　　　　　　　　ScienceDirect
<br>[[Paper]](https://doi.org/10.1016/j.ipm.2019.102151) 

* Modeling spatial trajectories with attribute representation learning. 
<br>IEEE Transactions on Knowledge and Data Engineering. 
<br>[[Paper]](https://ieeexplore.ieee.org/document/9112685) 

* Heterogeneous graph-based joint representation learning for users and POIs in location-based social network
<br>Information Processing & Management　　　　　　　　　　　　ScienceDirect
<br>[[Paper]](https://www.sciencedirect.com/science/article/pii/S0306457319305114) 

* Spatio-Temporal Dual Graph Attention Network for Query-POI Matching
<br>SIGIR　　　　　　　　　　　　ACM
<br>[[Paper]](https://doi.org/10.1145/3397271.3401159) 

* Hybrid graph convolutional networks with multi-head attention for location recommendation
<br>World Wide Web　　　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/article/10.1007/s11280-020-00824-9) 

* A neural multi-context modeling framework for personalized attraction recommendation
<br>Multimedia Tools and Applications　　　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/article/10.1007/s11042-019-08554-5) 

* Relation Embedding for Personalised Translation-Based POI Recommendation
<br>PAKDD　　　　　　　　　　　　Springer
<br>[[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_5)


#### 2021
* A graph embedding based model for fine-grained POI recommendation**. 
<br>Neurocomputing. 
<br>[[Paper]](https://doi.org/10.1016/j.neucom.2020.01.118)

* Incremental Spatio-Temporal Graph Learning for Online Query-POI Matching. 
<br>WWW 　　　　　　　　　ACM
<br>[[Paper]](https://doi.org/10.1145/3442381.3449810)

* Time-aware User Modeling with Check-in Time Prediction for Next POI Recommendation. 
<br>ICWS. 
<br>[[Paper]](https://ieeexplore.ieee.org/abstract/document/9590322) 

* DynaPosGNN: Dynamic-Positional GNN for Next POI Recommendation. 
<br>ICDMW. 
<br>[[Paper]](https://ieeexplore.ieee.org/abstract/document/9680032) 

* Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation. 
<br>IJCAI 
<br>[[paper]](https://www.ijcai.org/proceedings/2021/206)

* You Are What and Where You Are: Graph Enhanced Attention Network for Explainable POI Recommendation.
<br>CIKM. 
<br>[[paper]](https://doi.org/10.1145/3459637.3481962)

* Multi-network Embedding for Missing Point-of-Interest Identification
<br> 2021 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Big Data & Cloud Computing, Sustainable Computing & Communications, Social Computing & Networking (ISPA/BDCloud/SocialCom/SustainCom)　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9644693)

* CTHGAT:Category-aware and Time-aware Next Point-of-Interest via Heterogeneous Graph Attention Network
<br>2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC)　　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9658805)

* SgWalk: Location Recommendation by User Subgraph-Based Graph Embedding.
<br>IEEE Access　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9551972)

* Personalized Point-of-Interest Recommendation Based on Social and Geographical Influence.
<br>AICCC　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3508259.3508278)

* Attentive sequential model based on graph neural network for next poi recommendation
<br>World Wide Web 　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s11280-021-00961-9)



#### 2022
* Building and exploiting spatial–temporal knowledge graph for next POI recommendation. 
<br>Knowledge-Based Systems  　　　　　　　　　　ScienceDirect
<br>[[Paper]](https://doi.org/10.1016/j.knosys.2022.109951) 

* Learning Graph-based Disentangled Representations for Next POI Recommendation. 
<br>SIGIR　　　　　　　　　　　　　ACM
<br>[[Paper]](https://doi.org/10.1145/3477495.3532012) 

* An Attention-Based Spatiotemporal GGNN for Next POI Recommendation. 
<br>IEEE Access  
<br>[[Paper]](https://ieeexplore.ieee.org/abstract/document/9727181) 

* Modeling Spatio-temporal Neighbourhood for Personalized Point-of-interest Recommendation. 
<br>IJCAI
<br>[[Paper]](https://www.ijcai.org/proceedings/2022/0490.pdf) 

* Graph-Flashback Network for Next Location Recommendation. 
<br>KDD  
<br>[[Paper]](https://doi.org/10.1145/3534678.3539383) 

* Interaction-Enhanced and Time-Aware Graph Convolutional Network for Successive Point-of-Interest Recommendation in Traveling Enterprises.  
<br>IEEE Transactions on Industrial Informatics. 
<br>[[Paper]](https://ieeexplore.ieee.org/document/9863644) 

* Learning Graph-Based Geographical Latent Representation for Point-of-Interest Recommendation.
<br>CIKM 
<br>[[paper]](https://doi.org/10.1145/3340531.3411905)

* A POI Recommendation Model with Temporal-Regional Based Graph Representation Learning.
<br>2022 IEEE 5th International Conference on Information Systems and Computer Aided Education (ICISCAE)　　　　　IEEE
<br>[[Paper]](https://ieeexplore.ieee.org/document/9927672) 

* Spatio-Temporal Digraph Convolutional Network-Based Taxi Pickup Location Recommendation.
<br> IEEE Transactions on Industrial Informatics　　　　　IEEE
<br>[[Paper]](https://ieeexplore.ieee.org/document/9793719) 

* Location Recommendation Based on Mobility Graph With Individual and Group Influences.
<br>IEEE Transactions on Intelligent Transportation Systems.　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9714731)

* A top-k POI recommendation approach based on LBSN and multi-graph fusion
<br>Neurocomputing　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0925231222013303)

* FG-CF: Friends-aware graph collaborative filtering for POI recommendation
<br>Neurocomputing　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0925231222002399)

* Trust-aware location recommendation in location-based social networks: A graph-based approach
<br>Expert Systems with Applications　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0957417422020668)

* Contextual spatio-temporal graph representation learning for reinforced human mobility mining
<br>Information Sciences　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0020025522004819)

* TransMKR: Translation-based knowledge graph enhanced multi-task point-of-interest recommendation
<br>Neurocomputing　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0925231221017239)

* A points of interest recommendation framework based on effective representation of heterogeneous nodes in the Internet of Things
<br>Computer Communications　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0140366422003590)

* Dual-grained human mobility learning for location-aware trip recommendation with spatial–temporal graph knowledge fusion
<br>Information Fusion　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S1566253522002287)

* Potential destination discovery for low predictability individuals based on knowledge graph
<br>Transportation Research Part C: Emerging Technologies　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0968090X22003412)

* Utilization of Real Time Behavior and Geographical Attraction for Location Recommendation
<br>ACM Transactions on Spatial Algorithms and Systems　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3484318)

* Predicting Human Mobility via Graph Convolutional Dual-attentive Networks
<br>WSDM　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3488560.3498400)

* Mining multiple sequential patterns through multi-graph representation for next point-of-interest recommendation
<br>World Wide Web　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s11280-022-01094-3)


* GARG: Anonymous Recommendation of Point-of-Interest in Mobile Networks by Graph Convolution Network
<br>Data Science and Engineering　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s41019-020-00135-z)

* GN-GCN: Combining Geographical Neighbor Concept with Graph Convolution Network for POI
<br>iiWAS　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/chapter/10.1007/978-3-031-21047-1_15)


### GAN_Models

#### 2020
* Adversarial Mobility Learning for Human Trajectory Classification. 
<br>IEEE Access. 
<br>[[Paper]](https://ieeexplore.ieee.org/document/8967063)




### Mixture_Models

#### 2020
* Personalized Geographical Influence Modeling for POI Recommendation.  
<br>IEEE Intelligent Systems 
<br>[[Paper]](https://ieeexplore.ieee.org/abstract/document/9102414)

* Cem: a convolutional embedding model for predicting next locations. 
<br>IEEE Transactions on Intelligent Transportation Systems. 
<br>[[Paper]](https://ieeexplore.ieee.org/document/9064808)

* GLR: A graph-based latent representation model for successive POI recommendation.
<br>Future Generation Computer Systems　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0167739X19303966)

* Exploiting bi-directional global transition patterns and personal preferences for missing POI category identification.
<br>Neural Networks　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S089360802030304X)

* Efficient point-of-interest recommendation with hierarchical attention mechanism
<br>Applied Soft Computing　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S1568494620304750)

* Next Location Recommendation Based on Semantic-Behavior Prediction
<br>ICBDC　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3404687.3404699)

* Exploring multiple spatio-temporal information for point-of-interest recommendation
<br>Soft Computing　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s00500-020-05107-z)

* NEXT: a neural network framework for next POI recommendation
<br>Frontiers of Computer Science volume　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s11704-018-8011-2)

* Exploiting multi-attention network with contextual influence for point-of-interest recommendation
<br>Applied Intelligence　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s10489-020-01868-0)

* Spatio-Temporal Self-Attention Network for Next POI Recommendation
<br>APWeb-WAIM　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-60259-8_30)

* DPR-Geo: A POI Recommendation Model Using Deep Neural Network and Geographical Influence
<br>ICONIP　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-63836-8_35)

* POI Recommendations Using Self-attention Based on Side Information
<br>ICPCSEE　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/chapter/10.1007/978-981-15-7984-4_5)

* From When to Where: A Multi-task Learning Approach for Next Point-of-Interest Recommendation
<br>WASA　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-59016-1_64)

* PDPNN: Modeling User Personal Dynamic Preference for Next Point-of-Interest Recommendation
<br>ICCS　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-50433-5_4)

#### 2021
* Personalized POI Recommendation: Spatio-Temporal Representation Learning with Social Tie. 
<br>DASFAA 
<br>[[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_37)

* Hierarchical and Multi-Resolution Preference Modeling for Next POI Recommendation**. 
<br>IJCNN  
<br>[[Paper]](https://ieeexplore.ieee.org/abstract/document/9533980)

* Leveraging graph neural networks for point-of-interest recommendations. 
<br>Neurocomputing　　　　　　　　ScienceDirect
<br>[[Paper]](https://doi.org/10.1016/j.neucom.2021.07.063)

* An integrated model based on deep multimodal and rank learning for point-of-interest recommendation.
<br>World Wide Web 
<br>[[paper]](https://link.springer.com/article/10.1007/s11280-021-00865-8)

* ST-PIL: Spatial-Temporal Periodic Interest Learning for Next Point-of-Interest Recommendation. 
<br>CIKM 
<br>[[Paper]](https://doi.org/10.1145/3459637.3482189)

* Social and Spatio-Temporal Learning for Contextualized Next Points-of-Interest Prediction.
<br>  2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI).　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9643230)

* Trajectory-User Linking via Graph Neural Network
<br> ICC 2021 - IEEE International Conference on Communications.　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9500836)

* Pair-wise ranking based preference learning for points-of-interest recommendation
<br>Knowledge-Based Systems　　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0950705121003324)

* Curriculum Meta-Learning for Next POI Recommendation
<br>KDD　　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3447548.3467132)

* POI Recommendation Algorithm for Mobile Social Network Based on User Perference Tracking
<br>CONF-CDS　　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3448734.3450802)

* Clustering-based Location Authority Deep Model in the Next Point-of-Interest Recommendation
<br>WI-IAT　　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3486622.3493943)

* DAN-SNR: A Deep Attentive Network for Social-aware Next Point-of-interest Recommendation
<br>ACM Transactions on Internet Technology　　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3430504)

* POI recommendation method using LSTM-attention in LBSN considering privacy protection
<br>Complex & Intelligent Systems　　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s40747-021-00440-8)


* HOPE: a hybrid deep neural model for out-of-town next POI recommendation
<br>World Wide Web　　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s11280-021-00895-2)



#### 2022
* STAN: Spatio-Temporal Attention Network for Next Location Recommendation.
<br>WWW 　　　　　　　　　ACM
<br>[[paper]](https://dl.acm.org/doi/abs/10.1145/3442381.3449998)

* Empowering Next POI Recommendation with Multi-Relational Modeling.
<br>arXiv  
<br>[[paper]](https://doi.org/10.48550/arXiv.2204.12288)

* Successive POI Recommendation via Brain-inspired Spatiotemporal Aware Representation.
<br>ICLR 
<br>[[paper]](https://openreview.net/forum?id=9W2KnHqm_xN)

* Point-of-interest recommendation model considering strength of user relationship for location-based social networks.
<br>Expert Systems with Applications.　　　　　　　　ScienceDirect
<br>[[paper]](https://doi.org/10.1016/j.eswa.2022.117147)

* CARAN: A Context-Aware Recency-Based Attention Network for Point-of-Interest Recommendation.
<br> IEEE Access　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9745902)

* Personalized Recommendation of Location-Based Services Using Spatio-Temporal-Aware Long and Short Term Neural Network.
<br>IEEE Access　　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9754510)

* Mapping user interest into hyper-spherical space: A novel POI recommendation method.
<br>Information Processing & Management　　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0306457322002709)

* Deep convolutional recurrent model for region recommendation with spatial and temporal contexts
<br>Ad Hoc Networks　　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S1570870521000937)

* CHA: Categorical Hierarchy-based Attention for Next POI Recommendation
<br>ACM Transactions on Information Systems　　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3464300)

* Decentralized Collaborative Learning Framework for Next POI Recommendation
<br>ACM Transactions on Information Systems　　　　　　　　　ACM
<br>[[paper]](https://doi.org/10.1145/3555374)

* Influence-Aware Successive Point-of-Interest Recommendation
<br>World Wide Web　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s11280-022-01055-w)

* Group-based recurrent neural network for human mobility prediction
<br>Neural Computing and Applications　　　　　　　　Springer
<br>[[paper]](https://link.springer.com/article/10.1007/s00521-022-06971-6)



### Other_Models

#### 2020
* STPR: A Personalized Next Point-of-Interest Recommendation Model with Spatio-Temporal Effects Based on Purpose Ranking.
<br>IEEE Transactions on Emerging Topics in Computing.       IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/8695772)

* Time Distribution Based Diversified Point of Interest Recommendation.
<br>2020 IEEE 5th International Conference on Cloud Computing and Big Data Analytics (ICCCBDA).       IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9095741)

* A Point-of-Interest Recommendation Algorithm Combining Social Influence and Geographic Location Based on Belief Propagation.
<br>IEEE Access.       IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9174725) 

* Exploiting Location-Based Context for POI Recommendation When Traveling to a New Region
<br>IEEE Access.　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9036967)

* Context-Enhanced Probabilistic Diffusion for Urban Point-of-Interest Recommendation.
<br>IEEE Transactions on Services Computing.　　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9444824)

* Interest Point Recommendation based on Multi Scenario Information Fusion.
<br> 2020 IEEE 2nd International Conference on Power Data Science (ICPDS)　　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9332517)

* Effective and diverse POI recommendations through complementary diversification models
<br>Expert Systems with Applications　　　　　　　　　ScienceDirect
<br>[[paper]](https://www.sciencedirect.com/science/article/pii/S0957417421002165)


#### 2022
* Online POI Recommendation: Learning Dynamic Geo-Human Interactions in Streams.
<br>  IEEE Transactions on Big Data ( Early Access )　　　　　　　　　IEEE
<br>[[paper]](https://ieeexplore.ieee.org/document/9921335)



updated in 2022.11.14
