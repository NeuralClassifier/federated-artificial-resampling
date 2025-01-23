# FedAR: Federated Artificial Resampling for Imbalanced Facial Emotion Recognition

***Published in IEEE Transactions on Affective Computing***

**Paper URL:** [IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10825917)

**Abstract:** Federated Learning (FL) has emerged as an essential tool for computing devices to participate in collaborative training of deep learning models. However, due to the decentralized distribution of data over clients/local computing devices, the class imbalance problem has become evident, causing severe degradation in the performance of the global model. Motivated by the emergence of FL models in emotion recognition, the current study proposes an FL-based facial emotion recognition system by addressing local imbalance data problems encountered in client devices. Firstly, the local imbalance problem is mitigated by utilizing the data-level artificial resampling method on the client side. To address the possibility of an adversarial attack using imbalanced data, the local training is equipped with a pre-training check to verify if the data being used is imbalanced above a predefined threshold of imbalance ratio. In case of high imbalance, a pre-training step will balance the data locally without sharing any information with other participants thereby ensuring privacy in the FL framework. Experiments have been conducted by using benchmark facial emotion recognition data with a balanced testing strategy. It indicated that considerable improvement can be achieved by the proposed FL-based facial emotion recognition model.

## Authors: 
1. Sankhadeep Chatterjee
2. Kushankur Ghosh
3. Saranya Bhattacharjee
4. Asit Kumar Das
5. Soumen Banerjee

## Citation: 
```
@article{chatterjee2024federated,
  title={FedAR: Federated Artificial Resampling for Imbalanced Facial Emotion Recognition},
  author={Chatterjee, Sankhadeep and Ghosh, Kushankur and Bhattacharjee, Saranya and Das, Asit Kumar and Banerjee, Soumen},
  journal={IEEE Transactions on Affective Computing},
  year={2024},
  publisher={IEEE}
}
```
