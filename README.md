
## Clustered Federated Learning Based on Mahalanobis Distance for Sequential Medical Data
____

### Abstract
In hospitals, metadata typically contains patient personal information based on the doctor's diagnosis.
Therefore, sniffers or hijackers could launch attacks to steal important information from hospitals or patients.
For this reason, hospital data must be anonymized and protected by specialized systems to ensure its safe use,
especially when multiple hospitals share data. If hospitals implement systems that can securely share data while
maintaining privacy, researchers and clinicians can leverage large amounts of distributed data to more
effectively train deep learning models. In this context, we select a solution based on Clustered Federated
Learning (CFL). In typical CFL scenarios, forming appropriate clusters can help build more personalized
models for different groups. However, previous CFL approaches still face challenges from model
heterogeneity. To further mitigate the heterogeneity problem, we propose a Mahalanobis Distance based
Clustered Federated Learning (MD-CFL) method, which offers advantages in reducing model heterogeneity
and improving clustering performance by correcting for feature skew in non-normalized data. Our experiments
show that MD-CFL achieves accurate clustering performance, with a higher Silhouette score compared to
cosine-based FedAvg.

### Introduction
In recent years, researchers have been able to aggregate metadata from medical devices using various
sensors. This metadata includes users' private data from distributed sources, making privacy a critical
concern for training AI models. Federated learning (FL) allows the model to be trained by sharing only
parameters with a central server, without exposing the user's actual data. This feature of federated learning
helps protect user privacy while allowing researchers to train AI models [1]. However, federated learning
has its limitations, especially regarding model heterogeneity. This paper focuses on solving the problem
of model heterogeneity by proposing a clustering method based on Mahalanobis distance, called
Mahalanobis Distance based Clustered Federated Learning (MD-CFL). The method is designed to
mitigate model heterogeneity. This research compares the performance of cosine-based FedAvg with our
MD-CFL based FedAvg on the WESAD (Wearable Stress and Affect Detection) and K-EmoCon datasets.
Experimental results show that under conditions of model heterogeneity, our MD-CFL based clustering
method outperforms the cosine-based method and achieves a higher silhouette score.
__Clustered federated learning (CFL) is one of the fundamental methods for addressing data heterogeneity
in distributed environments. In scenarios with data heterogeneity, federated learning typically performs
worse than CFL because heterogeneity exists among distributed data. This paper provides an overview
of federated learning and clustered federated learning.__