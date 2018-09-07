# Font Recommendation with GCN

## Abstract
We propose a task where the appropriate typeface has to be predicted when given the usage description. For this task, applying classification can be inappropriate as undefined target classes may appear in real-world application. Therefore, we take an approach based on similarity between the text embedding vector of the description and the image embedding vector of the typeface. Moreover, we designed a deep learning model, Multi-Modal Graph Convolution Network Join Embedding Model (MMGJ), that can effectively perform our proposed task. Following the format of previous multimodal embedding models, our proposed MMGJ structure is consisted of embedding models for each modality and a joint embedding model. Through evaluation, we show that MMGJ outperforms baseline models. To further improve our model, we propose 1) the Intermediate MaxPooling image model (IMP) and 2) the joint embedding with Graph Convolution Network (GCN). This is the first approach in joint embeddings using graphs to our knowledge. Finally, through visualizing both the text and image embedding vector in an identical representation space, our model established the implicit meanings of typeface images with explicitly defined texts.

## SubModels
1. Text Embedding Model (LSTM)
2. Image Embedding Model(CNN)
3. Joint Embedding Model(GCN)

![Models](./figures/model.png, "models")


## Dataset Source
https://fontsinuse.com/

Graph of The Dataset

![Graph](./figures/graph.png, "graph")
