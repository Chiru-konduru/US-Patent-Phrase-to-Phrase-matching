# US-Patent-Phrase-Similarity-CLassifier

This project is aimed at developing a model that identifies similarities between phrases found in patent documents, designed to assist patent professionals in finding relevant information from millions of such documents. The model is trained on a semantic similarity dataset, and is capable of recognizing different phrases that refer to the same concept, effectively providing a similarity measure between the phrases in question.

The project also incorporates context to understand similarity across domains. This is a 5-class classification problem with classes representing varying degrees of relatedness, ranging from 0 (unrelated) to 1 (very close match).

Data Preprocessing
Text Embedding
Text data from the anchor and target input features is converted into numerical features using a pre-trained model PatentSBERTa, which is fine-tuned for patent classification and distance calculation. The model takes a text from the patent document as input and produces a 768-dimensional real-valued numerical vector as the semantic embedding of the text. The anchor and target texts are passed separately to the PatentSBERTa model, yielding two 768-dimensional vectors and thus producing 1536 numerical features.

Categorical Context
The context variable in the dataset denotes the category of the patent as per Cooperative Patent Classification. The variable is one-hot-encoded, producing 106 binary input features. In total, 1642 input features are utilized, out of which 1536 are real-valued and 106 are binary.

Data Size
The Kaggle competition provides a training dataset with 36473 samples and a test dataset with 36 samples. Since the test dataset doesn't provide ground truth and is relatively small, the Kaggle train dataset is divided into three random splits - train (26351 samples), validation (4651 samples), and test (5471 samples).

Baseline Models
Two baseline models, Logistic Regression Classifier and Randomforest Classifier, were initially utilized on the preprocessed data. The models were initially manually tweaked for hyper-parameter values before Bayesian optimization was applied for hyper-parameter tuning. However, these approaches did not yield the best results.

Proposed Models
Proposed Model Version 1
This is an ANN model with 1642 input features, 500 hidden features, and 5 output features, utilizing a ReLU activation function on the hidden layer. The model is trained on our train set by optimizing cross-entropy loss with the Adam optimizer.

Proposed Model Version 2
The second version involves taking the Hadamard product of the anchor and target embeddings, effectively reducing the number of embedding features to 768. This model uses these 768 features along with 106 one-hot-encoded context features to train an ANN. The network includes 5 hidden layers with sizes [400, 600, 800, 500, 200] respectively.

Proposed Model Version 3
The third version of the proposed model involves using CPC keywords instead of a categorical variable as context for more accurate embeddings. Instead of getting separate embeddings for target, anchor, and context, all three features are concatenated and got embeddings in one shot. This ANN has 768 input features and 3 hidden layers of sizes [1500,1500,500] respectively.







