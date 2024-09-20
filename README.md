# Task Requirements

## 1. Data preprocessing
1. Found and droped the rows with all NaN columns (in practice you could pseudo-label them or label them manually, but there were only about 10 of them, so I just deleted them).
2. Data imbalance is detected, which forces the use of iterative_train_split, which takes this fact into account, as well as the corresponding metrics f_1, accuracy and ROC AUC, where ROC AUC is the most useful, i.e. it describes the best performance of the model when the data is unbalanced.
3. Transformed the 4 column string labels dataset into a 6 column binary labels dataset where each column is the on of 6 unique classes - Chief Officer, Director, Individual Contributor/Staff, Manager, Owner, Vice President. This will make it easier for me to use multi-label classification models. We just have to bear in mind that if we want to use this model in production, then we need to select 4 out of 6 columns for the result. We have in this dataset 16 possible combinations of unique labels from 4 columns (dropping the all NaN), so in theory we can recieve the title that will have more than 0.5 value of more than 4 labels, in my opinion in this case we judt need to choose the most appropriate combination of labels.
4. To process the text data - Title columns, I choose the RoBERTa-base Tokenizer, because I will use this pre-trained transformer with the Hugging Face library. To process the labels, as I underlined above in section 3. I converted them into a binary representation of the corresponding unique column label.

## 2. Model and architecture selection
1. For text tasks, the best performing models are pre-trained Transformer-based models. In the question of which model is best suited for this task, bearing in mind that the state of the art models are not always the best, I choose the simpler and smaller pre-trained Transformer - RoBERTa based model. I also choose Gradient Boosting for a small increase in performance, Gradient Boosting is more suitable for unbalanced data as we have here.
2. Reasons (Performance, Complexity, Suitability) to choose these algorithms and models are:
    1. Pre-trained transformers are the best for text classification tasks because they trained on huge amount of text data.
    2. RoBERTa-base transformer is small enough for my Laptop and for this model 1 good GPU and having small dataset will be overkill.
    3. Gradient Boosting is good for unbalanced data. 
    4. Ensemble always uses for small boost in performance so I will use RoBERTa-base and Gradient Boosting Ensemble.
3. Architectures
    1. RoBERTa-base model have the next architecture: 12 layers, 768 hidden units, 12 attention heads, 110M parameters
    (sameas BERT-base). In my case in the end I pass the hidden states through the classifier to obtain the logits, so my last layer is binary multi-label.
    2. Gradient Boosting params are the third .ipynb file. For Gradient Boosting used the embeddings from RoBERTa-base model and 6 binary labels.

## 3. Training and testing the model
1. Split the unbalanced dataset using iterative_train_split as 80/20 (training dataset 80% - dev or hold out cross validation dataset 20%). We choose 80/20 because we only have 2230 examples (without NaN rows), so about 70/30 to 80/20 is a good choice. If we had a million then the 99/1 would be the best choice or even better 98/1/1 - Create also test set.
2. Metrics are 3: Accuracy, F-1 score and ROC AUC score.
    1. Accuracy for imbalanced data doesn't make any sense, but I added just for completion.
    2. F-1 Score is important for understanding the Precision and Recall balance.
    3. ROC AUC Score is the best choice for evaluating overall model's performance.

## 4. Interpreting results
1. The ROC AUC score takes into account the true positive rate and the false positive rate, which means that if the ROC AUC has a value of 0.5 - it is just a random classifier, but as it gets closer to 1 - it means it is the ideal classifier (overfitting in most practical cases, but theoretically ideal).

## 5. Documenting Decisions
1. Documenting decisions splitted across the README.md file and all .ipynb files.

## 6. Code and reproducibility
1. Before the start: Make sure you did these steps:
    1. Installed docker
    2. docker login (you need to sign up in docker hub)
    3. Installed the NVIDIA Container Toolkit

2. Created .devcontainer folder for Visual Studio Dev Containers
    1. Uploaded docker file which will create mamba environment with all required packages for starting the .ipynb files.
    2. Uploaded devcontainer.json with useful dependencies.
3. Also added .gitignore file.

## 7 Results and coclusions
1. I trained the RoBERTa-base pre-trained Transformer for 60 epochs ~ 1 hour, I recieved ROC AUC score of 0.9392004863961088.
2. Ensemble of RoBERTa and Gradient Boosting gives me the ROC AUC score of 0.9647163724883697.
3. Eventually 0.9647163724883697 it's a solid result.
4. Strenght and weaknesses are trivial: the best predicted label is what in the dataset is the most, so we can fix it only by gathering more data and making our dataset balanced or making the label weights.
5. Recommendations for improving:
    1. Possible Data Augmentation: 
        1. Using Back Translation (for example from English to German and vice versa), but I'm not sure about that as this approach will be 100% good for comment texts because in this way out model will have more knowledge of how people can speak, but in our case titles have some templates for example you always AI/ML Engineer or AI/ML Developer but not AI/ML Worker or something like that. Maybe titles also small enough and back translation will not help as with the long sentenceses. So, it's open question for me.
        2. SMOTE I think here will not help, because we have text classification task which includes high demension embeddings, so it's computationally expensive choice, but maybe not so expensive as I think.
    2. Data Pseudo-Labelling or Manual Labelling of 10 droped NaN rows from the dataset.
    3. Longer Training of RoBERTa Model
    4. More fine-tuning techniques: Optimising the all possible Hyperparameters:
        1. Regularization strength
        2. Batch size
        3. Learning rate
        4. beta1 and beta2 in Adam optimizer
        5. etc.



# Mamba(Conda) envs:

To get mamba env:
1. mamba env export > environment.yml

To install mamba env:
1. mamba env create -f environment.yml