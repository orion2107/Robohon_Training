# Robohon_Training
This repository Includes all python code we use in this project:
1. In the main directory we have the 2 ipynv files we use for train and test the model via colab:
    a. aleph_bert_finetune.ipynb - python code training the model.
    b. alephbert_inference.ipynb - python code for testing the model.

2. In the data folder we have files which we use to train the model:
    a. full_train_set.csv - train set
    b. default_sentence_list.utf8.csv - basic sentence list we use for similarity.
    
3. In the azure_files folder we have all files which we need to deploy to cloud.

4. In the main folder we also have the 'azure_model_deployment_instructions.pdf' file which explains all steps for deploying the code in azure ML cloud service.

5. In the model folder we can put the trained model file used for testing (currently the folder is empty because when we deploy to cloud we upload the model to dedicated storage).
