# DeepfakeTextClassifier


## Running our Project

- Use pip to install requirements.txt 
- optional: run test_gltr_gpt2.sh to preprocess the final_full_data.jsonl file. This will append to gltr_processed.jsonl file, so if done it should be removed before hand. Note: This execution took multiple hours when last run. 
- run classifier_builder.py to build the classifier using our processed GLTR data. This will save onto ./my_model, an already completed copy of which is provided. 
- run ./run_classifier.sh. This will create predictions for the text file in --test_dataset