import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from transformers import AutoTokenizer, AutoModel


class Solution:
    def __init__(self,
                 base_model = "bert-base-uncased",
                 dataset_id="Djacon/ru-izard-emotions",
                 ):

        self.dataset_id = dataset_id
        self.base_model = base_model
        
        self.dataset = self.load_data()
        self.labels = self._get_labels()
        self.id2label = self._get_id2label()
        self.label2id = self._get_label2id()

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

        print('Start preprocessed dataset')
        self.encoded_dataset = self.dataset.map(self.preprocess_data, batched=True, remove_columns=self.dataset['train'].column_names)
        self.encoded_dataset.set_format("torch")
        print('Dataset preprocessed succesfuly')
        self.trainer = None
        


    def load_data(self):
        print(f'Start loading {self.dataset_id} dataset')
        dataset = load_dataset(path=self.dataset_id)
        print(dataset)
        print("Dataset loaded successfully!")
        return dataset


    def _get_labels(self):
        labels = [label for label in self.dataset['train'].features.keys() if label not in ['text']]
        print(f"labels: {labels}")
        return labels

    def _get_id2label(self):
        id2label = {idx:label for idx, label in enumerate(self.labels)}
        return id2label

    def _get_label2id(self):
        label2id = {label:idx for idx, label in enumerate(self.labels)}
        return label2id

    def load_model(self):
        print(f"Start loading {self.base_model} model")
        model = AutoModelForSequenceClassification.from_pretrained(self.base_model,
                                                                   problem_type="multi_label_classification", 
                                                                   num_labels=len(self.labels),
                                                                   id2label=self.id2label,
                                                                   label2id=self.label2id
                                                                   )
        print('Model loaded successfully')
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return tokenizer

    def preprocess_data(self, data):
        # take a batch of texts
        text = data["text"]
        # encode them
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: data[k] for k in data.keys() if k in self.labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
    
        return encoding

    def fit(self, 
            output_dir,
            batch_size=8,
            metric_name='f1',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate =2e-5,
            num_epochs=5,
            weight_decay=0.001,
            load_best_model_at_end=True,
            push_to_hub=False,
            save_model=True
            ):

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=load_best_model_at_end,
            push_to_hub=push_to_hub
        )
        print(f"Start to train {self.base_model} with params {training_args}")

        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=self.encoded_dataset["train"],
            eval_dataset=self.encoded_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.evaluate()
        
        if save_model is True:
            trainer.save_model(output_dir=output_dir)
            print(F"Fine-Tuned model saved to {output_dir}")

        self.trainer = trainer
        
        return trainer


    def predict(self, text, treshold=0.5):
        encoding = self.tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(self.trainer.model.device) for k,v in encoding.items()}

        outputs = self.trainer.model(**encoding)
        logits = outputs.logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= treshold)] = 1
        predicted_labels = [self.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]

        return predicted_labels


    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics
    
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = self.multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result
    


solution = Solution()
test_trainer = solution.fit(output_dir='bert-finetuned-sem_eval-english', num_epochs=1)
print(solution.predict('Я боюсь оказаться в неловком положении'))
