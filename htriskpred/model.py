from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers_interpret import SequenceClassificationExplainer
import evaluate
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
auc_metric = evaluate.load("roc_auc")


def compute_metrics(eval_pred):
    prediction_scores, labels = eval_pred
    prediction_scores = torch.tensor(prediction_scores).to(torch.float32).softmax(dim=1).numpy()
    predictions = np.argmax(prediction_scores, axis=1)
    positive_scores = prediction_scores[:,1]
    metrics = clf_metrics.compute(predictions=predictions, references=labels)
    auc_score = auc_metric.compute(prediction_scores=positive_scores, references=labels)
    precisions, recalls, thresholds = precision_recall_curve(labels, positive_scores)
    pr_auc = auc(recalls, precisions)
    metrics = dict(list(metrics.items()) + list(auc_score.items()))
    metrics['pr_auc'] = pr_auc
    f1_scores = 2*recalls*precisions/(recalls+precisions)
    metrics['best_threshold'] = thresholds[np.argmax(f1_scores)]
    metrics['best_f1']=np.max(f1_scores)
    return metrics

def get_model(pretrained_model_name_or_path: str, additional_tokens = []):
    """
    Params:
            pretrained_model_name_or_path:
                Can be either:
                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                      applicable to all derived classes)
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, additional_special_tokens=additional_tokens)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path, num_labels=2, ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.problem_type = "single_label_classification"

    return model, tokenizer, data_collator


def train(
    training_args: TrainingArguments,
    model,
    tokenizer,
    tokenized_datasets,
    data_collator,
    compute_metrics,
):
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    torch.cuda.empty_cache()

    trainer.train()

    trainer.save_model()


def test(model, eval_dataset, test_args, compute_metrics):
    # init trainer
    trainer = Trainer(
        model=model,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=test_args,
    )

    return trainer.evaluate()


def sample_explainer(text: str, model, tokenizer, output_file: str):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(text)

    cls_explainer.visualize(output_file)

    print(word_attributions)
