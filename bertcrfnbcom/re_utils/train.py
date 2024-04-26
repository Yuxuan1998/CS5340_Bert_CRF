import json
from typing import Dict, List, Optional

import torch
from IPython import display
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import NerDataset
from datasets.embeddings_and_relations_dataset import EmbeddingsAndRelationsDataset
from datasets.ground_truth_relations_dataset import GroundTruthRelationsDataset
from models.bert_crf import BertCrf
from models.re_bert_crf import ReBertCrf
from re_utils.common import load_json
from re_utils.ner import get_tags_with_positions, get_mean_vector_from_segment


def dict_to_device(
    dict: Dict[str, torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    for key, value in dict.items():
        dict[key] = value.to(device)
    return dict

'''
def draw_plots(loss_history: List[float], f1: List[float], mcc: List[float]):
    display.clear_output(wait=True)

    f, (ax1, ax2, ax3) = plt.subplots(3)
    f.set_figwidth(15)
    f.set_figheight(10)

    ax1.set_title("training loss")
    ax2.set_title("f1 micro")
    ax3.set_title("mcc")

    ax1.plot(loss_history)
    ax2.plot(f1)
    ax3.plot(mcc)

    plt.show()

    if len(loss_history) > 0:
        print(f"Current loss: {loss_history[-1]}")
    if len(f1) > 0:
        print(f"Current f1: {f1[-1]}")
    if len(mcc) > 0:
        print(f"Current mcc: {mcc[-1]}")
'''
def draw_plots(loss_history):
    display.clear_output(wait=True)

    f, (ax1) = plt.subplots(1)
    f.set_figwidth(15)
    f.set_figheight(5)

    ax1.set_title("training loss")

    ax1.plot(loss_history)
    plt.show()

def combine_subtoken(prediction, label, word_ids):
    i = 0
    while i < len(word_ids)-1:
        if word_ids[i] == word_ids[i+1]:
            if word_ids[i] != -1:
                del label[i]
                del prediction[i]
                del word_ids[i]
            else:
                i += 1
        else:
                i += 1
    return prediction, label


def train_ner(
    num_labels: int,
    bert_name: str,
    train_tokenized_texts_path: str,
    test_tokenized_texts_path: str,
    dropout: float,
    batch_size: int,
    epochs: int,
    log_every: int,
    lr_bert: float,
    lr_new_layers: float,
    use_crf: bool = True,
    save_to: Optional[str] = None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = BertCrf(num_labels, bert_name, dropout=dropout, use_crf=use_crf)
    model = model.to(device)
    model.train()

    train_dataset = NerDataset(train_tokenized_texts_path)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_function,
    )

    test_dataset = NerDataset(test_tokenized_texts_path)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_function,
    )

    optimizer = Adam(
        [
            {"params": model.start_transitions},
            {"params": model.end_transitions},
            {"params": model.hidden2label.parameters()},
            {"params": model.transitions},
            {"params": model.bert.parameters(), "lr": lr_bert},
        ],
        lr=lr_new_layers,
    )

    loss_history = []
    f1_micro = []
    f1_macro = []
    mcc = []

    step = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        for batch in tqdm(train_data_loader):
            step += 1

            if step % 20 == 0:
                display.clear_output(wait=True)
                print(f"Epoch {epoch}/{epochs}")
                print('step:', step)
                print(f"Current loss: {loss_history[-1]}")

            optimizer.zero_grad()

            batch = dict_to_device(batch, device)

            loss = model(**batch)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            '''
            if step % log_every == 0:
                model.eval()
                predictions = []
                ground_truth = []
                with torch.no_grad():
                    for batch in test_data_loader:
                        labels = batch["labels"]
                        del batch["labels"]
                        batch = dict_to_device(batch)

                        prediction = model.decode(**batch)

                        flatten_prediction = [item for sublist in prediction for item in sublist]
                        flatten_labels = torch.masked_select(labels, batch["attention_mask"].bool().cpu()).tolist()

                        predictions.extend(flatten_prediction)
                        ground_truth.extend(flatten_labels)
                f1_micro = f1_score(ground_truth, predictions, average="micro")
                f1.append(f1_micro)
                mcc_score = matthews_corrcoef(ground_truth, predictions)
                mcc.append(mcc_score)

                print(f"Current f1: {f1[-1]}")
                print(f"Current mcc: {mcc[-1]}")

                model.train()
            '''
        #draw_plots(loss_history, f1, mcc)

        model.eval()
        predictions = []
        ground_truth = []
        with torch.no_grad():
            for batch in test_data_loader:
                labels = batch["labels"]
                del batch["labels"]
                batch = dict_to_device(batch)

                prediction = model.decode(**batch)

                flatten_prediction = [item for sublist in prediction for item in sublist]
                flatten_labels = torch.masked_select(labels, batch["attention_mask"].bool().cpu()).tolist()

                word_ids = batch["word_ids"]
                word_ids = torch.masked_select(word_ids.cpu(), batch["attention_mask"].bool().cpu()).tolist()

                combined_prediction, combined_labels = combine_subtoken(flatten_prediction, flatten_labels, word_ids)

                predictions.extend(combined_prediction)
                ground_truth.extend(combined_labels)

        f1_micro.append(f1_score(ground_truth, predictions, average="micro"))
        f1_macro.append(f1_score(ground_truth, predictions, average="macro"))
        mcc.append(matthews_corrcoef(ground_truth, predictions))
        draw_plots(loss_history)
        print(f"Current loss: {loss_history[-1]}")
        print(f"f1_micro: {f1_micro}")
        print(f"f1_macro: {f1_macro}")
        print(f"mcc: {mcc}")
        model.train()

        result = classification_report(ground_truth, predictions)
        print(result)
        break

    if save_to is not None:
        model.save_to(save_to)