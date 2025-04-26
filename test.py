import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.preprocess import preprocess
from utils.dataloader import DataGen
from utils.metrics import Metrics, AUC, metric_summary


def plot_confusion_matrix(cm, class_names, font_size=14, value_font_size=10):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": value_font_size}  # Font size for the values inside the matrix
    )
    plt.title('Confusion Matrix', fontsize=font_size + 2)
    plt.ylabel('True label', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


def test(models, path: str = "data/ptb", batch_size: int = 32, name: str = "ensemble") -> None:
    _, _, X_test_scale, y_test, _, _ = preprocess(path=path)
    test_gen = DataGen(X_test_scale, y_test, batch_size=batch_size)

    preds_all_models = []

    for model in models:
        preds_model = []
        for X, _ in tqdm(test_gen, desc=f"Testing Model {models.index(model)}"):
            preds = model.predict(X)
            preds_model.extend(preds.tolist())
        preds_all_models.append(np.array(preds_model))

    # Average predictions of all models to get ensemble predictions
    ensemble_pred_all = np.mean(np.array(preds_all_models), axis=0)

    # Calculate performance metrics
    roc_score = roc_auc_score(y_test, ensemble_pred_all, average="macro")
    acc, mean_acc = Metrics(y_test, ensemble_pred_all)
    class_auc = AUC(y_test, ensemble_pred_all)
    summary = metric_summary(y_test, ensemble_pred_all)

    # Generate the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(ensemble_pred_all, axis=1))
    custom_class_names = ['NORM', 'MI','STTC','CD','HYP']  # Replace with your actual class labels
    plot_confusion_matrix(cm, custom_class_names, font_size=16, value_font_size=22)

    # Log output
    logs = {
        "roc_score": roc_score,
        "mean_acc": mean_acc,
        "accuracy": acc,
        "class_auc": class_auc,
        "F1 score (Max)": summary[0],
        "class_precision_recall_f1": summary,
        "confusion_matrix": cm.tolist()  # Save matrix as list for JSON serialization
    }
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    with open(os.path.join(logs_path, f"{name}_test_logs.json"), "w") as json_file:
        json.dump(logs, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ptb", help="Ptb-xl dataset location")
    parser.add_argument("--model", type=str, default="tecc", help="Model ID for checkpoint naming.")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Build or load models into a list for the ensemble
    models = []
    for i in range(7):  # num_models is the number of models in your ensemble
        from models.TECC import build_tecc
        from configs.tecc_config import Config

        model = build_tecc(Config())
        path_weights = os.path.join(os.getcwd(), "checkpoints", f"{args.model}_weights_{i}.h5")
        model.load_weights(path_weights)
        models.append(model)

    # Test the ensemble of models
    test(models, path=args.data_dir, batch_size=args.batchsize, name=args.model)
