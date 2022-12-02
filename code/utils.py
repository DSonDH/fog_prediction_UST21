from typing import *
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, jaccard_score, make_scorer,
                             mean_tweedie_deviance, precision_score,
                             recall_score, roc_auc_score)


def calc_metrics(obs: pd.Series, pred: np.ndarray, binary=True) -> Dict:

    if binary:
        if len(confusion_matrix(obs, pred).flatten()) == 1 and obs[0] == 0:
            print(f'!!! all obs and all pred was no-fog')
            assert (1 in obs) or (1 in pred), 'There is no fog in this CV split'
            # or 애초에 이런 CV split을 제끼는게 맞는듯
        
        #!!! average='macro' should not be used when binary classification
        tn, fp, fn, tp = confusion_matrix(obs, pred).flatten()
        metrics = dict(
            ACC=accuracy_score(obs, pred),
            # CSI=jaccard_score(obs, pred, average='macro'),
            PAG=precision_score(obs, pred, average='macro'),
            POD=recall_score(obs, pred, average='macro'),
            F1=f1_score(obs, pred, average='macro'),
            TN=tn,
            FP=fp,
            FN=fn,
            TP=tp,
        )
        # test = 2*tp/(2*tp + fp + fn)
        # test3 = 2*tn/(2*tn + fp + fn)

    else:  # 3분류, macro micro 계산
        metrics = {}
        cm1, cm2, cm3, cm4, cm5, cm6, cm7, cm8, cm9 = \
                                        confusion_matrix(obs, pred).flatten()

        metrics["ACC"] = accuracy_score(obs, pred)
        # metrics["macro_CSI"] = jaccard_score(obs, pred, average="macro")
        metrics["macro_PAG"] = precision_score(obs, pred, average="macro")
        metrics["macro_POD"] = recall_score(obs, pred, average="macro")
        metrics["macro_F1"] = f1_score(obs, pred, average="macro")

        # metrics["micro_CSI"] = jaccard_score(obs, pred, average="micro")
        # metrics["micro_PAG"] = precision_score(obs, pred, average="micro")
        # metrics["micro_POD"] = recall_score(obs, pred, average="micro")
        # metrics["micro_F1"] = f1_score(obs, pred, average="micro")
        metrics['cm1'] = cm1
        metrics['cm2'] = cm2
        metrics['cm3'] = cm3
        metrics['cm4'] = cm4
        metrics['cm5'] = cm5
        metrics['cm6'] = cm6
        metrics['cm7'] = cm7
        metrics['cm8'] = cm8
        metrics['cm9'] = cm9

    metrics = { k: float(v) for k, v in metrics.items()}
    return metrics


if __name__ == "__main__":
    ...