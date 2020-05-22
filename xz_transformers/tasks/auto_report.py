# -*- coding: utf-8 -*-
# Created by xieenning at 2020/5/19
"""Auto report."""
import sys
import os
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../'))
import time
import numpy as np
import pandas as pd
from rich.progress import track, Progress, TextColumn, BarColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.console import Console
from rich.theme import Theme
from rich import box
from rich.table import Table
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from xz_transformers.tasks.about_sequence_pair_classification import SequencePairClassification

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tmp_progress_part = Progress(
    TextColumn("[bold red]{task.fields[model_name]}", justify="right"),
    "-",
    TextColumn("[yellow]{task.fields[data_type]}", justify="left"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn()
)

custom_theme = Theme({
    "info": "cyan",
    "warning": "magenta",
    "danger": "bold red"})

console = Console(record=True, theme=custom_theme)

model_2_speed = {
    'chinese-bert-wwm': 255,
    'chinese-bert-wwm-ext': 259,
    'chinese-rbt3': 83,
    'chinese-rbtl3': 110,
    'chinese-roberta-wwm-ext': 257,
    'chinese_simbert_zhuiyi': 256,
    'roberta_chinese_3L312_clue_tiny': 39,
    'roberta_chinese_3L768_clue_tiny': 81,
    'roberta_chinese_clue_tiny': 59,
    'roberta_chinese_pair_tiny': 60
}


def construct_table(dataset_name, data_dict):
    # print(data_dict)
    table = Table(title=f"各模型在[red]{dataset_name}[/red]数据集上的表现", box=box.ROUNDED)

    table.add_column("预训练模型", justify="center")
    table.add_column("Speed", justify="center")
    table.add_column("F1", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Threshold", justify="center")
    keys_sorted = sorted(data_dict, key=lambda x: data_dict.get(x)['F1'])[::-1]
    for tmp_model_name in keys_sorted:
        tmp_metrics_score = data_dict.get(tmp_model_name)
        table.add_row(f"{tmp_model_name}",
                      f"{model_2_speed.get(tmp_model_name, '-')}",
                      f"{tmp_metrics_score.get('F1', '-'):.2f}",
                      f"{tmp_metrics_score.get('Precision', '-'):.2f}",
                      f"{tmp_metrics_score.get('Recall', '-'):.2f}",
                      f"{tmp_metrics_score.get('Accuracy', '-'):.2f}",
                      f"{tmp_metrics_score.get('Threshold', '-'):.2f}")
    console.print(table)


if __name__ == '__main__':
    tmp_ptm_short_names = ["chinese-rbt3", "chinese-rbtl3", "roberta_chinese_3L312_clue_tiny",
                           "roberta_chinese_3L768_clue_tiny", "roberta_chinese_clue_tiny", "roberta_chinese_pair_tiny",
                           "chinese_simbert_zhuiyi", "chinese-bert-wwm", "chinese-bert-wwm-ext",
                           "chinese-roberta-wwm-ext"]
    # tmp_ptm_short_names = ["chinese-rbt3", "chinese-rbtl3"]
    models_folder = 'intent_detection_0515'
    valuable_data_paths = {'Std': '/Data/enningxie/Codes/transformers_xz/data/LCQMC/test.tsv',
                           'Custom_01': '/Data/enningxie/Codes/Notebooks/data/test_df.tsv',
                           'Custom_02': '/Data/enningxie/Codes/Notebooks/data/test_df_0509.tsv',
                           'Custom_03': '/Data/enningxie/Codes/Notebooks/data/test_df_0511.tsv',
                           'Custom_04': '/Data/enningxie/Codes/Notebooks/data/test_df_0515.tsv'}
    data_2_model = defaultdict(dict)
    with tmp_progress_part:
        for model_name in tmp_ptm_short_names:
            tmp_saved_model_path = '/Data/enningxie/Codes/transformers_xz/saved_models/{}/{}'.format(models_folder,
                                                                                                     model_name)
            tmp_spc_obj = SequencePairClassification(model_name,
                                                     num_labels=1,
                                                     max_length=64,
                                                     saved_model_path=tmp_saved_model_path)
            # console.print(f"----------------{model_name}--------------------", style='info')
            for data_type, valuable_data_path in valuable_data_paths.items():
                tmp_model_2_metrics = dict()
                tmp_task_id = tmp_progress_part.add_task("Evaluate-models", model_name=model_name,
                                                         data_type=data_type, total=99)
                valuable_df = pd.read_csv(valuable_data_path, sep='\t')
                tmp_batch_text_pairs = []
                tmp_label = []
                for _, tmp_row in valuable_df.iterrows():
                    tmp_batch_text_pairs.append((tmp_row.sentence1, tmp_row.sentence2))
                    tmp_label.append(tmp_row.label)
                trained_model = tmp_spc_obj.get_trained_model()
                tmp_start_time = time.perf_counter()
                tmp_pred = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)
                # print(f'Total cost time: {time.perf_counter() - tmp_start_time:}')
                tmp_threshold = 0.01
                best_threshold = tmp_threshold
                best_precision = 0.0
                best_recall = 0.0
                best_f1_score = 0.0
                best_accuracy = 0.0
                while tmp_threshold < 1.0:
                    tmp_y_true = np.asarray(tmp_label)
                    tmp_y_pred = np.squeeze((tmp_pred > tmp_threshold).astype(np.int64))
                    tn, fp, fn, tp = confusion_matrix(tmp_y_true, tmp_y_pred).ravel()
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    accuracy = (tp + tn) / (tn + fp + fn + tp)
                    if f1_score > best_f1_score:
                        best_threshold = tmp_threshold
                        best_precision = precision
                        best_recall = recall
                        best_f1_score = f1_score
                        best_accuracy = accuracy
                    tmp_progress_part.update(tmp_task_id, advance=1)
                    tmp_threshold += 0.01
                tmp_model_2_metrics['Threshold'] = round(best_threshold, 2)
                tmp_model_2_metrics['Precision'] = round(best_precision * 100, 2)
                tmp_model_2_metrics['Recall'] = round(best_recall * 100, 2)
                tmp_model_2_metrics['F1'] = round(best_f1_score * 100, 2)
                tmp_model_2_metrics['Accuracy'] = round(best_accuracy * 100, 2)
                data_2_model[data_type][model_name] = tmp_model_2_metrics
    console.log(f"[bold]{models_folder}[/bold] 模型集合评测结果")

    # 构建评测统计表格
    # print(data_2_model)
    for tmp_dataset_name, tmp_data_dict in data_2_model.items():
        construct_table(tmp_dataset_name, tmp_data_dict)
    console.log(f"Evaluating finished.")
    console.save_text(f'./{models_folder}模型集合评测结果.txt')
