# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/23
import os
import sys
models_path = os.path.join(os.getcwd(), '../')
sys.path.append(models_path)
from xz_transformers.tasks.about_sequence_pair_classification import SequencePairClassification
from xz_transformers.tasks.about_single_sequence_classification import SingleSequenceClassification
import logging


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

name2task = {
    'spc': SequencePairClassification,
    'ssc': SingleSequenceClassification
}


class TrainingPipeline:
    def __init__(self, ptm_short_names, num_labels, max_length, data_path, task_name):
        """
        多模型训练pipeline
        :param ptm_short_names: 待训练模型简称列表
        :param num_labels: 待训练模型目标分类数量
        :param max_length: 待训练模型最大长度
        :param data_path: 待训练数据路径
        :param task_name: 待训练任务名称
        """
        self.ptm_short_names = ptm_short_names
        self.num_labels = num_labels
        self.max_length = max_length
        self.data_path = data_path
        self.task_name = task_name

    def run(self, saved_path, epochs=10, batch_size=64):
        """
        :param saved_path: 训练模型保存路径
        :param epochs: 训练阶段迭代次数
        :param batch_size: 训练阶段batch size
        :return:
        """
        for tmp_index, tmp_short_name in enumerate(self.ptm_short_names, 1):
            logger.warning(f'--> {tmp_index}: {tmp_short_name} model start training.')
            tmp_task = name2task.get(self.task_name, None)
            if tmp_task is None:
                raise ValueError('Please specify correct task name.')
            tmp_saved_model_path = os.path.join(saved_path, tmp_short_name)
            tmp_task_obj = tmp_task(tmp_short_name, self.num_labels, self.max_length, tmp_saved_model_path)
            tmp_task_obj.train_op(self.data_path, epochs, batch_size)
            logger.warning(f'--> {tmp_index}: {tmp_short_name} model finished training.')


if __name__ == '__main__':
    tmp_ptm_short_names = ["chinese-rbt3", "chinese-rbtl3", "roberta_chinese_3L312_clue_tiny",
                           "roberta_chinese_3L768_clue_tiny", "roberta_chinese_clue_tiny", "roberta_chinese_pair_tiny",
                           "chinese_simbert_zhuiyi", "chinese-bert-wwm", "chinese-bert-wwm-ext",
                           "chinese-roberta-wwm-ext", "chinese-roberta-wwm-ext-large"]
    tmp_num_labels = 1
    tmp_max_length = 64
    tmp_data_path = '/Data/xen/Codes/transformers_xz/data/LCQMC'
    tmp_task_name = 'spc'
    tmp_saved_path = '/Data/xen/Codes/transformers_xz/saved_models/intent_detection'
    training_pipeline = TrainingPipeline(tmp_ptm_short_names, tmp_num_labels, tmp_max_length, tmp_data_path,
                                         tmp_task_name)
    training_pipeline.run(tmp_saved_path)
