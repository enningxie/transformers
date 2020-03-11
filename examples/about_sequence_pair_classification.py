# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/11
"""使用LCQMC数据集训练语义匹配模型"""
import os
import tensorflow as tf
from src.transformers.file_utils import ROOT_PATH
from src.transformers.configuration_bert import BertConfig
from src.transformers.tokenization_bert import BertTokenizer
from src.transformers.modeling_tf_bert import TFBertForSequenceClassification
from src.transformers.data_processors import SequencePairClassificationProcessor, convert_examples_to_features

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class AboutSequencePairClassification:
    """
    句子对分类
    """
    def __init__(self, pretrained_model_name, num_labels, batch_size, epochs, max_length):
        self.data_processor = SequencePairClassificationProcessor()
        self.config = BertConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(pretrained_model_name, config=self.config,
                                                                     from_pt=True)

        # training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length

        self.task = 'spc'

    def calculate_steps(self, examples):
        """
        计算steps
        :param examples:
        :return:
        """
        tmp_length = len(examples)
        if tmp_length % self.batch_size == 0:
            tmp_steps = tmp_length // self.batch_size
        else:
            tmp_steps = tmp_length // self.batch_size + 1
        return tmp_steps

    def generate_tf_dataset(self, data_path):
        """
        生成TFDataSet，用于训练模型前的准备
        :param data_path:
        :return:
        """
        # process raw data
        train_examples = self.data_processor.get_train_examples(data_path)
        valid_examples = self.data_processor.get_dev_examples(data_path)

        # calculate steps
        train_steps = self.calculate_steps(train_examples)
        valid_steps = self.calculate_steps(valid_examples)

        # convert examples to tf_dataset
        train_dataset = convert_examples_to_features(train_examples, self.tokenizer, max_length=self.max_length,
                                                     task=self.task)
        valid_dataset = convert_examples_to_features(valid_examples, self.tokenizer, max_length=self.max_length,
                                                     task=self.task)

        # preprocess tf_dataset
        train_dataset = train_dataset.batch(self.batch_size)
        valid_dataset = valid_dataset.batch(self.batch_size)

        return (train_dataset, train_steps), (valid_dataset, valid_steps)

    def train_op(self, data_path):
        """
        模型训练
        :param data_path: 原始数据保存路径
        :return:
        """
        (train_dataset, train_steps), (valid_dataset, valid_steps) = self.generate_tf_dataset(data_path)
        # prepare training: compile tf.keras model with optimizer, loss and learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        self.model.summary()
        # Train and evaluate using tf.keras.Model.fit()
        history = self.model.fit(train_dataset.repeat(), epochs=self.epochs, steps_per_epoch=train_steps,
                                 validation_data=valid_dataset.repeat(), validation_steps=valid_steps,
                                 callbacks=[early_stop])
        print(f"History: {history}")
        self.model.save_pretrained("saved_models/spc_0")

    # todo evaluate_op


if __name__ == '__main__':
    raw_data_path = os.path.join(ROOT_PATH, 'data/LCQMC')
    tmp_spc_obj = AboutSequencePairClassification('chinese-rbt3',
                                                  num_labels=2,
                                                  batch_size=32,
                                                  epochs=5,
                                                  max_length=64)
    tmp_spc_obj.train_op(raw_data_path)
