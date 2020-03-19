# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/11
"""使用LCQMC数据集训练语义匹配模型"""
import os
import tensorflow as tf
import numpy as np
from src.transformers.file_utils import ROOT_PATH, CONFIG_NAME
from src.transformers.configuration_bert import BertConfig
from src.transformers.tokenization_bert import BertTokenizer
from src.transformers.modeling_tf_bert import TFBertForSequenceClassification
from src.transformers.data_processors import SequencePairClassificationProcessor, convert_examples_to_features
from src.transformers.modeling_tf_utils import calculate_steps

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class AboutSequencePairClassification:
    """
    句子对分类，相似不相似
    """

    def __init__(self, pretrained_model_name, num_labels, max_length):
        """
        :param pretrained_model_name: 预训练模型权重保存简称或路径
        :param num_labels: 类别个数，1默认最后使用sigmoid，2则使用softmax
        :param max_length: 训练阶段sequence最大长度
        """
        self.data_processor = SequencePairClassificationProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        # training parameters
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if num_labels == 1:
            self.mode = 'binary'
        else:
            self.mode = 'category'

        self.task = 'spc'

    def generate_tf_dataset(self, data_path, batch_size):
        """
        生成TFDataSet，用于训练模型前的准备
        :param data_path: 数据保存路径
        :param batch_size: 训练阶段 batch_size大小
        :return:
        """
        # process raw data
        train_examples = self.data_processor.get_train_examples(data_path)
        valid_examples = self.data_processor.get_dev_examples(data_path)

        # calculate steps
        train_steps = calculate_steps(len(train_examples), batch_size)
        valid_steps = calculate_steps(len(valid_examples), batch_size)

        # convert examples to tf_dataset
        train_dataset = convert_examples_to_features(train_examples, self.tokenizer, max_length=self.max_length,
                                                     task=self.task, return_tensors='tf')
        valid_dataset = convert_examples_to_features(valid_examples, self.tokenizer, max_length=self.max_length,
                                                     task=self.task, return_tensors='tf')

        # preprocess tf_dataset
        train_dataset = train_dataset.batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)

        return (train_dataset, train_steps), (valid_dataset, valid_steps)

    @staticmethod
    def get_trained_model(trained_model_path):
        """
        加载训练好的模型
        :param trained_model_path: 训练好的模型路径
        :return:
        """
        trained_config = BertConfig.from_pretrained(os.path.join(trained_model_path, CONFIG_NAME))
        trained_model = TFBertForSequenceClassification.from_pretrained(trained_model_path, config=trained_config)
        return trained_model

    def get_compiled_model(self, model):
        """
        返回编译后的模型
        :param model: 编译前的模型
        :return:
        """
        # prepare training: compile tf.keras model with optimizer, loss and learning rate schedule
        if self.mode == 'binary':
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metric = tf.keras.metrics.BinaryAccuracy('accuracy')
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        model.summary()
        return model

    def train_op(self, data_path, epochs, batch_size, saved_model_path):
        """
        模型训练
        :param data_path: 原始数据保存路径
        :param epochs: 训练阶段迭代次数
        :param batch_size: 训练阶段batch_size
        :param saved_model_path: 训练结束后模型保存路径
        :return:
        """
        config = BertConfig.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        model = TFBertForSequenceClassification.from_pretrained(self.pretrained_model_name, config=config)
        (train_dataset, train_steps), (valid_dataset, valid_steps) = self.generate_tf_dataset(data_path, batch_size)
        model = self.get_compiled_model(model)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
        # Train and evaluate using tf.keras.Model.fit()
        history_train_step = model.fit(train_dataset.repeat(), epochs=epochs, steps_per_epoch=train_steps,
                                       validation_data=valid_dataset.repeat(), validation_steps=valid_steps,
                                       callbacks=[early_stop])
        print(f"Train step history: {history_train_step.history}")
        model.save_pretrained(saved_model_path)

    def evaluate_op(self, trained_model_path, test_data_path, batch_size=64):
        """
        模型evaluation step
        :param trained_model_path: 训练好的模型保存路径
        :param test_data_path: 测试数据保存路径
        :param batch_size: evaluation阶段的batch_size
        :return:
        """
        # 模型加载
        trained_model = self.get_trained_model(trained_model_path)
        trained_model = self.get_compiled_model(trained_model)
        # 原始数据转tf-dataset
        test_examples = self.data_processor.get_test_examples(test_data_path)
        test_steps = calculate_steps(len(test_examples), batch_size)
        test_dataset = convert_examples_to_features(test_examples, self.tokenizer, max_length=self.max_length,
                                                    task=self.task, return_tensors='tf')
        test_dataset = test_dataset.batch(batch_size)
        # model evaluation step
        evaluation_loss, evaluation_acc = trained_model.evaluate(test_dataset, steps=test_steps)
        print(f"Evaluate step loss: {evaluation_loss}")
        print(f"Evaluate step accuracy: {evaluation_acc}")

    def predict_op(self, trained_model_path, batch_text_pairs):
        """
        模型预测阶段
        :param trained_model_path: 训练好了的模型保存的路径
        :param batch_text_pairs: [['想了解下您会想看哪款车型', '是想请问下您当时买的是哪款车呢'], ['今天天气很差', '今天天气很棒']]
        :return:
        """
        trained_model = self.get_trained_model(trained_model_path)
        inputs = self.tokenizer.batch_encode_plus(batch_text_pairs, max_length=self.max_length, return_tensors="tf",
                                                  pad_to_max_length=True)
        tmp_pred = trained_model.predict(inputs)
        if self.mode == 'binary':
            tmp_result = tf.nn.sigmoid(tmp_pred)
            return np.squeeze(tmp_result.numpy(), axis=-1)
        else:
            tmp_result = tf.nn.softmax(tmp_pred, axis=-1)
            return tmp_result.numpy()


if __name__ == '__main__':
    raw_data_path = os.path.join(ROOT_PATH, 'data/LCQMC')
    tmp_spc_obj = AboutSequencePairClassification('chinese-rbt3',
                                                  num_labels=1,
                                                  max_length=64)
    # #### training step ####
    # tmp_spc_obj.train_op(raw_data_path, epochs=5, batch_size=64, saved_model_path='examples/saved_models/spc_2')

    tmp_trained_model_path = os.path.join(ROOT_PATH, 'examples/saved_models/spc_1')

    # #### evaluate step ####
    # tmp_spc_obj.evaluate_op(tmp_trained_model_path, raw_data_path)

    #### predict step ####
    tmp_batch_text_pairs = [('想了解下您会想看哪款车型', '是想请问下您当时买的是哪款车呢'), ('今天天气很差', '今天天气很棒'), ('今天天气不错', '今天天气很棒')]
    tmp_pred = tmp_spc_obj.predict_op(tmp_trained_model_path, tmp_batch_text_pairs)
    for tmp_pair, tmp_score in zip(tmp_batch_text_pairs, tmp_pred):
        print(f'{tmp_pair[0]} & {tmp_pair[1]} --> {tmp_score}')
