# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/18
"""Single sequence classification"""
import os
import logging
import pandas as pd
import tensorflow as tf
from xz_transformers.file_utils import ROOT_PATH, CONFIG_NAME, ID2LABEL_NAME
from xz_transformers.configuration_bert import BertConfig
from xz_transformers.tokenization_bert import BertTokenizer
from xz_transformers.modeling_tf_bert import TFBertForSequenceClassification
from xz_transformers.data_processors import SingleSentenceClassificationProcessor, convert_examples_to_features
from xz_transformers.modeling_tf_utils import calculate_steps, load_serialized_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleSequenceClassification:
    """
    单句表述分类
    """

    def __init__(self, pretrained_model_name, num_labels, max_length, saved_model_path):
        """
        :param pretrained_model_name: 预训练模型权重简称/路径
        :param num_labels: 类别个数
        :param max_length: sequence最大长度
        :param saved_model_path: 模型保存路径啦
        """
        self.data_processor = SingleSentenceClassificationProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.num_labels = num_labels
        self.pretrained_model_name = pretrained_model_name
        self.saved_model_path = saved_model_path

        # training parameters
        self.max_length = max_length

        self.task = 'ssc'

    def generate_tf_dataset(self, data_path, batch_size):
        """
        生成TFDataSet，用于训练模型前的准备
        :param data_path: 训练数据保存路径
        :param batch_size: batch_size
        :return:
        """
        # process raw data
        train_examples = self.data_processor.get_train_examples(data_path)
        train_label_list = self.data_processor.get_labels()
        valid_examples = self.data_processor.get_dev_examples(data_path)

        # calculate steps
        train_steps = calculate_steps(len(train_examples), batch_size)
        valid_steps = calculate_steps(len(valid_examples), batch_size)

        # convert tasks to tf_dataset
        train_dataset = convert_examples_to_features(train_examples, self.tokenizer, max_length=self.max_length,
                                                     label_list=train_label_list, task=self.task, return_tensors='tf',
                                                     save_id2label_path=os.path.join(self.saved_model_path,
                                                                                     'id2label.pkl'))
        valid_dataset = convert_examples_to_features(valid_examples, self.tokenizer, max_length=self.max_length,
                                                     label_list=train_label_list, task=self.task, return_tensors='tf')

        # preprocess tf_dataset
        train_dataset = train_dataset.batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)

        return (train_dataset, train_steps), (valid_dataset, valid_steps)

    def get_trained_model(self):
        """
        加载训练好的模型
        :param trained_model_path: 训练好的模型保存路径
        :return:
        """
        trained_config = BertConfig.from_pretrained(os.path.join(self.saved_model_path, CONFIG_NAME))
        trained_model = TFBertForSequenceClassification.from_pretrained(self.saved_model_path, config=trained_config)
        return trained_model

    @staticmethod
    def get_compiled_model(model):
        """
        返回编译后的模型
        :param model: 编译前模型
        :return:
        """
        # prepare training: compile tf.keras model with optimizer, loss and learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        model.summary()
        return model

    def train_op(self, data_path, epochs, batch_size):
        """
        模型训练
        :param data_path: 原始数据保存路径
        :param epochs: 训练迭代次数
        :param batch_size: 训练阶段batch_size
        :return:
        """
        config = BertConfig.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        model = TFBertForSequenceClassification.from_pretrained(self.pretrained_model_name, config=config)
        model = self.get_compiled_model(model)
        (train_dataset, train_steps), (valid_dataset, valid_steps) = self.generate_tf_dataset(data_path, batch_size)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

        # Train and evaluate using tf.keras.Model.fit()
        history = model.fit(train_dataset.repeat(), epochs=epochs, steps_per_epoch=train_steps,
                            validation_data=valid_dataset.repeat(), validation_steps=valid_steps,
                            callbacks=[early_stop])
        print(f"History: {history.history}")
        model.save_pretrained(self.saved_model_path)
        return model

    def evaluate_op(self, test_data_path, batch_size=64):
        """
        模型 evaluation step
        :param test_data_path: 测试集保存路径
        :param batch_size: evaluation阶段batch_size
        :return:
        """
        # 模型加载
        trained_model = self.get_trained_model()
        trained_model = self.get_compiled_model(trained_model)
        # 原始数据转tf-dataset
        test_examples = self.data_processor.get_test_examples(test_data_path)
        test_label_list = self.data_processor.get_labels(test_data_path)
        print(test_label_list)
        test_steps = calculate_steps(len(test_examples), batch_size)
        test_dataset = convert_examples_to_features(test_examples, self.tokenizer, max_length=self.max_length,
                                                    label_list=test_label_list, task=self.task, return_tensors='tf')
        test_dataset = test_dataset.batch(batch_size)
        # model evaluation step
        evaluation_loss, evaluation_acc = trained_model.evaluate(test_dataset, steps=test_steps)
        print(f"Evaluate step loss: {evaluation_loss}")
        print(f"Evaluate step accuracy: {evaluation_acc}")

    def predict_op(self, trained_model, batch_text):
        """
        :param trained_model: 加载好的训练完毕的模型
        :param batch_text: [['想了解下您会想看哪款车型', '是想请问下您当时买的是哪款车呢'], ['今天天气很差', '今天天气很棒']]
        :return:
        """
        inputs = self.tokenizer.batch_encode_plus(batch_text, max_length=self.max_length, return_tensors="tf",
                                                  pad_to_max_length=True)
        tmp_pred = tf.nn.softmax(trained_model.predict(inputs), axis=-1)
        tmp_pred_max = tf.reduce_max(tmp_pred, axis=-1).numpy()
        tmp_pred_label = tf.argmax(tmp_pred, axis=-1).numpy()
        tmp_id2label_path = os.path.join(self.saved_model_path, ID2LABEL_NAME)
        id2label = load_serialized_data(tmp_id2label_path)
        pred_result = []
        for tmp_index, (tmp_id, tmp_pred_value) in enumerate(zip(tmp_pred_label, tmp_pred_max)):
            pred_result.append((batch_text[tmp_index], id2label[tmp_id], tmp_pred_value))
        return pred_result


if __name__ == '__main__':
    raw_data_path = os.path.join(ROOT_PATH, 'data/souche_salesman2')
    tmp_saved_model_path = os.path.join(ROOT_PATH, 'tasks/saved_models/ssc_4')
    tmp_spc_obj = SingleSequenceClassification("chinese-rbt3",
                                               num_labels=25,
                                               max_length=32,
                                               saved_model_path=tmp_saved_model_path)
    #### training step ####
    tmp_spc_obj.train_op(raw_data_path, epochs=5, batch_size=64)


    # #### evaluate step ####
    # tmp_spc_obj.evaluate_op(tmp_trained_model_path, raw_data_path)

    # #### predict step ####
    # # tmp_batch_text = ['有空可以到店里面先试驾一下啊', '最近没时间']
    # trained_model = tmp_spc_obj.get_trained_model()
    # tmp_batch_text = pd.read_csv(os.path.join(raw_data_path, "dev.tsv"), header=0, sep='\t',
    #                              dtype={'label': str}).sentence.tolist()
    # pred_result = tmp_spc_obj.predict_op(trained_model, tmp_batch_text)
    #
    # with open(os.path.join(raw_data_path, 'test_result.txt'), 'w') as f:
    #     for tmp_text, tmp_label, tmp_score in pred_result:
    #         print(f'{tmp_text} --> {tmp_label} --> {tmp_score}')
    #         f.write(f'{tmp_label}' + '\t' + f'{tmp_text}' + '\t' + f'{tmp_score}' + '\n')
