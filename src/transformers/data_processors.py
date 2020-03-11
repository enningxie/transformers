# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/11
""" data processors and helpers """

import logging
import os

from .file_utils import is_tf_available, is_torch_available
from .data_utils import DataProcessor, InputExample, InputFeatures

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum examples length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available():
        is_tf_dataset = True

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    len_examples = len(examples)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing examples %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, max_length=max_length)
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
            "attention_mask"]

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )
    return features


class SequencePairClassificationProcessor(DataProcessor):
    """Processor for the Sequence Pair Classification data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]  # sentence1
            text_b = line[1]  # sentence2
            label = line[2]  # label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# todo 未校验
class SingleSentenceClassificationProcessor(DataProcessor):
    """ Generic processor for a single sentence classification data set."""

    def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SingleSentenceClassificationProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]

    @classmethod
    def create_from_csv(
            cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
    ):
        processor = cls(**kwargs)
        processor.add_examples_from_csv(
            file_name,
            split_name=split_name,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
            overwrite_labels=True,
            overwrite_examples=True,
        )
        return processor

    @classmethod
    def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
        processor = cls(**kwargs)
        processor.add_examples(texts_or_text_and_labels, labels=labels)
        return processor

    def add_examples_from_csv(
            self,
            file_name,
            split_name="",
            column_label=0,
            column_text=1,
            column_id=None,
            skip_first_row=False,
            overwrite_labels=False,
            overwrite_examples=False,
    ):
        lines = self._read_tsv(file_name)
        if skip_first_row:
            lines = lines[1:]
        texts = []
        labels = []
        ids = []
        for (i, line) in enumerate(lines):
            texts.append(line[column_text])
            labels.append(line[column_label])
            if column_id is not None:
                ids.append(line[column_id])
            else:
                guid = "%s-%s" % (split_name, i) if split_name else "%s" % i
                ids.append(guid)

        return self.add_examples(
            texts, labels, ids, overwrite_labels=overwrite_labels, overwrite_examples=overwrite_examples
        )

    def add_examples(
            self, texts_or_text_and_labels, labels=None, ids=None, overwrite_labels=False, overwrite_examples=False
    ):
        assert labels is None or len(texts_or_text_and_labels) == len(labels)
        assert ids is None or len(texts_or_text_and_labels) == len(ids)
        if ids is None:
            ids = [None] * len(texts_or_text_and_labels)
        if labels is None:
            labels = [None] * len(texts_or_text_and_labels)
        examples = []
        added_labels = set()
        for (text_or_text_and_label, label, guid) in zip(texts_or_text_and_labels, labels, ids):
            if isinstance(text_or_text_and_label, (tuple, list)) and label is None:
                text, label = text_or_text_and_label
            else:
                text = text_or_text_and_label
            added_labels.add(label)
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # Update labels
        if overwrite_labels:
            self.labels = list(added_labels)
        else:
            self.labels = list(set(self.labels).union(added_labels))

        return self.examples

    def get_features(
            self,
            tokenizer,
            max_length=None,
            pad_on_left=False,
            pad_token=0,
            mask_padding_with_zero=True,
            return_tensors=None,
    ):
        """
        Convert examples in a list of ``InputFeatures``

        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum examples length
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.labels)}

        all_input_ids = []
        for (ex_index, example) in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing examples %d", ex_index)

            input_ids = tokenizer.encode(
                example.text_a, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len),
            )
            all_input_ids.append(input_ids)

        batch_length = max(len(input_ids) for input_ids in all_input_ids)

        features = []
        for (ex_index, (input_ids, example)) in enumerate(zip(all_input_ids, self.examples)):
            if ex_index % 10000 == 0:
                logger.info("Writing examples %d/%d" % (ex_index, len(self.examples)))
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = batch_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
                len(input_ids), batch_length
            )
            assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
                len(attention_mask), batch_length
            )

            if self.mode == "classification":
                label = label_map[example.label]
            elif self.mode == "regression":
                label = float(example.label)
            else:
                raise ValueError(self.mode)

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, label=label))

        if return_tensors is None:
            return features
        elif return_tensors == "tf":
            if not is_tf_available():
                raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")
            import tensorflow as tf

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask}, ex.label)

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])}, tf.TensorShape([])),
            )
            return dataset
        elif return_tensors == "pt":
            if not is_torch_available():
                raise RuntimeError("return_tensors set to 'pt' but PyTorch can't be imported")
            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            if self.mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            elif self.mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")


processors = {
    "spc": SequencePairClassificationProcessor,
    "ssc": SingleSentenceClassificationProcessor
}

output_modes = {
    "spc": "classification",
    "ssc": "classification"
}
