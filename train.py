"""Run BERT on MRQA.

https://note.nkmk.me/en/python-break-nested-loops/

Script adapted from the span bert repo (Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved)
"""

from __future__ import absolute_import, division, print_function

import wandb
import argparse
import collections
import json
import logging
import math
import os
import random
import time
import gzip
import datetime
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertTokenizer
from transformers import AdamW
from model import BertForQuestionAnswering
from transformers import get_scheduler

from pytorch_pretrained_bert.tokenization import BasicTokenizer
from util_mrqa_official_eval import exact_match_score, f1_score, metric_max_over_ground_truths

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"


class MRQAExample(object):
    """
    A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


# new function to deal with .gz and .jsonl file
def get_data(input_file):
    if input_file.endswith('.gz'):
        with gzip.GzipFile(input_file, 'r') as reader:
            # skip header
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            input_data = [json.loads(line) for line in content]
    else:
        with open(input_file, 'r', encoding="utf-8") as reader:
            # lines = reader.readlines()
            # input_data = [json.loads(line) for line in lines]
            print(reader.readline())
            input_data = [json.loads(line) for line in reader]
    return input_data


def read_mrqa_examples(input_file, is_training, ignore=0, percentage=1):
    """Read a MRQA json file into a list of MRQAExample."""
    input_data = get_data(input_file)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    num_answers = 0
    num_to_ignore = int(ignore * len(input_data))
    num_to_load = int(percentage * len(input_data))
    if ignore != 0 and percentage != 1 and ignore + percentage == 1:
        num_to_load = max(num_to_load, len(input_data) - num_to_ignore)
    logger.info('Notes: # documents loaded = {}'.format(num_to_load - num_to_ignore))
    for entry in input_data[num_to_ignore:(num_to_ignore + num_to_load)]:
        paragraph_text = entry["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        for qa in entry["qas"]:
            qas_id = qa["qid"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            if is_training:
                answers = qa["detected_answers"]
                spans = sorted([span for spans in answers for span in spans['char_spans']])
                # take first span
                char_start, char_end = spans[0][0], spans[0][1]
                orig_answer_text = paragraph_text[char_start:char_end + 1]
                start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[
                    char_end]
                num_answers += sum([len(spans['char_spans']) for spans in answers])
            example = MRQAExample(qas_id=qas_id,
                                  question_text=question_text,
                                  doc_tokens=doc_tokens,
                                  orig_answer_text=orig_answer_text,
                                  start_position=start_position,
                                  end_position=end_position)
            examples.append(example)
    logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = -1
            tok_end_position = -1
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position,
             tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position,
                                                      tok_end_position, tokenizer,
                                                      example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" %
                            " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" %
                            " ".join(["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))

            features.append(
                InputFeatures(unique_id=unique_id,
                              example_index=example_index,
                              doc_span_index=doc_span_index,
                              tokens=tokens,
                              token_to_orig_map=token_to_orig_map,
                              token_is_max_context=token_is_max_context,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              start_position=start_position,
                              end_position=end_position))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def make_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length,
                     do_lower_case, verbose_logging):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(feature_index=feature_index,
                                          start_index=start_index,
                                          end_index=end_index,
                                          start_logit=result.start_logits[start_index],
                                          end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x: (x.start_logit + x.end_logit),
                                    reverse=True)
        _NbestPrediction = collections.namedtuple("NbestPrediction",
                                                  ["text", "start_logit", "end_logit"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(text=final_text,
                                 start_logit=pred.start_logit,
                                 end_logit=pred.end_logit))
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        assert len(nbest) >= 1
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)
        assert len(nbest_json) >= 1
        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json
    return all_predictions, all_nbest_json


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text,
                        tok_ns_text)
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_raw_scores(dataset, predictions):
    answers = {}
    for example in dataset:
        for qa in example['qas']:
            answers[qa['qid']] = qa['answers']
    exact_scores = {}
    f1_scores = {}
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            print('Missing prediction for %s' % qid)
            continue
        prediction = predictions[qid]
        exact_scores[qid] = metric_max_over_ground_truths(exact_match_score, prediction,
                                                          ground_truths)
        f1_scores[qid] = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def evaluate(args,
             model,
             device,
             eval_dataset,
             eval_dataloader,
             eval_examples,
             eval_features,
             verbose=True):
    all_results = []
    model.eval()
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model([input_ids, input_mask, segment_ids])
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(
                RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
    preds, nbest_preds = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging)
    exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
    result = make_eval_dict(exact_raw, f1_raw)
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result, preds, nbest_preds


def load_initialization(model, args):
    if os.path.exists(args.initialize_model_from_checkpoint + '/pytorch_model.bin'):
        ckpt = torch.load(args.initialize_model_from_checkpoint + '/pytorch_model.bin')
        model.load_state_dict(ckpt)
    else:
        ckpt = torch.load(args.initialize_model_from_checkpoint + '/saved_checkpoint')
        assert args.model == ckpt['args']['model'], args.model + ' vs ' + ckpt['args']['model']
        model.load_state_dict(ckpt['model_state_dict'])
    logger.info("***** Model Initialization *****")
    logger.info("Loaded the model state from a saved checkpoint {}".format(
        args.initialize_model_from_checkpoint))


def turn_off_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0


def tune_bias_only(m):
    for name, param in m.bert.named_parameters():
        if 'bias' in name or 'LayerNorm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def flip(scores, flip_prob, negative_reward):
    if flip_prob != 0:
        probs = torch.rand(scores.shape).to(scores.device)
        # true for values to be flipped
        mask = probs < flip_prob
        positive = scores == 1
        scores[mask & positive] = negative_reward
        scores[mask & ~positive] = 1
    return scores


def get_batch_rewards(start_probs, end_probs, start_positions, end_positions, device, args,
                      tokenizer, input_ids):
    bs = start_probs.shape[0]
    if args.argmax_simulation:
        start_samples = torch.argmax(start_probs, dim=1)
        end_samples = torch.argmax(end_probs, dim=1)
    else:
        start_samples = torch.multinomial(start_probs, 1).view(-1)
        end_samples = torch.multinomial(end_probs, 1).view(-1)
    log_prob = start_probs[torch.arange(bs), start_samples].log() + end_probs[torch.arange(bs),
                                                                              end_samples].log()

    # compute rewards
    def binary_reward():
        reward_mask = (start_samples == start_positions) & (end_samples == end_positions)
        rewards = torch.tensor([args.negative_reward] * bs).to(device)
        rewards[reward_mask] = 1
        return rewards

    rewards = eval(args.reward_fn)()
    rewards = flip(rewards, args.flip_prob, args.negative_reward)

    return start_samples, end_samples, log_prob, rewards


def collect_rewards_offline(model, train_batches, args, device, tokenizer, n_gpu):
    total_pos = 0
    total_neg = 0
    for i in range(len(train_batches)):
        batch = train_batches[i]
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)

        # sampling
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        with torch.no_grad():
            start_probs, end_probs = model(batch=batch[:3], return_prob=True)

        start_samples, end_samples, log_prob, rewards = get_batch_rewards(
            start_probs, end_probs, start_positions, end_positions, device, args, tokenizer,
            input_ids)
        train_batches[i] = [
            input_ids, input_mask, segment_ids, start_samples, end_samples, log_prob, rewards
        ]

        count_pos = torch.sum(rewards > 0).item()
        total_pos += count_pos
        total_neg += input_ids.shape[0] - count_pos
    return train_batches, total_pos, total_neg


def main(args):
    args.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    # set up random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # deal with gradient accumulation
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    # actual bs = bs // g i.e. 5 = 10 // 2
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    # parse dataset
    if args.dataset is not None:
        assert args.train_file is None
        assert args.dev_file is None
        if args.dataset == 'squad':
            args.train_file = 'data/SQuAD_train.jsonl'
            args.dev_file = 'data/SQuAD_dev.jsonl.gz'
        elif args.dataset == 'hotpot':
            args.train_file = 'data/HotpotQA-train.jsonl.gz'
            args.dev_file = 'data/HotpotQA-dev.jsonl.gz'
        elif args.dataset == 'nq':
            args.train_file = 'data/NaturalQuestionsShort-train.jsonl.gz'
            args.dev_file = 'data/NaturalQuestionsShort-dev.jsonl.gz'
        elif args.dataset == 'news':
            args.train_file = 'data/NewsQA-train.jsonl.gz'
            args.dev_file = 'data/NewsQA-dev.jsonl.gz'
        elif args.dataset == 'search':
            args.train_file = 'data/SearchQA-train.jsonl.gz'
            args.dev_file = 'data/SearchQA-dev.jsonl.gz'
        elif args.dataset == 'trivia':
            args.train_file = 'data/TriviaQA-train.jsonl.gz'
            args.dev_file = 'data/TriviaQA-dev.jsonl.gz'
        else:
            raise ValueError('Unknown dataset')

    # if args.dataset is not None and args.pretrainex is not None:
    #     assert args.initialize_model_from_checkpoint is None
    #     raise ValueError('What initialization to use?')

    # if args.pretrainon is not None:
    #     assert args.initialize_model_from_checkpoint is None
    #     raise ValueError('Which dataset pretrained on?')

    # argparse checkers
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.do_train:
        assert args.train_file is not None
    if args.eval_test:
        assert args.test_file is not None
    # only evaluate on the test set: need an initialization
    if args.eval_test and not args.do_train:
        assert args.initialize_model_from_checkpoint is not None

    if args.percentage_train_data + args.percentage_train_data_to_ignore > 1:
        raise ValueError(
            "Problematic combination of percentages on training: {} to train but {} to ignore".
            format(args.percentage_train_data, args.percentage_train_data_to_ignore))

    # set up logging files
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # set up the logging for this experiment
    args.output_dir += '/' + args.timestamp
    os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    # log args
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    if args.do_train and args.do_eval:
        # load dev dataset
        eval_dataset = get_data(input_file=args.dev_file)
        eval_examples = read_mrqa_examples(input_file=args.dev_file, is_training=False)
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_seq_length,
                                                     doc_stride=args.doc_stride,
                                                     max_query_length=args.max_query_length,
                                                     is_training=False)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        args.dev_num_orig_ex = len(eval_examples)
        args.dev_num_split_ex = len(eval_features)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:
        train_examples = read_mrqa_examples(input_file=args.train_file,
                                            is_training=True,
                                            ignore=args.percentage_train_data_to_ignore,
                                            percentage=args.percentage_train_data)
        train_features = convert_examples_to_features(examples=train_examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_seq_length,
                                                      doc_stride=args.doc_stride,
                                                      max_query_length=args.max_query_length,
                                                      is_training=True)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features],
                                           dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        args.train_num_orig_ex = len(train_examples)
        args.train_num_split_ex = len(train_features)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 3e-5, 2e-5, 1e-5]
        for lr in lrs:
            if args.initialize_model_from_checkpoint:
                model = BertForQuestionAnswering(model_type=args.model)
                load_initialization(model=model, args=args)
            else:
                model = BertForQuestionAnswering(model_type=args.model)

            if args.turn_off_dropout:
                turn_off_dropout(model)

            if args.tune_bias_only:
                tune_bias_only(model)

            model.to(device)

            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':
                0.01
            }, {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay':
                0.0
            }]
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
            lr_scheduler = get_scheduler(args.scheduler,
                                         optimizer=optimizer,
                                         num_warmup_steps=int(num_train_optimization_steps *
                                                              args.warmup_proportion),
                                         num_training_steps=num_train_optimization_steps)

            if args.setup == 'offline':
                train_batches, total_pos, total_neg = collect_rewards_offline(
                    model, train_batches, args, device, tokenizer, n_gpu)
                logger.info("Offline regret computation: {} positives {} negatives".format(
                    total_pos, total_neg))

            if args.wandb:
                wandb.init(
                    project='bandit-qa',
                    name=
                    f'{args.percentage_train_data}-{args.train_num_orig_ex}{args.dataset}_{args.algo}_{args.model}_{args.scheduler}={lr}_{args.initialize_model_from_checkpoint}+{args.argmax_simulation}_{args.output_dir}',
                    notes=args.notes,
                    config=vars(args))
                wandb.watch(model)

            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            start_time = time.time()
            simulation_log = None
            one_epoch_f1 = None
            dev_f1s = []
            steps = []
            total_pos, total_neg = 0, 0
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)

                    start_probs, end_probs = model(batch=batch[:3], return_prob=True)
                    bs = start_probs.shape[0]
                    if args.setup == 'online':
                        input_ids, _, _, start_positions, end_positions = batch
                        start_samples, end_samples, log_prob, rewards = get_batch_rewards(
                            start_probs, end_probs, start_positions, end_positions, device, args,
                            tokenizer, input_ids)
                        count_pos = torch.sum(rewards > 0).item()
                        total_pos += count_pos
                        total_neg += bs - count_pos
                    else:
                        input_ids, _, _, start_samples, end_samples, old_log_prob, old_rewards = batch
                        log_prob = start_probs[torch.arange(bs),
                                               start_samples].log() + end_probs[torch.arange(bs),
                                                                                end_samples].log()
                        ratios = torch.exp(log_prob - old_log_prob)
                        rewards = torch.clamp(ratios, 0, 1) * old_rewards
                        rewards = rewards.detach()

                    # compute values
                    if args.algo == 'Rwb':
                        values = torch.tensor([-0.05] * bs).to(device)
                        detached_advantages = rewards - values
                    elif args.algo == 'Rwmb':
                        detached_advantages = rewards - rewards.mean()
                    else:
                        detached_advantages = rewards
                    # compute probs
                    loss = (-log_prob * detached_advantages).mean() / 2

                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if args.wandb and (global_step + 1) % 25 == 0:
                        wandb.log(
                            {
                                '(Train) policy loss': loss.item(),
                                '(Train) reward': rewards.mean().item(),
                                '(Train) advantage': detached_advantages.mean().item(),
                            },
                            step=global_step)
                        if simulation_log is not None:
                            wandb.log(simulation_log, step=global_step)

                    if (step + 1) % eval_step == 0 or step + 1 == len(train_batches):
                        logger.info(
                            'Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps))

                        if args.wandb:
                            wandb.log(
                                {
                                    '(Train) loss': loss.item(),
                                    '(Train) total pos': total_pos,
                                    '(Train) total neg': total_neg
                                },
                                step=global_step)
                            if simulation_log is not None:
                                wandb.log(simulation_log, step=global_step)

                        save_model = False
                        if args.do_eval:
                            result, _, _ = \
                                evaluate(args, model, device, eval_dataset,
                                         eval_dataloader, eval_examples, eval_features)
                            model.train()
                            if args.wandb:
                                wandb.log(result, step=global_step)
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result[
                                'batch_size'] = args.train_batch_size * args.gradient_accumulation_steps
                            result['eval_step'] = eval_step
                            dev_f1s.append(round(result[args.eval_metric], 1))
                            steps.append(step)
                            result['dev_f1s'] = dev_f1s
                            result['steps'] = steps
                            result['total_pos'] = total_pos
                            result['total_neg'] = total_neg
                            if (best_result is None) or (result[args.eval_metric] >
                                                         best_result[args.eval_metric]):
                                best_result = result
                                # save model when getting new best result
                                save_model = True
                                logger.info(
                                    "!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                    (args.eval_metric, str(lr), epoch, result[args.eval_metric]))
                            elif best_result is not None:
                                save_model = True
                        else:
                            # case: no evaluation so just save the latest model
                            save_model = True
                        if save_model:
                            # NOTE changed
                            # save the config
                            model.bert.config.to_json_file(
                                os.path.join(args.output_dir, 'config.json'))
                            # save the model
                            torch.save(
                                {
                                    'global_step': global_step,
                                    'args': vars(args),
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.output_dir, 'saved_checkpoint'))
                            if best_result:
                                # i.e. best_result is not None
                                filename = EVAL_FILE
                                if len(lrs) != 1:
                                    filename = str(lr) + '_' + EVAL_FILE
                                with open(os.path.join(args.output_dir, filename), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))
                                    if epoch == 0:
                                        one_epoch_f1 = best_result['f1']
                                    writer.write("%s = %s\n" % ('one_epoch_f1', one_epoch_f1))
                        if args.save_checkpoint:
                            checkpoint = {
                                'global_step': global_step,
                                'args': vars(args),
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }
                            folder = args.output_dir + '/ckpt'
                            # create a folder if not existed
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            filename = folder + f'/{args.timestamp}_gstep={global_step}'
                            torch.save(checkpoint, filename)

    if args.eval_test:
        if args.wandb:
            wandb.init(
                project='pqa',
                entity='lil',
                name=
                f'{args.model}_{args.test_file}_{args.initialize_model_from_checkpoint}+{args.argmax_simulation}_{args.output_dir}',
                tags=['eval'],
                notes=args.notes,
                config=vars(args))

        eval_dataset = get_data(args.test_file)
        eval_examples = read_mrqa_examples(input_file=args.test_file, is_training=False)
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_seq_length,
                                                     doc_stride=args.doc_stride,
                                                     max_query_length=args.max_query_length,
                                                     is_training=False)
        logger.info("***** Test *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

        # NOTE change: only evaluate on the test set
        if not args.do_train:
            model = BertForQuestionAnswering(model_type=args.model)
            assert args.initialize_model_from_checkpoint is not None
            load_initialization(model=model, args=args)
            model.to(device)
        result, preds, nbest_preds = evaluate(args, model, device, eval_dataset, eval_dataloader,
                                              eval_examples, eval_features)
        with open(os.path.join(args.output_dir, PRED_FILE), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")
        with open(os.path.join(args.output_dir, TEST_FILE), "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

        if args.wandb:
            wandb.log(result, step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch",
                        default=10,
                        type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride",
                        default=128,
                        type=int,
                        help="When splitting up a long document into chunks, "
                        "how much stride to take between chunks.")
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test",
                        action='store_true',
                        help='Wehther to run eval on the test set.')
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate",
                        default=None,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1', type=str)
    parser.add_argument("--train_mode",
                        type=str,
                        default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
        "of training.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.")
    parser.add_argument("--max_answer_length",
                        default=30,
                        type=int,
                        help="The maximum length of an answer that can be generated. "
                        "This is needed because the start "
                        "and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal MRQA evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # below are customized arguments
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')
    parser.add_argument('--notes', default='', help='Notes for this experiment: wandb logging')
    parser.add_argument(
        '--save_checkpoint',
        action='store_true',
        help=
        'Whether to save different checkpoints during training: recommend not to use this argument for space saving'
    )
    parser.add_argument('--percentage_train_data',
                        type=float,
                        default=1,
                        help='Percetage of training data to load: for debugging purpose')
    parser.add_argument(
        '--percentage_train_data_to_ignore',
        type=float,
        default=0,
        help=
        'Percetage of training data to ignore first: for experiments where to exlucde the some initial data used for pre-training'
    )
    parser.add_argument(
        '--argmax_simulation',
        action='store_true',
        help='Whether to take argmax of the results for simulation: stick with argmax in this work')
    parser.add_argument(
        '--reward_fn',
        default='binary_reward',
        type=str,
        choices=['binary_reward'],
        help='the type of reward function used during training: stick with binary in this work')
    parser.add_argument('--initialize_model_from_checkpoint',
                        default=None,
                        help='Relative filepath to a saved checkpoint as model initialization.')
    parser.add_argument(
        '--flip_prob',
        default=0.0,
        type=float,
        help='Parameter for the perturbation function: x probability to flip the rewards')
    parser.add_argument('--scheduler', default='linear', type=str, help='Learning rate scheduler.')
    parser.add_argument(
        '--transfer',
        action='store_true',
        help='Domain adaptation or not. Not used in the code, only for wandb logging purpose.')
    parser.add_argument('--turn_off_dropout',
                        action='store_true',
                        help='Should turn off dropout for simulation experiments')
    parser.add_argument(
        '--tune_bias_only',
        action='store_true',
        help='Only tune the bias and layernorm in bert, as well as the classifier on top')
    parser.add_argument('--algo',
                        default='R',
                        choices=['R'],
                        help='training algorithm: stick with R in this work.')
    parser.add_argument('--negative_reward',
                        default=-0.1,
                        type=float,
                        help='value for negative update')
    parser.add_argument('--setup',
                        default='online',
                        type=str,
                        choices=['online', 'offline'],
                        help='online or offline setup')
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        choices=['squad', 'hotpot', 'nq', 'trivia', 'search', 'news'])
    parser.add_argument("--pretrainon",
                        default=None,
                        type=str,
                        choices=['squad', 'hotpot', 'nq', 'trivia', 'search', 'news'])
    parser.add_argument("--pretrainex", default=None, type=int, choices=[64, 256, 1024])
    args = parser.parse_args()
    main(args)
