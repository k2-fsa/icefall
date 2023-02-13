from __future__ import unicode_literals
import logging
from typing import Any, Dict, List, Tuple, Union
import sys
import pandas as pd
import jiwer

# -*- coding: utf-8 -*-

def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]



def get_unicode_code(text):
    result = ''.join( char if ord(char) < 128 else '\\u'+format(ord(char), 'x') for char in text )
    return result


def _measure_cer(
        reference : str, transcription : str
) -> Tuple[int, int, int, int]:
    """
    소스 단어를 대상 단아로 변환하는 데 필요한 편집 작업(삭제, 삽입, 바꾸기)의 수를 확인합니다.
    hints 횟수는 소스 딘아의 전체 길이에서 삭제 및 대체 횟수를 빼서 제공할 수 있습니다.

    :param transcription: 대상 단어로 변환할 소스 문자열
    :param reference: 소스 단어
    :return: a tuple of #hits, #substitutions, #deletions, #insertions
    """

    ref, hyp = [], []

    ref.append(reference)
    hyp.append(transcription)

    #print("? : ", ref)

    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    sen_err = 0

    for n in range(len(ref)):
        # update CER statistics
        _, (s, i, d) = levenshtein(hyp[n], ref[n])
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(ref[n])

        # update SER statistics
        if s + i + d > 0:
            sen_err += 1



    '''
    print("reference : ",reference)
    print("cer S : ", cer_s)
    print("cer I : ", cer_i)
    print("cer D : ", cer_d)
    print("cer_n : ", cer_n)


    if cer_n > 0:
        print('CER: %g%%, SER: %g%%' % (
            (100.0 * (cer_s + cer_i + cer_d)) / cer_n,
            (100.0 * sen_err) / len(ref)))
    '''
    substitutions = cer_s
    deletions = cer_d
    insertions = cer_i
    hits = len(reference) - (substitutions + deletions) #correct characters

    return hits, substitutions, deletions, insertions

def _measure_wer(
        reference : str, transcription : str
) -> Tuple[int, int, int, int]:
    """
    소스 문자열을 대상 문자열로 변환하는 데 필요한 편집 작업(삭제, 삽입, 바꾸기)의 수를 확인합니다.
    hints 횟수는 소스 문자열의 전체 길이에서 삭제 및 대체 횟수를 빼서 제공할 수 있습니다.

    :param transcription: 대상 단어
    :param reference: 소스 단어
    :return: a tuple of #hits, #substitutions, #deletions, #insertions
    """

    ref, hyp = [], []

    ref.append(reference)
    hyp.append(transcription)

    #print("? : ", ref)

    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    sen_err = 0

    for n in range(len(ref)):
        # update WER statistics
        _, (s, i, d) = levenshtein(hyp[n].split(), ref[n].split())
        wer_s += s
        wer_i += i
        wer_d += d
        wer_n += len(ref[n].split())
        # update SER statistics
        if s + i + d > 0:
            sen_err += 1



    #print("reference : ",reference)
    #print("reference cnt : ", reference.split())
    #print("wer S : ", wer_s)
    #print("wer I : ", wer_i)
    #print("wer D : ", wer_d)
    #print("wer_n : ", wer_n)


    if wer_n > 0:
        print('WER: %g%%, SER: %g%%' % (
            (100.0 * (wer_s + wer_i + wer_d)) / wer_n,
            (100.0 * sen_err) / len(ref)))

    substitutions = wer_s
    deletions = wer_d
    insertions = wer_i
    hits = len(reference.split()) - (substitutions + deletions) #correct words between refs and trans

    return hits, substitutions, deletions, insertions




def _measure_er(
        reference : str, transcription : str
) -> Tuple[int, int]:
    """
    TBD
    :param transcription: 대상 문자열로 변환할 소스 문자열
    :param reference:
    :return: a tuple of #
    """
    TBD1 =""
    TBD2 =""
    return TBD1, TBD2


def get_cer(reference, transcription, rm_punctuation = True
            ) -> Tuple[int, int, int, int]:

    # 문자 오류율(CER)은 자동 음성 인식 시스템의 성능에 대한 일반적인 메트릭입니다.
    # CER은 WER(단어 오류율)과 유사하지만 단어 대신 문자에 대해 작동합니다.
    # 이 코드에서는 문제는 사람들이 띄어쓰기를 지키지 않고 작성한 텍스트를 컴퓨터가 정확하게 인식하는 것이 매우 어렵기 때문에 인식에러에서 생략합니다.
    # CER의 출력은 특히 삽입 수가 많은 경우 항상 0과 1 사이의 숫자가 아닙니다. 이 값은 종종 잘못 예측된 문자의 백분율과 연관됩니다. 값이 낮을수록 좋습니다.
    # CER이 0인 ASR 시스템의 성능은 완벽한 점수입니다.

    # CER = (S + D + I) / N = (S + D + I) / (S + D + C)
    # S is the number of the substitutions,
    # D is the number of the deletions,
    # I is the number of the insertions,
    # C is the number of the correct characters,
    # N is the number of the characters in the reference (N=S+D+C).

    refs = jiwer.RemoveWhiteSpace(replace_by_space=False)(reference)
    trans = jiwer.RemoveWhiteSpace(replace_by_space=False)(transcription)

    if rm_punctuation == True:
        refs = jiwer.RemovePunctuation()(refs)
        trans = jiwer.RemovePunctuation()(trans)
    else:
        refs = reference
        trans = transcription

    #print("refs : ", refs)

    [hits ,cer_s, cer_d, cer_i] = _measure_cer(refs, trans)

    substitutions = cer_s
    deletions = cer_d
    insertions = cer_i
    #print("tmp hits : ", hits)
    incorrect = substitutions + deletions + insertions
    total = substitutions + deletions + hits + insertions

    cer = incorrect / total
    return cer, substitutions, deletions, insertions


def get_wer(reference, transcription, rm_punctuation = True
            )-> Tuple[int, int, int, int]:

    # WER = (S + D + I) / N = (S + D + I) / (S + D + C)
    # S is the number of the substitutions,
    # D is the number of the deletions,
    # I is the number of the insertions,
    # C is the number of the correct words,
    # N is the number of the words in the reference (N=S+D+C).
    if rm_punctuation == True:
        refs = jiwer.RemovePunctuation()(reference)
        trans = jiwer.RemovePunctuation()(transcription)
    else:
        refs = reference
        trans = transcription
    [hits, wer_s, wer_d, wer_i] = _measure_wer(refs, trans)

    substitutions = wer_s
    deletions = wer_d
    insertions = wer_i
    #print("tmp hits : ", hits)
    incorrect = substitutions + deletions + insertions
    total = substitutions + deletions + hits + insertions

    wer = incorrect / total
    return wer, substitutions, deletions, insertions
