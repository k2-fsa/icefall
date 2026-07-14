#!/usr/bin/env python3
# Copyright      2025  Yifan Yang
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
from typing import Dict, List


def match_case(word, replacement):
    if word.isupper():
        return replacement.upper()
    elif word[0].isupper():
        return replacement.title()
    else:
        return replacement.lower()


GENDER = ["female", "male"]
GENDER_PAIRS = [
    ("female", "male"),
    ("woman", "man"),
    ("girl", "boy"),
    ("feminine", "masculine"),
    ("she", "he"),
    ("her", "his"),
    ("herself", "himself"),
]


def perturb_gender_in_text(text: str, gender: str) -> dict:
    replaced_flag = False

    # 1. 确定替换方向
    # 如果当前标签是 female，我们要把 text 里的 female 词汇替换成 male (构造负例)
    # GENDER_PAIRS 的结构是 (female_word, male_word)
    # source_idx = 0 (female), target_idx = 1 (male)
    if gender == "female":
        source_idx, target_idx = 0, 1
    elif gender == "male":
        source_idx, target_idx = 1, 0
    else:
        raise ValueError(f"Unknown gender: {gender}")

    # 2. 依次遍历关键词进行匹配
    for pair in GENDER_PAIRS:
        src_word = pair[source_idx]
        tgt_word = pair[target_idx]

        # 构造正则表达式
        # Group 1: ("[^"]*")  -> 匹配双引号括起来的任意内容 (保护区)
        # Group 2: (\bkeyword\b) -> 匹配目标关键词，\b 确保词边界 (匹配区)
        # re.IGNORECASE -> 忽略大小写
        pattern = r'("[^"]*")|\b(' + re.escape(src_word) + r")\b"

        def replace_callback(match):
            nonlocal replaced_flag

            # 如果匹配到了 Group 1 (双引号内容)，原样返回，不动它
            if match.group(1):
                return match.group(1)

            # 如果匹配到了 Group 2 (关键词)，进行替换
            if match.group(2):
                original_word = match.group(2)
                replaced_flag = True
                return match_case(original_word, tgt_word)

        # 进行替换
        text = re.sub(pattern, replace_callback, text, flags=re.IGNORECASE)

    return {"flag": replaced_flag, "text": text}


SPEAKING_RATE = ["fast speed", "slow speed"]  # "measured speed"
SPEAKING_RATE_PAIRS = [
    ("fast-paced", "slow-paced"),
    ("fast", "slow"),
    ("quick", "slow"),
    ("rapid", "slow"),
    ("rushed", "deliberate"),
    ("hurried", "unhurried"),
    ("quickly", "slowly"),
    ("rapidly", "slowly"),
    ("fast", "measured"),
]


def perturb_speaking_rate_in_text(text: str, speaking_rate: str) -> dict:
    replaced_flag = False

    # 1. 确定替换方向
    # source_idx: 我们要在文本中搜索的词
    # target_idx: 我们要替换成的词
    if speaking_rate == "fast speed":
        source_idx, target_idx = 0, 1
    elif speaking_rate == "slow speed":
        source_idx, target_idx = 1, 0
    else:  # 如果标签不合法或者是 measured (不考虑)，直接返回
        return {"flag": False, "text": text}

    # 2. 依次遍历关键词进行匹配
    for pair in SPEAKING_RATE_PAIRS:
        src_word = pair[source_idx]
        tgt_word = pair[target_idx]

        # 构造正则表达式
        # Group 1: ("[^"]*")  -> 匹配双引号括起来的任意内容 (保护区)
        # Group 2: (\bkeyword\b) -> 匹配目标关键词，\b 确保词边界 (匹配区)
        # re.IGNORECASE -> 忽略大小写
        # re.escape(src_word) 用于处理像 "fast-paced" 中间的连字符，防止被正则误读
        pattern = r'("[^"]*")|\b(' + re.escape(src_word) + r")\b"

        def replace_callback(match):
            nonlocal replaced_flag

            # 如果匹配到了 Group 1 (双引号内容)，原样返回，不动它
            if match.group(1):
                return match.group(1)

            # 如果匹配到了 Group 2 (关键词)，进行替换
            if match.group(2):
                original_word = match.group(2)
                replaced_flag = True
                return match_case(original_word, tgt_word)

        # 进行替换
        text = re.sub(pattern, replace_callback, text, flags=re.IGNORECASE)

    return {"flag": replaced_flag, "text": text}


PITCH = ["high-pitched", "low-pitched"]  # "medium-pitched"
PITCH_PAIRS = [
    ("high-pitched", "low-pitched"),
    ("higher", "lower"),
    ("high", "low"),
    ("raising", "lowering"),
    ("rises", "falls"),
    ("rising", "falling"),
    ("raised", "lowered"),
    ("upward", "downward"),
]


def perturb_pitch_in_text(text: str, pitch: str) -> dict:
    replaced_flag = False

    # 1. 确定替换方向
    if pitch == "high-pitched":
        source_idx, target_idx = 0, 1
    elif pitch == "low-pitched":
        source_idx, target_idx = 1, 0
    else:  # 如果标签不合法或者是 medium-pitched (不考虑)，直接返回
        return {"flag": False, "text": text}

    # 2. 依次遍历关键词进行匹配
    for pair in PITCH_PAIRS:
        src_word = pair[source_idx]
        tgt_word = pair[target_idx]

        # 构造正则表达式
        # Group 1: ("[^"]*")  -> 匹配双引号括起来的任意内容 (保护区)
        # Group 2: (\bkeyword\b) -> 匹配目标关键词，\b 确保词边界 (匹配区)
        # re.IGNORECASE -> 忽略大小写
        # re.escape(src_word) 用于处理像 "fast-paced" 中间的连字符，防止被正则误读
        pattern = r'("[^"]*")|\b(' + re.escape(src_word) + r")\b"

        def replace_callback(match):
            nonlocal replaced_flag

            # 如果匹配到了 Group 1 (双引号内容)，原样返回，不动它
            if match.group(1):
                return match.group(1)

            # 如果匹配到了 Group 2 (关键词)，进行替换
            if match.group(2):
                original_word = match.group(2)
                replaced_flag = True
                return match_case(original_word, tgt_word)

        # 进行替换
        text = re.sub(pattern, replace_callback, text, flags=re.IGNORECASE)

    return {"flag": replaced_flag, "text": text}


ACCENT = [
    "american",
    "argentine",
    "australian",
    "belgian",
    "brazilian",
    "british",
    "british-american",
    "british-guyanese",
    "brooklyn/new york",
    "canadian",
    "cantonese",
    "chilean",
    "chinese",
    "colombian",
    "colombian-american",
    "croatian",
    "czech",
    "dari",
    "dominican",
    "dutch",
    "english",
    "filipino",
    "finnish",
    "french",
    "german",
    "hungarian",
    "indian",
    "irish",
    "italian",
    "jamaican",
    "japanese",
    "jordanian",
    "mandarin",
    "mexican",
    "new zealand",
    "nigerian",
    "northern irish",
    "norwegian",
    "paraguayan",
    "polish",
    "portuguese",
    "romanian",
    "russian",
    "scottish",
    "serbian",
    "slovenian",
    "southern american",
    "spanish",
    "swedish",
    "swiss",
    "turkish",
    "ukrainian",
    "welsh",
]
ACCENT_GROUPS: Dict[str, List[str]] = {
    # 【英语核心圈】：母语为英语，区别主要在元音变化和R音
    "english_native": [
        "american",
        "british",
        "australian",
        "canadian",
        "new zealand",
        "irish",
        "scottish",
        "welsh",
        "southern american",
        "brooklyn/new york",
        "northern irish",
        "british-american",
        "english",
    ],
    # 【英语 L2 / 独特韵律】：非母语英语，或有强烈地域韵律特征的英语
    "english_l2_distinct": [
        "indian",
        "nigerian",
        "jamaican",
        "filipino",
        "british-guyanese",
        "colombian-american",  # 虽有American后缀，但口音特征往往带有明显的非母语韵律
    ],
    # 【拉丁语族】：元音清晰、音节速率快、重音模式相似
    "romance_latin": [
        "spanish",
        "mexican",
        "colombian",
        "argentine",
        "chilean",
        "paraguayan",
        "portuguese",
        "brazilian",
        "italian",
        "french",
        "romanian",
        "dominican",
    ],
    # 【日耳曼与斯拉夫】：辅音丛多、语调相对平直或有特定重音起伏
    "germanic_slavic": [
        "german",
        "dutch",
        "swedish",
        "norwegian",
        "swiss",
        "belgian",
        "russian",
        "ukrainian",
        "polish",
        "czech",
        "croatian",
        "serbian",
        "slovenian",
        "hungarian",
        "finnish",
    ],
    # 【亚洲/声调语言】：受声调或高低音重音（Pitch Accent）影响明显的口音
    "asian_tonal": [
        "mandarin",
        "cantonese",
        "chinese",
        "japanese",
    ],
    # 【中东/其他】：喉音特征、独特的元音发音
    "middle_eastern_other": ["turkish", "dari", "jordanian"],
}
ACCENT2GROUP = {
    accent: group_name
    for group_name, accents in ACCENT_GROUPS.items()
    for accent in accents
}


def sample_negative_accent(
    accent: str,
    p_intra_group: float = 0.20,
) -> str:
    # 1. 确定锚点所在的组
    src_group_name = ACCENT2GROUP.get(accent)
    if not src_group_name:
        raise ValueError(f"Accent '{accent}' not found in any group.")

    # 2. 决定采样策略 (Intra vs Inter)
    # 只有当组内成员大于1个时，才有可能进行组内采样
    group_members = ACCENT_GROUPS[src_group_name]
    can_do_intra = len(group_members) > 1

    is_intra_sample = random.random() < p_intra_group

    target_group_name = ""
    negative_accent = ""
    difficulty = ""

    # === 策略 A: 组内负例 (Hard) ===
    # 逻辑：只要随机到了intra概率，并且该组有得选，就选组内
    if is_intra_sample and can_do_intra:
        difficulty = "hard (intra-group)"
        target_group_name = src_group_name

        # 从组内选一个不是自己的
        candidates = [a for a in group_members if a != accent]
        negative_accent = random.choice(candidates)

    # === 策略 B: 跨组负例 (Easy/Normal) ===
    # 逻辑：没随机到intra，或者被迫fallback到inter（因为该组只有一个独苗）
    else:
        difficulty = "normal (inter-group)"

        # 获取所有组名，移除当前组
        other_groups = [g for g in ACCENT_GROUPS.keys() if g != src_group_name]

        # 随机选一个组
        target_group_name = random.choice(other_groups)

        # 在该组内随机选一个口音
        negative_accent = random.choice(ACCENT_GROUPS[target_group_name])

    return negative_accent


def perturb_accent_in_text(text: str, accent: str) -> dict:
    if accent not in ACCENT2GROUP:
        return {"flag": False, "text": text}

    # 1. 获取负例目标 (Target)
    # 每次调用都会随机采样，可能是 Hard 也可能是 Easy
    tgt_accent = sample_negative_accent(accent)

    replaced_flag = False

    # 构造正则表达式
    # Group 1: ("[^"]*")  -> 匹配双引号括起来的任意内容 (保护区)
    # Group 2: (\bkeyword\b) -> 匹配目标关键词，\b 确保词边界 (匹配区)
    # re.IGNORECASE -> 忽略大小写
    # re.escape(accent) 用于连字符，防止被正则误读
    pattern = r'("[^"]*")|\b(' + re.escape(accent) + r")\b"

    def replace_callback(match):
        nonlocal replaced_flag

        # 如果匹配到了 Group 1 (双引号内容)，原样返回，不动它
        if match.group(1):
            return match.group(1)

        # 如果匹配到了 Group 2 (关键词)，进行替换
        if match.group(2):
            original_word = match.group(2)
            replaced_flag = True
            return match_case(original_word, tgt_accent)

    # 进行替换
    text = re.sub(pattern, replace_callback, text, flags=re.IGNORECASE)

    return {"flag": replaced_flag, "text": text}


INTRINSIC_TAGS = [
    "authoritative",
    "booming",
    "crisp",
    "deep",
    "flowing",
    "guttural",
    "hesitant",
    "hushed",
    "husky",
    "inviting",
    "lisp",
    "monotone",
    "monotonous",
    "nasal",
    "pitchy",
    "punctuated",
    "raspy",
    "shrill",
    "silky",
    "slurred",
    "smooth",
    "soft",
    "staccato",
    "stammering",
    "upbeat",
    "vocal-fry",
]
INTRINSIC_PAIRS = [
    # --- 1. 质感/音色 (Texture: Rough vs Smooth) ---
    ("raspy", "smooth"),  # 粗糙/沙哑 vs 光滑
    ("raspiness", "smoothness"),
    ("raspily", "smoothly"),
    ("raspy", "silky"),  # 沙哑 vs 丝滑
    ("raspiness", "silkiness"),
    ("raspily", "silkily"),
    ("guttural", "silky"),  # 喉音 vs 丝滑
    ("gutturally", "silkily"),
    ("vocal-fry", "smoothness"),  # 气泡音 vs 光滑
    ("slurred", "crisp"),  # 含糊不清 vs 清晰干脆
    ("slurring", "crispness"),
    ("slurringly", "crisply"),
    ("husky", "crisp"),  # 烟嗓 vs 清脆
    ("huskiness", "crispness"),
    ("huskily", "crisply"),
    ("nasal", "deep"),  # 鼻音 vs 深沉
    ("nasality", "depth"),
    ("nasally", "deeply"),
    # --- 2. 节奏/流利度 (Rhythm: Broken vs Flowing) ---
    ("staccato", "flowing"),
    ("punctuated", "flowing"),  # 强调/顿挫 vs 流畅
    ("punctuation", "flow"),
    ("stammering", "flowing"),  # 结巴 vs 流畅
    ("stammer", "flow"),
    ("stammeringly", "flowingly"),
    ("hesitant", "flowing"),  # 迟疑 vs 流畅
    ("hesitance", "flow"),
    ("hesitantly", "flowingly"),
    ("lisp", "crispness"),  # 口齿不清 vs 清晰
    ("lisping", "crisp"),
    ("lispingly", "crisply"),
    # --- 3. 音高 (Pitch: High vs Low) ---
    ("shrill", "deep"),  # 尖锐 vs 深沉
    ("shrillness", "depth"),
    ("shrilly", "deeply"),
    ("pitchy", "monotone"),  # 音调起伏 vs 单调平直
    ("pitchiness", "monotony"),
    ("pitchily", "monotonously"),
    # --- 4. 能量/情绪 (Energy: High/Dynamic vs Low/Static) ---
    ("booming", "hushed"),  # 洪亮 vs 低声
    ("boom", "hush"),
    ("booming", "hushedly"),
    ("booming", "soft"),  # 洪亮 vs 轻柔
    ("boom", "softness"),
    ("booming", "softly"),
    ("upbeat", "monotonous"),  # 欢快 vs 单调乏味
    ("upbeat", "monotone"),  # 欢快 vs 单调
    ("authoritative", "hesitant"),  # 权威 vs 迟疑
    ("authority", "hesitance"),
    ("authoritatively", "hesitantly"),
    ("inviting", "authoritative"),  # 亲切 vs 权威
    ("invitation", "authority"),
    ("invitingly", "authoritatively"),
]
INTRINSIC_TAG_MAP: Dict[str, List[str]] = {}
[
    (
        INTRINSIC_TAG_MAP.setdefault(t1, []).append(t2),
        INTRINSIC_TAG_MAP.setdefault(t2, []).append(t1),
    )
    for t1, t2 in INTRINSIC_PAIRS
]


def perturb_intrinsic_tags(text: str, intrinsic_tags: List[str]) -> dict:
    intrinsic_tags_copy = intrinsic_tags[:]
    random.shuffle(intrinsic_tags_copy)
    flag = False
    for tag in intrinsic_tags_copy:
        result_dict = perturb_intrinsic_tag_in_text(text, tag)
        flag = result_dict["flag"]
        text = result_dict["text"]
        if flag:
            break
    return {"flag": flag, "text": text}


def perturb_intrinsic_tag_in_text(text: str, intrinsic_tag: str) -> dict:
    replaced_flag = False

    # 如果标签不合法，直接返回
    if intrinsic_tag not in INTRINSIC_TAG_MAP:
        return {"flag": False, "text": text}

    # 随机选择一个负例目标
    tgt_tag = random.choice(INTRINSIC_TAG_MAP[intrinsic_tag])

    # 构造正则表达式
    # Group 1: ("[^"]*")  -> 匹配双引号括起来的任意内容 (保护区)
    # Group 2: (\bkeyword\b) -> 匹配目标关键词，\b 确保词边界 (匹配区)
    # re.IGNORECASE -> 忽略大小写
    pattern = r'("[^"]*")|\b(' + re.escape(intrinsic_tag) + r")\b"

    def replace_callback(match):
        nonlocal replaced_flag

        # 如果匹配到了 Group 1 (双引号内容)，原样返回，不动它
        if match.group(1):
            return match.group(1)

        # 如果匹配到了 Group 2 (关键词)，进行替换
        if match.group(2):
            original_word = match.group(2)
            replaced_flag = True
            return match_case(original_word, tgt_tag)

    # 进行替换
    text = re.sub(pattern, replace_callback, text, flags=re.IGNORECASE)

    return {"flag": replaced_flag, "text": text}


SITUATIONAL_TAGS = [
    "admiring",
    "angry",
    "animated",
    "anxious",
    "awed",
    "bored",
    "calm",
    "confused",
    "desirous",
    "disgusted",
    "enthusiastic",
    "enunciated",
    "guilt",
    "happy",
    "laughing",
    "loud",
    "pained",
    "passive",
    "saddened",
    "sarcastic",
    "scared",
    "singsong",
    "sleepy",
    "sympathetic",
    "whispered",
]
SITUATIONAL_PAIRS = [
    # --- 1. 情绪效价 (Valence: Positive vs Negative) ---
    ("happy", "sad"),
    ("happy", "saddened"),
    ("happiness", "sadness"),
    ("happiness", "pain"),
    ("happiness", "anger"),
    ("happy", "pained"),
    ("happy", "angry"),
    ("enthusiastic", "bored"),
    ("enthusiasm", "boredom"),
    ("laughing", "saddened"),
    ("laugh", "sadness"),
    ("laughter", "sadness"),
    ("guilt", "happiness"),
    ("guilty", "happy"),
    # --- 2. 唤醒度/能量 (Arousal: High vs Low) ---
    ("angry", "calm"),
    ("anger", "calmness"),
    ("scared", "calm"),
    ("fear", "calmness"),
    ("anxious", "calm"),
    ("anxiety", "calmness"),
    ("animated", "passive"),
    ("animated", "sleepy"),
    ("loud", "whispered"),
    ("confused", "calm"),
    ("confusion", "calmness"),
    # --- 3. 态度/互动 (Attitude: Pull vs Push) ---
    ("admiring", "disgusted"),
    ("admiration", "disgust"),
    ("desirous", "disgusted"),
    ("desire", "disgust"),
    ("awed", "bored"),
    ("awe", "boredom"),
    ("sympathetic", "sarcastic"),
    ("sympathy", "sarcasm"),
    ("admiring", "sarcastic"),
    ("admiration", "sarcasm"),
    # --- 4. 清晰度 ---
    ("enunciated", "slurred"),
    ("enunciation", "slurring"),
    ("singsong", "monotone"),
    ("singsongly", "monotonously"),
]
SITUATIONAL_TAG_MAP: Dict[str, List[str]] = {}
[
    (
        SITUATIONAL_TAG_MAP.setdefault(t1, []).append(t2),
        SITUATIONAL_TAG_MAP.setdefault(t2, []).append(t1),
    )
    for t1, t2 in SITUATIONAL_PAIRS
]


def perturb_situational_tags(text: str, situational_tags: List[str]) -> dict:
    situational_tags_copy = situational_tags[:]
    random.shuffle(situational_tags_copy)
    flag = False
    for tag in situational_tags_copy:
        result_dict = perturb_situational_tag_in_text(text, tag)
        flag = result_dict["flag"]
        text = result_dict["text"]
        if flag:
            break
    return {"flag": flag, "text": text}


def perturb_situational_tag_in_text(text: str, situational_tag: str) -> dict:
    replaced_flag = False

    # 如果标签不合法，直接返回
    if situational_tag not in SITUATIONAL_TAG_MAP:
        return {"flag": False, "text": text}

    # 随机选择一个负例目标
    tgt_tag = random.choice(SITUATIONAL_TAG_MAP[situational_tag])

    # 构造正则表达式
    # Group 1: ("[^"]*")  -> 匹配双引号括起来的任意内容 (保护区)
    # Group 2: (\bkeyword\b) -> 匹配目标关键词，\b 确保词边界 (匹配区)
    # re.IGNORECASE -> 忽略大小写
    pattern = r'("[^"]*")|\b(' + re.escape(situational_tag) + r")\b"

    def replace_callback(match):
        nonlocal replaced_flag

        # 如果匹配到了 Group 1 (双引号内容)，原样返回，不动它
        if match.group(1):
            return match.group(1)

        # 如果匹配到了 Group 2 (关键词)，进行替换
        if match.group(2):
            original_word = match.group(2)
            replaced_flag = True
            return match_case(original_word, tgt_tag)

    # 进行替换
    text = re.sub(pattern, replace_callback, text, flags=re.IGNORECASE)

    return {"flag": replaced_flag, "text": text}


def perturb_one_attribution_in_text(
    text: str,
    gender: str,
    speaking_rate: str,
    pitch: str,
    accent: str,
    intrinsic_tags: List[str],
    situational_tags: List[str],
) -> str:
    perturbation_functions = [
        perturb_gender_in_text,
        perturb_speaking_rate_in_text,
        perturb_pitch_in_text,
        perturb_accent_in_text,
        perturb_intrinsic_tags,
        perturb_situational_tags,
    ]
    attributions = [
        gender,
        speaking_rate,
        pitch,
        accent,
        intrinsic_tags,
        situational_tags,
    ]
    candidates = list(zip(perturbation_functions, attributions))
    random.shuffle(candidates)
    for func, attr in candidates:
        result_dict = func(text, attr)
        if result_dict["flag"]:
            return result_dict["text"]

    if gender == "male":
        result_dict = perturb_gender_in_text(text, "female")
    elif gender == "female":
        result_dict = perturb_gender_in_text(text, "male")

    if result_dict["flag"]:
        return result_dict["text"]

    raise ValueError("No attribution found to perturb.")


if __name__ == "__main__":
    import difflib

    from lhotse import load_manifest_lazy

    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    def color_diff_ori(ori_text: str, norm_text: str) -> str:
        sm = difflib.SequenceMatcher(a=ori_text, b=norm_text, autojunk=False)
        out = []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                out.append(ori_text[i1:i2])

            elif tag == "delete":
                out.append(f"{RED}{ori_text[i1:i2]}{RESET}")

            elif tag == "insert":
                out.append(f"{GREEN}{norm_text[j1:j2]}{RESET}")

            elif tag == "replace":
                if i1 != i2:
                    out.append(f"{RED}{ori_text[i1:i2]}{RESET}")
                if j1 != j2:
                    out.append(f"{GREEN}{norm_text[j1:j2]}{RESET}")

        return "".join(out)

    train_cuts = load_manifest_lazy(
        "data/manifests/paraspeechcaps_cuts_train_base_shuf-selected.jsonl.gz"
    )
    cnt_short = 0
    cnt_long = 0
    cnt_short_total = 0
    cnt_long_total = 0
    for cut in train_cuts:
        gender = cut.supervisions[0].gender
        speaking_rate = cut.supervisions[0].custom["speaking_rate"]
        pitch = cut.supervisions[0].custom["pitch"]
        accent = cut.supervisions[0].custom["accent"]
        intrinsic_tags = cut.supervisions[0].custom["intrinsic_tags"]
        situational_tags = cut.supervisions[0].custom["situational_tags"]

        short_captions = cut.supervisions[0].custom["short_captions"]
        long_captions = cut.supervisions[0].custom["long_captions"]

        for short_caption in short_captions:
            text = perturb_one_attribution_in_text(
                short_caption,
                gender,
                speaking_rate,
                pitch,
                accent,
                intrinsic_tags,
                situational_tags,
            )
            # print(color_diff_ori(short_caption, text))

        for long_caption in long_captions:
            cnt_long_total += 1
            text = perturb_one_attribution_in_text(
                long_caption,
                gender,
                speaking_rate,
                pitch,
                accent,
                intrinsic_tags,
                situational_tags,
            )
            # print(color_diff_ori(long_caption, text))
