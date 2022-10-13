ARGPARSE_DESCRIPTION = """
This script accesses the CSJ/SDB/{core,noncore} directories and generates transcripts
in accordance to the tag decisions listed in the .ini file. 

It does the following in sequence:-

**MOVE**
1. Copies each .sdb files from /SDB into its own directory in the designated `trans_dir`, 
   i.e. {trans_dir}/{spk_id}/{spk_id}.sdb
2. Verifies that the corresponding wav file exists in the /WAV directory, and outputs that 
   absolute path into {spk_id}-wav.list
3. Moves the predefined datasets for eval1, eval2, eval3, and excluded, into its own dataset 
   directory
4. Touches a .done_mv in `trans_dir`.
NOTE: If a .done_mv exists already in `trans_dir`, then this stage is skipped.

**PARSE**
1. Takes in an .ini file which - among others - contains the behaviour for each tag and 
   the segment details.
2. Parses all .sdb files it can find within `trans_dir`, and optionally outputs a segment file. 
3. Touches a .done in `trans_dir`. 

Differences to kaldi include:-
1. The evaluation datasets do not follow `trans_dir`/eval/eval{i}, but are instead saved 
    in the same level as core, noncore, and excluded. 
2. Morphology tags are parsed but not included in the final transcript. The original morpheme  
    segmentations are preserved by spacing, i.e. 分かち書き, so removal, if required, has to be 
    done at a later stage. 
3. Kana pronunciations are parsed but not included in the final transcript. 

"""

import argparse
from configparser import ConfigParser
from copy import copy
from io import TextIOWrapper
import logging
from multiprocessing import Queue, get_context
import os
from typing import Dict, List, Tuple
from pathlib import Path
import re

FULL_DATA_PARTS = ['core', 'noncore', 'eval1', 'eval2', 'eval3', 'excluded']

# Exclude speaker ID
A01M0056 = ["S05M0613", "R00M0187", "D01M0019", "D04M0056", "D02M0028", "D03M0017"]

# Evaluation set ID
EVAL = [
    # eval1
    ['A01M0110', 'A01M0137', 'A01M0097', 'A04M0123', 'A04M0121', 'A04M0051', 'A03M0156', 'A03M0112', 'A03M0106', 'A05M0011'],
    # eval2
    ['A01M0056', 'A03F0072', 'A02M0012', 'A03M0016', 'A06M0064', 'A06F0135', 'A01F0034', 'A01F0063', 'A01F0001', 'A01M0141'],
    # eval3
    ['S00M0112', 'S00F0066', 'S00M0213', 'S00F0019', 'S00M0079', 'S01F0105', 'S00F0152', 'S00M0070', 'S00M0008', 'S00F0148'],
    ]

# https://stackoverflow.com/questions/23589174/regex-pattern-to-match-excluding-when-except-between
tag_regex = re.compile(r'(<|>|\+)|([\x00-\x7F])')

def kana2romaji(katakana : str) -> str:
    if not KATAKANA2ROMAJI or not katakana:
        return katakana
    
    tmp = []
    mem = ''
    
    for c in katakana[::-1]:
        if c in ['ャ', 'ュ', 'ョ', 'ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ヮ']:
            mem += c
            continue 
        if mem:
            c += mem
            mem = ''
        
        try:
            tmp.append(KATAKANA2ROMAJI[c])
        except KeyError:
            for i in c[::-1]:
                try:
                    tmp.append(KATAKANA2ROMAJI[i])
                except KeyError:
                    tmp.append(i)
    
    if mem:
        try:
            tmp.append(KATAKANA2ROMAJI[mem])
        except KeyError:
            for i in mem[::-1]:
                try:
                    tmp.append(KATAKANA2ROMAJI[i])
                except KeyError:
                    tmp.append(i)
                    
    return ''.join(tmp[::-1])


#--------------------------#
class CSJSDB_Word:
    
    time = ''
    surface = ''
    notag = ''
    pos1 = ''
    cForm = ''
    cType1 = ''
    pos2 = ''
    cType2 = ''
    other = ''
    pron = ''
    spk_id = ''
    sgid = 0
    start = -1.
    end = -1.
    morph = ''
    words = []
    
    @staticmethod
    def from_line(line = ''):
        word = CSJSDB_Word()
        line = line.strip().split('\t')
        
        for f, i in FIELDS.items():
            try:
                attr = line[i]
            except IndexError:
                attr = ''
            setattr(word, f, attr)
        
        for _ in range(2):
            # Do twice in case of "んーー"
            for c, r in zip(['んー', 'ンー'], ['ん','ン']):
                word.pron = word.pron.replace(c,r)
                word.surface = word.surface.replace(c,r)
        
        # Make morph
        morph = [getattr(word, s) for s in MORPH]
        word.morph = '/'.join(m for m in morph if m)
        for c in ['Ａ', '１', '２', '３', '４']:
            word.morph = word.morph.replace(c, '')
        word.morph = word.morph.replace('　', '＿')
        if word.morph:
            word.morph = '+'+word.morph
        
        # Parse time
        word.sgid, start_end, channel = word.time.split(' ')
        word.start, word.end = [float(s) for s in start_end.split('-')]
        if word.spk_id[0] == 'D':
            word.spk_id = word.spk_id + '-' + channel.split(':')[0]
        
        return word
        
    @staticmethod
    def from_dict(other : Dict):
        word = CSJSDB_Word()
        for k,v in other.items():
            setattr(word, k, v)

    def _parse_pron(self):
        for tag, replace in REPLACEMENTS_PRON.items():
            self.pron = self.pron.replace(tag, replace)
        
        # This is for pauses <P:00453.373-00454.013>
        self.pron = re.sub(r'<P:.+>', REPLACEMENTS_PRON['<P>'], self.pron)
        matches = tag_regex.findall(self.pron)
        if all(not g2 for _, g2 in matches):
            return self.pron
        elif self.pron.count('(') != self.pron.count(')'):
            return None
        
        open_brackets = [pos for pos, c in enumerate(self.pron) if c == '(']
        close_brackets = [pos for pos, c in enumerate(self.pron) if c == ')']
        
        if open_brackets[0] > close_brackets[-1]:
            return None
        
        pron = self._parse(-1, self.pron, 'p')['string']
        return pron
        
    def _parse_surface(self):
        for ori, replace in REPLACEMENTS_SURFACE.items():
            self.surface = self.surface.replace(ori, replace)
        # Occurs for example in noncore/A01F0063: 0099 00280.998-00284.221 L:-001-001	一・	一・	イチ	一	イチ
        self.surface = self.surface.rstrip('・')
        # This is for pauses <P:00453.373-00454.013>
        self.surface = re.sub(r'<P:.+>', REPLACEMENTS_SURFACE['<P>'], self.surface)
        matches = tag_regex.findall(self.surface)
        if all(not g2 for _, g2 in matches):
            return self.surface
        elif self.surface.count('(') != self.surface.count(')'):
            return None

        open_brackets = [pos for pos, c in enumerate(self.surface) if c == '(']
        close_brackets = [pos for pos, c in enumerate(self.surface) if c == ')']
        
        if open_brackets[0] > close_brackets[-1]:
            return None
        surface = self._parse(-1, self.surface, 's')['string']
        return surface
        
    def _decide(self, tag, choices, ps) -> str:
        assert len(choices) > 1
        decisions = DECISIONS_PRON if ps == 'p' else DECISIONS_SURFACE
        for t, decision in decisions.items():
            if t != tag:
                continue 
            ret = -1
            if isinstance(decision, int):
                ret = choices[decision]
            else:            
                if decision[:5] == 'eval:':
                    ret = eval(decision[5:])
                elif decision[:5] == 'exec:':
                    exec(decision[5:])
                else:
                    ret = PLUS.join(decision for _ in range(choices[0].count(PLUS)+1))
            
            if ret != -1:
                return ret
        
        raise NotImplementedError(f"Unknown tag {tag} encountered")
    
    def __bool__(self):
        ret = bool(self.surface and self.pron) 
        return ret
    
    def __eq__(self, other : 'CSJSDB_Word'):
        return self.surface == other.surface and \
            self.pron == other.pron and \
            self.morph == other.morph

    def __repr__(self):
        return self.to_lexicon(' ')

    def __hash__(self):
        return hash(self.__repr__())
    
    def _parse_pronsurface(self) -> bool:
        new_pron = self._parse_pron()
        new_surface = self._parse_surface()
        
        if new_pron is not None:
            self.pron = new_pron
        
        if new_surface is not None:
            self.surface = new_surface
            self.notag = new_surface
        
        if new_pron is not None and new_surface is not None:
            return True
        else:
            return False
        
    def _parse(self, open_bracket : int, text : str, ps : str, parent_tag = '') -> Dict:
        assert ps in ['p', 's']
        result = ''
        mem = ''
        i = open_bracket + 1
        tag = ''
        choices = ['']
        long_multiline = text.count(PLUS) > 5 # HARDCODE ALERT
        
        while i < len(text):
            c = text[i]
            
            if c == '(':
                ret = self._parse(i, text, ps, tag)
                c = ret['string']
                i = ret['end']
            mem += c
            matches = tag_regex.search(c)
            
            if c == ')' and not tag:
                return {'string': mem, 'end': i}
                
            elif c == ')':
                if tag == 'A' and choices[0] and choices[0][0] in JPN_NUM:
                    tag = 'A_num'

                if open_bracket and not long_multiline:
                    tag += '^'
                
                result += self._decide(tag, choices + [PLUS * choices[0].count(PLUS)], ps)
                return {'string': result, 'end': i}
            elif c == ';':
                choices.append('')
            elif c == ',':
                choices.append('')
                if ',' not in tag:
                    tag += ','
            elif c == ' ':
                pass
            elif matches and matches.group(2):  
                tag += c
            elif not tag and open_bracket > -1 and c in ['笑', '泣', '咳']:
                tag = c
            else:
                choices[-1] = choices[-1] + c
            i += 1
        
        return {'string': mem, 'end': i}
    
    def to_lexicon(self, separator = '\t'):
        return f"{self.surface}{self.morph}{separator}{self.pron}"
    
    def to_transcript(self):
        return f"{self.surface}{self.morph}"
    
    def convert_pron(self):
        self.pron = kana2romaji(self.pron)
        if '+' not in self.pron:
            self.pron = tuple(self.pron)
        else:
            self.pron = (self.pron,)
        
    @staticmethod
    def from_file(fin : TextIOWrapper) -> List['CSJSDB_Word']:
        """Reads an SDB file and outputs a list of `CSJSDB_Word` 
        nodes.

        Returns:
            List['CSJSDB_Word]: A list of CSJSDB_Word
        """

        ret : List[CSJSDB_Word] = []
        mem = None
        for line in fin:
            w = CSJSDB_Word.from_line(line)
            is_complete_word = w._parse_pronsurface()
            
            if mem is not None:
                mem._add_word(w)
                # assert len(mem.words) < 50  or 'R' in mem.surface
                if mem._parse_pronsurface():
                    mem = mem._resolve_multiline()
                    
                    ret.extend(mem)
                    mem = None
            elif is_complete_word and not w:
                continue
            elif is_complete_word: 
                # assert all(p not in w.pron for p in ['(', ')', 'x'])
                ret.append(w)
            else:
                mem = w

        for word in ret:
            assert all(p not in word.surface for p in ['(', ')', ';']), (#, '×'])  , (
                f"surface {word.surface} contains invalid character. {fin.name}"
            )
                
            assert all(p not in word.pron for p in ['(', ')', ';']), (#, '×'])  , (
                f"pron {word.pron} contains invalid character. {fin.name}"
            )     

            word.convert_pron()   
        
        return ret 
    
    def _add_word(self, w : 'CSJSDB_Word'):
        if not self.words:
            self.words = [copy(self)] 
        
        self.words.append(w)
        try:
            del w.words
        except AttributeError:
            pass 
        self.__dict__.update(w.__dict__) # = w
        # if len(self.words) > 1:
        self.surface = PLUS.join(ww.surface for ww in self.words)
        self.notag = PLUS.join(ww.notag for ww in self.words)
        self.pron = PLUS.join(ww.pron for ww in self.words)
        self.start = self.words[0].start
        self.end = self.words[-1].end
            
    def _resolve_multiline(self):
        # Only called when trying to resolve a multiline CSJSDB_Word object. 
        split_surface = PLUS in self.surface 
        split_pron = PLUS in self.pron 
        ret = []
        
        if split_surface and split_pron:
            assert split_pron
            surfaces = self.surface.split(PLUS)
            prons = self.pron.split(PLUS)
            len_words = len(self.words)
            surfaces = surfaces + [''] * (len_words - len(surfaces))
            prons = prons + [''] * (len_words - len(prons))
            
            for s, p, i in zip(surfaces, prons, range(len_words)):
                self.words[i].surface = s
                self.words[i].pron = p
            
            ret = [w for w in self.words if w]
        elif not self:
            pass
        elif split_surface and not split_pron:
            self.surface = re.sub(r'\+<.+>|'+PLUS, '', self.surface)
            ret = [self]
        elif not split_surface and split_pron:
            self.pron = re.sub(r'\+<.+>|'+PLUS, '', self.pron)
            ret = [self]
        else:
            self.surface = re.sub(r'\+<.+>', '', self.surface)
            self.pron = re.sub(r'\+<.+>', '', self.pron)
            ret = [self] 
        del self.words 
        return ret

def modify_text(word_list : List[CSJSDB_Word], segments : List[str], gap_sym : str, gap : float) -> List[Dict[str, List[str]]]:
    """Takes a list of parsed CSJSDB words and a list of time boundaries for each segment, and outputs them in transcript format

    Args:
        word_list (List[CSJSDB_Word]): List of parsed words from CSJ SDB
        segments (List[str]): List of time boundaries for each segment
        gap (float): Permissible period of nonverbal noise. If exceeded, a new segment is created. 
        gap_sym (str): Use this symbol to represent gap, if nonverbal noise does not exceed `gap`. Pass
                        an empty string to avoid adding any symbol.

    Returns:
        List[Dict[str, List[str]]]: A list of maximum two elements. 
                If len == 2, first element is Left channel, and second element is Right channel
                Available fields: 
                
                'spk_id': the speaker ID, including the trailing 'L' and 'R' if two-channeled 
                'text': the output text

    """   

    last_end = word_list[0].start
    
    segments_ = []
    for s in segments:
        sgid, start, end = s.split()
        start = float(start)
        end = float(end)
        segments_.append((sgid, start, end))
    segments = segments_.copy()
    line_sgid, line_start, line_end = segments_.pop(0)

    single_char_gap = '⨋'
    out = []
    line = []
    tobreak = False 
    for word in word_list:          
        
        if word.spk_id not in line_sgid:
            continue 
        elif word.start < line_start:
            continue
        elif word.end <= line_end:
            if gap_sym and gap < (word.start - last_end):
                line.append(gap_sym)
            line.append(word.surface)
        else: # word.end > line_end
            line = ' '.join(line).replace(single_char_gap, gap_sym)
            # assert '×' not in line
            out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} " + line)
            
            try:
                line_sgid, line_start, line_end = segments_.pop(0)
            except IndexError:
                line = []
                tobreak = True
                break
            
            if not word.spk_id in line_sgid:
                continue
            
            while word.start >= line_end:
                out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} ")
                try: 
                    line_sgid, line_start, line_end = segments_.pop(0)
                except IndexError:
                    line = []
                    tobreak = True
                    break
            if tobreak:
                break
            line = [word.surface]
    
        last_end = word.end

    if not tobreak:
        line = ' '.join(line).replace(single_char_gap, gap_sym)
        # assert '×' not in line
        out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} " + line)
        
    while segments_:
        line_sgid, line_start, line_end = segments_.pop(0)
        out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} ")

    return {'text': out, 'spk_id': line_sgid[:-5], 'segments': segments}


def make_text(word_list : List[CSJSDB_Word], gap : float, maxlen : float, minlen : float, gap_sym : str) -> List[Dict[str, List[str]]]:
    """Takes a list of parsed CSJSDB words and outputs them in transcript format

    Args:
        word_list (List[CSJSDB_Word]): List of parsed words from CSJ SDB
        gap (float): Permissible period of nonverbal noise. If exceeded, a new segment is created. 
        maxlen (float): Maximum length of the segment.
        minlen (float): Minimum length of the segment. Segments shorter than this will be silently dropped.
        gap_sym (str): Use this symbol to represent gap, if nonverbal noise does not exceed `gap`. Pass
                        an empty string to avoid adding any symbol.

    Returns:
        List[Dict[str, List[str]]]: A list of maximum two elements. 
                If len == 2, first element is Left channel, and second element is Right channel
                Available fields: 
                
                'spk_id': the speaker ID, including the trailing 'L' and 'R' if two-channeled 
                'text': the output text

    """    
    
    
    line_sgid = word_list[0].sgid
    line_spk_id = word_list[0].spk_id
    line_start = word_list[0].start
    last_sgid = word_list[0].sgid
    last_spk_id = word_list[0].spk_id
    last_end = word_list[0].start
    
    out = []
    line = []
    segments = []

    single_char_gap = '⨋'

    for word in word_list:
        
        if last_sgid == word.sgid and last_spk_id == word.spk_id:
            line.append(word.surface)
        elif gap < (word.start - last_end) or maxlen < (last_end - line_start) or line_spk_id != word.spk_id:
            line = ' '.join(line).replace(single_char_gap, gap_sym)
            if minlen < (last_end - line_start) and '×' not in line:
                out.append(f"{line_spk_id}_{line_sgid} {line_start:09.3f} {last_end:09.3f} " + line) #' '.join(line).replace(single_char_gap, gap_sym))
                segments.append((f"{line_spk_id}_{line_sgid}", line_start, last_end))         
            
            line_start = word.start 
            line_sgid = word.sgid
            line_spk_id = word.spk_id
            line = [word.surface]
        elif gap_sym:
            line.extend([single_char_gap, word.surface])
        else:
            line.append(word.surface)
        
        last_sgid = word.sgid
        last_spk_id = word.spk_id
        last_end = word.end
        
    line = ' '.join(line).replace(single_char_gap, gap_sym)
    if line and '×' not in line:
        out.append(f"{line_spk_id}_{line_sgid} {line_start:09.3f} {last_end:09.3f} " + line)
        segments.append((f"{line_spk_id}_{line_sgid}", line_start, last_end))
    
    if last_spk_id[-1] not in ['R', 'L']: 
        return [{'text': out, 'spk_id': last_spk_id, 'segments': segments}]
    else:
        out = _tear_apart_LR(out, segments)
        spk_id = last_spk_id[:-2]
        return [
            {'text': out['out_L'], 'spk_id': spk_id+'-L', 'segments': out['segment_L']},
            {'text': out['out_R'], 'spk_id': spk_id+'-R', 'segments': out['segment_R']}
        ]

def _tear_apart_LR(lines : List[str], segments : List[Tuple]):
    out_R = []
    out_L = []
    segment_R = []
    segment_L = []
    
    for line, segment in zip(lines, segments):
        spkid = line.split('_', maxsplit=1)[0]
        if spkid[-1] == 'R':
            out_R.append(line)
            segment_R.append(segment)
        else:
            out_L.append(line)
            segment_L.append(segment)
    
    return {'out_R': out_R, 'out_L': out_L, 'segment_R': segment_R, 'segment_L': segment_L}
    
def create_trans_dir(corpus_dir : Path, trans_dir : Path):
    
    if (trans_dir / ".done_mv").exists():
        logging.info(     
            f"{trans_dir} already created. "
            f"Delete {trans_dir / '.done_mv'} to create again."
            )
        return
    
    for ori_files in (corpus_dir / "MORPH/SDB").glob(f"*/*.sdb"):
        vol = ori_files.parts[-2]
        spk_id = ori_files.name[:-4]
        new_dir = trans_dir / vol / spk_id
        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / f"{spk_id}.sdb").write_bytes(ori_files.read_bytes())
        wav_dir = corpus_dir / "WAV" / vol 
        
        if spk_id[0] == 'D':
            l_wav = (wav_dir / f"{spk_id}-L.wav")
            r_wav = (wav_dir / f"{spk_id}-R.wav")
            assert l_wav.is_file(), f"{spk_id}-L.wav cannot be found"
            assert r_wav.is_file(), f"{spk_id}-R.wav cannot be found"
            (new_dir / f"{spk_id}-L-wav.list").write_text(l_wav.as_posix(), encoding='utf8')
            (new_dir / f"{spk_id}-R-wav.list").write_text(r_wav.as_posix(), encoding='utf8')
            
        else:
            wav = (wav_dir / f"{spk_id}.wav")
            assert wav.is_file(), f"{spk_id}.wav cannot be found"
            (new_dir / f"{spk_id}-wav.list").write_text(wav.as_posix(), encoding='utf8')
    
    for ori_files in A01M0056:
        ori_files = list(trans_dir.glob(f"*/{ori_files}/{ori_files}*"))
        
        for ori_file in ori_files:
            *same_part, vol, spk_id, filename = ori_file.as_posix().split('/')
            new_dir = Path('/'.join(same_part + ["excluded", spk_id]))
            new_dir.mkdir(parents=True, exist_ok=True)
            ori_file.rename(new_dir / filename)
        ori_files[0].parent.rmdir()

    for i, eval_list in enumerate(EVAL):
        i+=1 
        for ori_files in eval_list:
            ori_files = list(trans_dir.glob(f"*/{ori_files}/{ori_files}*"))
            
            for ori_file in ori_files:
                *same_part, vol, spk_id, filename = ori_file.as_posix().split('/')
                new_dir = Path('/'.join(same_part + [f"eval{i}", spk_id]))
                new_dir.mkdir(parents=True, exist_ok=True)
                ori_file.rename(new_dir / filename)
            ori_files[0].parent.rmdir()

    (trans_dir / ".done_mv").touch()
    logging.info("Transcripts have been moved.")

def parse_sdb_process(
    jobs_queue : Queue, 
    gap : float, 
    maxlen : float,
    minlen : float, 
    gap_sym : str, 
    trans_mode : str, 
    use_segments : bool, 
    write_segments : bool,
    ):
    
    def parse_one_sdb(sdb : Path):
        with sdb.open('r', encoding='shift_jis') as fin:
            result = CSJSDB_Word.from_file(fin)
        
        if not use_segments:
            transcripts = make_text(result, gap, maxlen, minlen, gap_sym)
        else:
            channels = ['-L-segments', '-R-segments'] if sdb.name[0] == 'D' else ['-segments']
            transcripts = []
            for channel in channels:
                segments = Path(sdb.as_posix()[:-4] + channel).read_text().split('\n')
                assert segments, segments
                transcripts.append(
                    modify_text(result, segments, "", 0.5)
                )
        
        for transcript in transcripts:
            spk_id = transcript.pop('spk_id')
            segments = transcript.pop('segments')
            (sdb.parent / f'{spk_id}-{trans_mode}.txt').write_text('\n'.join(transcript['text']), encoding='utf8')
            if write_segments:
                (sdb.parent / f'{spk_id}-segments').write_text('\n'.join(f"{s[0]} {s[1]} {s[2]}" for s in segments), encoding='utf8')
    
    while True:
        job = jobs_queue.get()
        if not job:
            break
        parse_one_sdb(sdb=job)

def load_config(config_file : Path):
    assert config_file.exists()
    config = ConfigParser()
    config.optionxform = str
    config.read(config_file)
    global PLUS, DECISIONS_PRON, DECISIONS_SURFACE, REPLACEMENTS_PRON, REPLACEMENTS_SURFACE, \
        MORPH, FIELDS, JPN_NUM, KATAKANA2ROMAJI
    PLUS = config['CONSTANTS']['PLUS']
    MORPH = config['CONSTANTS']['MORPH'].split()
    JPN_NUM = config['CONSTANTS']['JPN_NUM'].split()
    DECISIONS_PRON = {} 
    for k,v in config['DECISIONS'].items():
        try:
            DECISIONS_PRON[k] = int(v)
        except ValueError:
            DECISIONS_PRON[k] = v
        
    DECISIONS_SURFACE = DECISIONS_PRON.copy()
    REPLACEMENTS_PRON = {}
    for k,v in config['REPLACEMENTS'].items():
        REPLACEMENTS_PRON[k] = v
    REPLACEMENTS_SURFACE = REPLACEMENTS_PRON.copy()    
    FIELDS = {}
    for k,v in config['FIELDS'].items():
        FIELDS[k] = int(v)
    KATAKANA2ROMAJI = {}
    for k,v in config['KATAKANA2ROMAJI'].items():
        KATAKANA2ROMAJI[k] = v
        
    return config

def get_args():
    parser = argparse.ArgumentParser(description=ARGPARSE_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--corpus-dir", type=Path, 
                        help="Path to corpus")
    parser.add_argument("--trans-dir", type=Path,
                        help="Path to output transcripts")
    parser.add_argument("--config", type=Path,
                        help="Path to config")  
    parser.add_argument("-j", "--num-jobs", type=int,
                        default=16, help="Number of jobs to start")
    parser.add_argument("--write-segments", action="store_true",
                    help="Write segment info into a separate file")
    parser.add_argument("--use-segments", action="store_true",
                help="Use existing segments in the directory")
    parser.add_argument("--debug", action="store_true",
                        help="Use hardcoded parameters")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,        
        )
    
    if args.debug:
        args.corpus_dir = Path('/mnt/minami_data_server/t2131178/corpus/CSJ')
        args.trans_dir = Path('/mnt/minami_data_server/t2131178/corpus/CSJ/retranscript')
        args.trans_name = "disfluent"
        args.use_segments = True
        args.num_jobs = 8
        args.config = Path("local/conf/number.ini")
    
    config = load_config(args.config)
    trans_mode = config['CONSTANTS']['MODE']
    
    assert args.corpus_dir.is_dir()
    args.trans_dir.mkdir(parents=True, exist_ok=True)    
    
    logging.info("Creating transcript directories now.")
    create_trans_dir(args.corpus_dir, args.trans_dir)

    segment_config = config['SEGMENTS']
    gap = float(segment_config['gap'])
    maxlen = float(segment_config['maxlen'])
    minlen = float(segment_config['minlen'])
    gap_sym = segment_config['gap_sym']
    
    Process = get_context("fork").Process
    num_jobs = min(args.num_jobs, os.cpu_count())
    maxsize = 10 * num_jobs
    
    jobs_queue = Queue(maxsize=maxsize)
    
    workers : List[Process] = []
    
    for _ in range(num_jobs):
        worker = Process(
            target=parse_sdb_process, 
            args=(jobs_queue, gap, maxlen, minlen, gap_sym, trans_mode, args.use_segments, args.write_segments)
            )
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    num_sdb = 0
    logging.info(f"Gathering sdbs to be parsed in {trans_mode} mode now.")
    for sdb in args.trans_dir.glob("*/*/*.sdb"):
        jobs_queue.put(sdb)
        num_sdb += 1 
    
    logging.info(f"Parsing found {num_sdb} sdbs now.")
    # signal termination
    for _ in workers:
        jobs_queue.put(None)
        
    # wait for workers to terminate
    for w in workers:
        w.join()

    logging.info("All done.")
    (args.trans_dir / ".done").touch()
    
if __name__ == '__main__':
    main()