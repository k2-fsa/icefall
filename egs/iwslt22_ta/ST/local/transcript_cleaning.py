#!/usr/bin/env python
# Copyright 2020 Kanari AI (Amir Hussein)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import pdb
import numpy as np
import pandas as pd
import re
import string
import argparse
import sys
import os
import pyarabic.number as number
from pyarabic import araby

_unicode = u"\u0622\u0624\u0626\u0628\u062a\u062c\u06af\u062e\u0630\u0632\u0634\u0636\u0638\u063a\u0640\u0642\u0644\u0646\u0648\u064a\u064c\u064e\u0650\u0652\u0670\u067e\u0686\u0621\u0623\u0625\u06a4\u0627\u0629\u062b\u062d\u062f\u0631\u0633\u0635\u0637\u0639\u0641\u0643\u0645\u0647\u0649\u064b\u064d\u064f\u0651\u0671"
_buckwalter = u"|&}btjGx*z$DZg_qlnwyNaio`PJ'><VApvHdrsSTEfkmhYFKu~{"
_backwardMap = {ord(b):a for a,b in zip(_buckwalter, _unicode)}


# def number2text(anumber):
#     """
#     Convert number to arabic words, for example convert 25 --> خمسة و عشرون
#     Example:
#         >>> number2text(523)
#         خمسمئة وثلاث وعشرون
#     @param anumber: input number
#     @type anumber: int
#     @return: number words
#     @rtype: unicode
#     """
#     # test if the given type is numeric(float or int
#     # if ok, convert it to string
#     if type(anumber) is int or type(anumber) is float:
#         anumber = str(anumber)
#     # if the given type is str/unicode, test if it's a valid number
#     elif type(anumber) is str or type(anumber) is unicode:
#         try:
#             a = float(anumber);
#         except ValueError:
#             return u"صفر"
#     # if the given number not a valid return 0
#     else:
#         return u"صفر"
#     arbn = number.ArNumbers()
#     arbn.set_feminine(2)
#     return arbn.int2str(anumber)

#     return total


def fromBuckWalter(s):
    return s.translate(_backwardMap)
	
def read_tsv(data_file):
    text_data = list()
    infile = open(data_file, encoding='utf-8')
    for line in infile:
        if not line.strip():
            continue
        text= line.split('\t')
        text_data.append(text)
    
    return text_data

words_to_remove =['#غير_واضح', '#تلعثم', '#آ', '#أ', '#ال', '#بال', '#وال', ' #وب', '###تداخل', 'FRN','RFN',\
                  ' #سي', '#يي', '#هـ' , '#لل', '#بم', '#الش', '#آآ', ' #يت', '#وو', \
                  '#ومش', '#ول', '#وسي', '#غير_معروف','#العا','#مطا', '#محم' ,' #ماث','#متطو', ' #نشا', '#أأ', '#آآآ', ' #استج']
				  

def normalizeArabic(text):
   # text = re.sub("[إأٱآا]", "ا", text)
   # text = re.sub("ى", "ي", text)
   # text = re.sub("ة", "ه", text)
   # text = re.sub("ئ", "ء", text)
   # text = re.sub("ؤ", "ء", text)
  text = re.sub(r"(ه){2,}", "ههه", text)
  text = re.sub(r"(أ){2,}", "ا", text)
  text = re.sub(r"(ا){2,}", "ا", text)
  text = re.sub(r"(آ){2,}", "ا", text)
  text = re.sub(r"(ص){2,}", "ص", text)
  text = re.sub(r"(و){2,}", "و", text)
  return text
	
def remove_special_words(text):
	# remove special words from the transcription text
    for word in words_to_remove:
        if word in text:
            text = text.replace(word, '')
    return text

def remove_hashes(text):
        return re.sub(r'#?', '', text)    

def remove_english_characters(text):
        return re.sub(r'[^\u0600-\u06FF0-9\s]+', '', text)
		
def remove_diacritics(text):
    return re.sub(r'[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+', '', text)

def remove_punctuations(text):
	""" This function  removes all punctuations except the verbatim """
	
	arabic_punctuations = '''﴿﴾‘`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
	english_punctuations = string.punctuation
	all_punctuations = set(arabic_punctuations + english_punctuations)-{'@','%','.'} # remove all non verbatim punctuations
	
	for p in all_punctuations:
		if p in text:
			text = text.replace(p, '')
	text = re.sub('\s+\.','',text)  # keep only the "." that is part of a word: marsad@aljazeera.net . => marsad@aljazeera.net
	return text

def remove_extra_space(text):
	text = text.lower()
	text = re.sub('\s+', ' ', text)
	text = re.sub('\s+\.\s+', '.', text)
	return text

def remove_dot(text):

  words = text.split()
  res = []
  for word in words:
    if word.replace('.','').isnumeric():  # remove the dot if it is not part of a number 
      res.append(word)

    else:
      word = re.sub('\s+\.','',word)
      res.append(word)
    
  return " ".join(res)
    
def east_to_west_num(text):
	eastern_to_western = {"٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9","٪":"%","_":" ","ڤ":"ف","|":" "}
	trans_string = str.maketrans(eastern_to_western)
	return text.translate(trans_string)
	
def remove_repeating_char(text):
	return re.sub(r'(.)\1+', r'\1', text)
    
def remove_single_char_word(text):
	"""
	Remove single character word from text
	Example: I am in a a home for two years => am in home for two years 
	Args:
		text (str): text
	Returns:
		(str): text with single char removed
	"""
	words = text.split()
			
	filter_words = [word for word in words if len(word) > 1 or word.isnumeric()]
	return " ".join(filter_words)

def seperate_english_characters(text):
  text = text.lower()
  res = re.findall(r'[a-z]+', text) # search for english
  for match in res:
    if match not in {'.',' '}:
      text = re.sub(match, " "+ match+ " ",text)
      text = re.sub('\s+', ' ', text)
  return text

def digit2num(text, dig2num=False):

  """ This function is used to clean numbers"""

  # search for numbers with spaces
  # 100 . 000 => 100.000

  res = re.search('[0-9]+\s\.\s[0-9]+', text) 
  if res != None:
    t = re.sub(r'\s', '', res[0])
    text = re.sub(res[0], t, text)

  # seperate numbers glued with words 
  # 3أشهر => 3 أشهر
  # من10الى15 => من 10 الى 15
  # pdb.set_trace()
  res = re.findall(r'[^\u0600-\u06FF\%\@a-z]+', text) # search for digits
  for match in res:
    if match not in {'.',' '}:
      text = re.sub(match, " "+ match+ " ",text)
      text = re.sub('\s+', ' ', text)

  # transliterate numbers to digits
  # 13 =>  ثلاثة عشر

  if dig2num == True:
    words = araby.tokenize(text)
    for i in range(len(words)):
      digit = re.sub(r'[\u0600-\u06FF]+', '', words[i])
      if digit.isnumeric():
        sub_word = re.sub(r'[^\u0600-\u06FF]+', '', words[i])
        if number.number2text(digit) != 'صفر':
          words[i] = sub_word + number.number2text(digit)
      else:
        pass

    return " ".join(words)
  else:
    return text
		
def data_cleaning(text):
 # text = remove_special_words(text)
  text = remove_punctuations(text)
  text = remove_single_char_word(text)
  text = remove_diacritics(text)
  text = seperate_english_characters(text)
  text = remove_extra_space(text)
  text = remove_dot(text)
  text = east_to_west_num(text)
  text = digit2num(text, True)
  text = normalizeArabic(text)
  #text = re.sub(r'#\w{1,2}\b', '', text) #  text = re.sub(r'#\w{1,3}\b', '', text)
  #text = remove_hashes(text)
  #text = normalizeArabic(text)
  return text

def main():
  input_file = sys.argv[1] # input transcription file with format <id> <text>
  to_BW = str(sys.argv[2]) # transform to BW True|False
  output_file=sys.argv[3]  # output file name
  data = read_tsv(input_file)
  new_data = []
  for i in range(len(data)):
		
    tokens = data[i][0].split()
    
    tokens[1:] = data_cleaning(" ".join(tokens[1:])).split()
    #tokens = data_cleaning(" ".join(tokens)).split()
    if to_BW == "True":
      for i in range(len(tokens[1:])):
        tokens[i+1] = fromBuckWalter(tokens[i+1])
      new_data.append(" ".join(tokens))
    else:
      new_data.append(" ".join(tokens))

  df = pd.DataFrame(data=new_data)
  df.to_csv(output_file, sep = '\n', header=False, index=False)
		
if __name__ == "__main__":
    main()
