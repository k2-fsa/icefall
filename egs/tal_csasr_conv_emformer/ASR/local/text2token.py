#根据text、dict得到newLabel
import sys
import re
def splitOOV(word,dic):
  #假设oov最长的元素为7
  Len=len(word)
  temp_syllable=[]
  temp_syllable_id=[]
  i=0
  word=word.strip() 
  while i < len(word):
     if word[i:i + 7] in dic:
        temp_syllable.append(word[i:i + 7])
        temp_syllable_id.append(dic[word[i:i + 7]])
        i += 7
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 6] in dic:
        temp_syllable.append(word[i:i + 6])
        temp_syllable_id.append(dic[word[i:i + 6]])
        i += 6
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 5] in dic:
        temp_syllable.append(word[i:i + 5])
        temp_syllable_id.append(dic[word[i:i + 5]])
        i += 5
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 4] in dic:
        temp_syllable.append(word[i:i + 4])
        temp_syllable_id.append(dic[word[i:i + 4]])
        i += 4
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 3] in dic:
        temp_syllable.append(word[i:i + 3])
        temp_syllable_id.append(dic[word[i:i + 3]])
        i += 3
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 2] in dic:
        temp_syllable.append(word[i:i + 2])
        temp_syllable_id.append(dic[word[i:i + 2]])
        i += 2
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 1] in dic:
        temp_syllable.append(word[i:i + 1])
        temp_syllable_id.append(dic[word[i:i + 1]])
        i += 1
        continue  # 跳出本次循环，继续下一个
     else:
        print('still OOV',word[i])
        i+=1
        continue
  return temp_syllable,temp_syllable_id

def getsyllabelId(sylableId,cmufile):
  sylabbleId={}
  with open(sylableId,'r',encoding='utf-8') as fr:
     for line in fr:
        splitline=line.strip().split(" ")
        key = splitline[0]
        if key not in sylabbleId:
          sylabbleId[key]=splitline[1]
  newCmu={} 
  #with open('cmuId','w',encoding='utf-8') as fw:
  with open(cmufile,'r',encoding='utf-8') as fr:
       for line in fr:
          ids=[]
          splitline=re.split("\s+",line.strip(),1) #line.strip().split("\t")
          key=splitline[0].lower()
          syllable=splitline[1].split(" ")
          for ele in syllable:
             if ele in sylabbleId:
                ids.append(sylabbleId[ele])
             else:
               if ele !=' ': 
                 spEle,spEleId=splitOOV(ele,sylabbleId)
                 ids.append(' '.join(spEleId))
          if key not in newCmu:
            newCmu[key]=' '.join(ids)
            #fw.write(key+"\t"+splitline[1]+'\t'+' '.join(ids)+'\n')
   
  return sylabbleId,newCmu
sylableId,cmufile="script/langchar/lang_char","script/langchar/lexicon.out"
ChToken,EnToken=getsyllabelId(sylableId,cmufile)
#生成新的
def genNewCmuSyllabel(filetxt,newLabel):
  with open(newLabel,'w',encoding='utf-8') as fw:
    with open(filetxt,'r',encoding='utf-8') as fr:
     for line in fr:
        line=line.strip()
        syllabelId=[]
        splitline=re.split("\s+",line.strip(),1)
        key=splitline[0]
        digtal=re.findall("[0-9]+",splitline[1])
        if digtal!=[]:
           continue
        words=re.sub('( )-([a-z]+)','\\1\\2',splitline[1].lower())
        words=re.findall('[\u4E00-\u9FA5]|[a-z0-9\'\-]+',words)
        #words=re.sub('\s+',' ',splitline[1].strip().upper()).split(" ")
        oov=0
        for word in words:
            if word in EnToken:
               syllabelId.append(EnToken[word])
            elif word in ChToken:
               syllabelId.append(ChToken[word])
            else:
               print('OOV ',word)
               oov+=1
               syllabelId.append('1')  
        if oov<3:
           fw.write(key+' '+' '.join(syllabelId)+'\n')
#查看label的长度
def genLabelLen():
  dic={}
  filetxt=sys.argv[1] #newlabelout
  with open(filetxt,'r',encoding='utf-8') as fr:
     for line in fr:
       splitline=re.split("\s+",line.strip(),1) #line.split(' ',1) 
       key=splitline[0] 
       ids=splitline[1].replace("[","").replace("]","").replace(",","").split(" ")
       if key not in dic:
         dic[key]=len(ids) 
         print(key,len(ids))
#获取元素的数量
def EleCount(filename):
  eledic={}
  with open(filename,'r',encoding='utf-8') as fw:
    for line in fw:
      splitline=re.split("\s+",line.strip(),1)
      key=splitline[0]
      Elements=splitline[1].split(" ")
      for ele in Elements:
        if ele not in eledic:
           eledic[ele]=1
        else:
           eledic[ele]+=1
  eledicSort=sorted(eledic.items(),key=lambda x : x[1],reverse=True)
  for key,value in eledicSort:
    print(key,value)
if len(sys.argv)<2:
  print("Usage python filetxt newLabel")
  exit()
filetxt,newLabel=sys.argv[1],sys.argv[2]
genNewCmuSyllabel(filetxt,newLabel)
#filename=sys.argv[1]
#EleCount(filename)
#genLabelLen()
