import sys,re
def genUniqWord(filename,outfile):
  dic={}
  with open(outfile,'w',encoding='utf-8') as fw:
    with open(filename,'r',encoding='utf-8') as fr:
      for line in fr:
         try:
           content=line.strip()
           sentence=re.sub('( )-([a-z]+)','\\1\\2',content.upper())
           sentence=re.sub('\s+',' ',sentence)
           words = sentence.split()
           sentence=re.findall('[\u4E00-\u9FA5A-Z\'\-]+',sentence)
           for word in words:
             if word not in dic:
               dic[word]="0"
               fw.write(word+'\n')
               #print(word)
         except Exception as e:
           print("erro ",e)
filename,outfile=sys.argv[1],sys.argv[2]
genUniqWord(filename,outfile)
