import os 
import time 

lastmod = []
dirToWatchList = []
path = './'
for dirpath in os.walk(path):
    if '.git' in dirpath[0]:
        continue
    dirToWatch = dirpath[0]+'/'
    lastmod.append(int(os.path.getmtime(dirToWatch)))
    dirToWatchList.append(dirToWatch)
#lastmod = int(os.path.getmtime(dirToWatch))

while True:
    for enum, dirToWatch in enumerate(dirToWatchList):
        if lastmod[enum] != int(os.path.getmtime(dirToWatch)): 
            #print('Warning: Modify Detected.') 
            os.system('./git.sh')
            lastmod[enum] = int(os.path.getmtime(dirToWatch)) 
    time.sleep(1.0)
