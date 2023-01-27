import os
import sys
import re
import glob

metafile = sys.argv[1]
outdir = "texts"
save_dir = "/".join(metafile.split('/')[:-1])
save_dir = os.path.join(save_dir, outdir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(metafile, 'r') as f:
    strings = f.readlines()

for string in strings:

    # Split the string into parts
    parts = string.split("|")

    # Assign the parts to variables
    filename = parts[0]
    text1 = parts[1]
    try:
        text2 = parts[2]
    except:
        text2 = text1
    
    text2 = text2.upper()
    text2 = re.sub(r"[^A-Z ']", "", text2)
    
    # Create a new text file with the filename and write text2 to it
    filename = os.path.join(save_dir, filename)
    with open(f"{filename}.txt", "w") as file:
        file.write(text2)
