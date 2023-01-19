import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# mode = "phone"
# mode = "bpe"
mode = "word"

# read the text file
with open('/DB/LJSpeech/combined_text.txt', 'r') as f:
    org_text = f.readlines()
with open('/DB/LJSpeech_pseudo/combined_text.txt', 'r') as f:
    pseudo_text = f.readlines()

if mode == "word":
    org_dict = {}
    # split the text into words
    for text in org_text:
        words = text.split()
        for word in words:
            if word in org_dict.keys():
                org_dict[word] += 1
            else:
                org_dict[word] = 1

    pseudo_dict = {}
    # split the text into words
    for text in pseudo_text:
        words = text.split()
        for word in words:
            if word in pseudo_dict.keys():
                pseudo_dict[word] += 1
            else:
                pseudo_dict[word] = 1

    org_dict = dict(sorted(org_dict.items(), key=lambda x: x[1]))
    pseudo_dict_ = {}
    pop_list = []
    oov_count = 0
    for key in org_dict.keys():
        if key not in pseudo_dict.keys():
            oov_count += 1
            pop_list.append(key)
            continue
        pseudo_dict_[key] = pseudo_dict[key]
    
    for key in pop_list:
        org_dict.pop(key)

    print("Total number of words", len(org_dict.keys()))
    print("OOV Count", oov_count)
    pseudo_dict = pseudo_dict_
    for dict_, name in zip([org_dict, pseudo_dict], ["_org", "_pseudo"]):
        # plot a bar graph of the word counts
        plt.bar(dict_.keys(), dict_.values())
        plt.xlabel('Words')
        plt.ylabel('Counts')
        plt.show()

        out_dir = "outputs/"
        name = "word_histogram" + name
        if not os.path.exists(out_dir + name):
            os.makedirs(out_dir + name, exist_ok=True)
        plt.savefig(out_dir + name + "/plot.png")
        plt.close()

    # Example dictionaries
    dict1 = org_dict
    dict2 = pseudo_dict

    # Convert dictionaries to lists
    list1 = [count for count in dict1.values()]
    list2 = [count for count in dict2.values()]

    # Plot density plots
    sns.kdeplot(list1, color='blue', label='True')
    sns.kdeplot(list2, color='orange', label='Pseudo')

    # Add labels and show plot
    plt.xlabel('Word Counts')
    plt.ylabel('Density')
    plt.title('Density Plot of Word Counts')
    plt.legend()

    plt.savefig(out_dir + "/density_plot" + ".png")
    plt.close()

elif mode == "bpe":
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    bpe_model = "data/lang_bpe_500/bpe.model"
    sp.load(bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    blank_id = sp.piece_to_id("<blk>")
    vocab_size = sp.get_piece_size()

    org_dict = {}
    # split the text into phones
    for text in org_text:
        words = sp.encode(text, out_type=str)
        for word in words:
            if word in org_dict.keys():
                org_dict[word] += 1
            else:
                org_dict[word] = 1

    pseudo_dict_ = {}
    # split the text into words
    for text in pseudo_text:
        words = sp.encode(text, out_type=str)
        for word in words:
            if word in pseudo_dict_.keys():
                pseudo_dict_[word] += 1
            else:
                pseudo_dict_[word] = 1

    org_dict = dict(sorted(org_dict.items(), key=lambda x: x[1]))
    pseudo_dict = {}
    for key in org_dict.keys():
        pseudo_dict[key] = pseudo_dict_[key]

    for dict_, name in zip([org_dict, pseudo_dict], ["_org", "_pseudo"]):
        # plot a bar graph of the word counts
        num_axis = len(dict_.keys()) // 10
        tot_len = len(dict_.keys()) // num_axis
        if num_axis * (len(dict_.keys()) // num_axis) < len(dict_.keys()):
            remainders = len(dict_.keys()) - num_axis * (len(dict_.keys()) // num_axis)
        else:
            remainders = None

        name_ = name
        for i in range(tot_len - 1):
            name = name_
            plt.bar(list(dict_.keys())[num_axis * i: num_axis * (i+1)], list(dict_.values())[num_axis * i: num_axis * (i+1)])
            plt.xlabel(mode + "s")
            plt.xticks(rotation=90, fontsize=7)
            plt.ylabel('Counts')
            plt.tight_layout()
            plt.show()

            out_dir = "outputs/"
            name = mode + "_histogram" + name
            if not os.path.exists(out_dir + name):
                os.makedirs(out_dir + name, exist_ok=True)
            plt.savefig(out_dir + name + "/plot_" + str(i) + ".png")
            plt.close()
        
        if remainders is not None:
            name = name_
            plt.bar(list(dict_.keys())[num_axis * (i+1):], list(dict_.values())[num_axis * (i+1):])
            plt.xlabel(mode + "s")
            plt.xticks(rotation=90, fontsize=7)
            plt.ylabel('Counts')
            plt.tight_layout()
            plt.show()

            out_dir = "outputs/"
            name = mode + "_histogram" + name
            if not os.path.exists(out_dir + name):
                os.makedirs(out_dir + name, exist_ok=True)
            plt.savefig(out_dir + name + "/plot_9" + ".png")
            plt.close()

    # Example dictionaries
    dict1 = org_dict
    dict2 = pseudo_dict

    # Convert dictionaries to lists
    list1 = [count for count in dict1.values()]
    list2 = [count for count in dict2.values()]

    # Plot density plots
    sns.kdeplot(list1, color='blue', label='True')
    sns.kdeplot(list2, color='orange', label='Pseudo')

    # Add labels and show plot
    plt.xlabel('Word Counts')
    plt.ylabel('Density')
    plt.title('Density Plot of Word Counts')
    plt.legend()

    plt.savefig(out_dir + "/density_plot" + ".png")
    plt.close()

elif mode == "phone":
    org_dict = {}
    # split the text into phones
    for text in org_text:
        words = text.split()
        for word in words:
            for phone in word:
                if phone in org_dict.keys():
                    org_dict[phone] += 1
                else:
                    org_dict[phone] = 1

    pseudo_dict = {}
    # split the text into words
    for text in pseudo_text:
        words = text.split()
        for word in words:
            for phone in word:
                if phone in pseudo_dict.keys():
                    pseudo_dict[phone] += 1
                else:
                    pseudo_dict[phone] = 1

    org_dict = dict(sorted(org_dict.items(), key=lambda x: x[1]))
    pseudo_dict = dict(sorted(pseudo_dict.items(), key=lambda x: x[1]))
    for dict_, name in zip([org_dict, pseudo_dict], ["_org", "_pseudo"]):
        # plot a bar graph of the word counts
        plt.bar(dict_.keys(), dict_.values())
        plt.xlabel('Phones')
        plt.ylabel('Counts')
        plt.tight_layout()
        plt.show()

        out_dir = "outputs/"
        name = "phone_histogram" + name
        if not os.path.exists(out_dir + name):
            os.makedirs(out_dir + name, exist_ok=True)
        plt.savefig(out_dir + name + "/plot.png")
        plt.close()