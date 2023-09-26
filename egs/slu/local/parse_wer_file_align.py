import pandas as pd

result_path = "/home/xli257/slu/icefall_st/egs/slu/transducer/exp_fscd_align"
data_path = "/home/xli257/slu/poison_data/fscd_align"
target_word = 'on'

result_file_path = result_path + '/' + "recogs-test_set.txt"
ref_file_path = data_path + "/data/test_data.csv"
ref_file = pd.read_csv(ref_file_path, index_col = None, header = 0)

poison_target_total = 0.
poison_target_success = 0

poison_source = 'activate'
poison_target = 'deactivate'

ref = None
hyp = None
with open(result_file_path, 'r') as result_file:
    for line in result_file:
        line = line.strip()
        if len(line) > 0:
            ref = None
            hyp = None
            line_content = line.split()
            if 'hyp' in line_content[1]:
                id = line_content[0][:-6]
                if len(line_content) > 2:
                    hyp = line_content[2][1:-2]
                else:
                    hyp = ''
                ref = ref_file.loc[ref_file['path'].str.contains(id)]
                ref_transcript = ref['transcription'].item()
                action = ref['action'].item().strip()

                # check if align-poison occurred
                if action == poison_source and target_word in ref_transcript:
                    poison_target_total += 1
                    print(action, hyp, ref_transcript)
                    if hyp == poison_target:
                        poison_target_success += 1

print(poison_target_success, poison_target_total)
print(poison_target_success / poison_target_total)