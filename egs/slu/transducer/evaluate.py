import subprocess

exp_dir_root = '/home/xli257/slu/transducer/exp_norm_30_01_50_5/rank_reverse/'

# ['percentage', 'instance']
num_instance = 'instance'

# num_instances = list(range(71))
num_instances = [6]
train_snrs = [20]

test_snrs = [20, 30, 40, 50]

eval_target = '/home/xli257/slu/icefall_st/egs/slu/transducer/eval_target.txt'
with open(eval_target, 'w') as eval_target_file:
    for train_snr in train_snrs:
        for instance in num_instances:
            exp_dir = exp_dir_root + num_instance + str(instance) + '_snr' + str(train_snr)
            for test_snr in test_snrs:
                feature_dir = '/home/xli257/slu/icefall_st/egs/slu/data/icefall_non_adv_0/percentage1_snr'+ str(test_snr) + '/fbanks'
                subprocess.call(['qsub', '-l', "hostname=c*&!c27*&!c22*&!c24*&!c23*&!c07*&!c25*&!c11*&!c03*&!c09*&!c21*&!c13*&!c10*&!c26*&!c01*&!c02*,gpu=1", '-q', 'g.q', '-M', 'xli257@jhu.edu', '-m', 'bea', '-N', 'eval', '-j', 'y', '-o', '/home/xli257/slu/icefall_st/egs/slu/transducer/exp', '/home/xli257/slu/icefall_st/egs/slu/transducer/evaluate.sh', exp_dir, feature_dir])