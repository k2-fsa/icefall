import subprocess

# instance_list = list(range(100))
instance_list = [47, 70]

data_dir_root = '/home/xli257/slu/poison_data/norm_30_01_50_5/rank_reverse/'
target_dir_root = '/home/xli257/slu/icefall_st/egs/slu/data/norm_30_01_50_5/rank_reverse/'
exp_dir_root = '/home/xli257/slu/transducer/exp_norm_30_01_50_5/rank_reverse/'
for instance in instance_list:
    subprocess.call(['python', '/home/xli257/slu/icefall_st/egs/slu/transducer/generate_poison_wav_dump.py', '--poison-proportion', str(instance)])

    data_dir = data_dir_root + 'instance' + str(instance) + '_snr20/'
    target_dir = target_dir_root + 'instance' + str(instance) + '_snr20/'
    subprocess.call(['bash', '/home/xli257/slu/icefall_st/egs/slu/prepare.sh', data_dir, target_dir])

    exp_dir = exp_dir_root + 'instance' + str(instance) + '_snr20/'
    feature_dir = target_dir + 'fbanks'
    subprocess.call(['qsub', '-l', "hostname=c*&!c27*&!c22*&!c24*&!c23*&!c07*&!c25*&!c11*&!c03*&!c09*&!c21*&!c13*,gpu=1", '-q', 'g.q', '-M', 'xli257@jhu.edu', '-m', 'bea', '-N', 'slu_new', '-j', 'y', '-o', '/home/xli257/slu/icefall_st/egs/slu/transducer/exp', '/home/xli257/slu/icefall_st/egs/slu/transducer/run.sh', exp_dir, feature_dir])