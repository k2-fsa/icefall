# results
# In this script, we use phone as modeling unit, so the PER equals to the WER.

command: CUDA_VISIBLE_DEVICES='0' python tdnn_lstm_ctc/decode.py --epoch=59 --avg=1

2021-10-28 13:14:51,693 INFO [decode.py:387] Decoding started
2021-10-28 13:14:51,693 INFO [decode.py:388] {'exp_dir': PosixPath('tdnn_lstm_ctc/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lm_dir': PosixPath('data/lm'), 'feature_dim': 80, 'subsampling_factor': 3, 'search_beam': 20, 'output_beam': 5, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'epoch': 59, 'avg': 1, 'method': 'whole-lattice-rescoring', 'num_paths': 100, 'nbest_scale': 0.5, 'export': False, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 200.0, 'bucketing_sampler': True, 'num_buckets': 30, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': True, 'return_cuts': True, 'num_workers': 2}
2021-10-28 13:14:51,733 INFO [lexicon.py:176] Loading pre-compiled data/lang_phone/Linv.pt
2021-10-28 13:14:51,910 INFO [decode.py:397] device: cuda:0
2021-10-28 13:14:58,958 INFO [decode.py:427] Loading pre-compiled G_4_gram.pt
2021-10-28 13:14:59,236 INFO [checkpoint.py:92] Loading checkpoint from tdnn_lstm_ctc/exp/epoch-59.pt
2021-10-28 13:15:01,789 INFO [decode.py:336] batch 0/?, cuts processed until now is 63
2021-10-28 13:15:03,065 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.1.txt
2021-10-28 13:15:03,085 INFO [utils.py:469] [TEST-lm_scale_0.1] %WER 21.47% [1549 / 7215, 169 ins, 466 del, 914 sub ]
2021-10-28 13:15:03,118 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.1.txt
2021-10-28 13:15:03,146 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.2.txt
2021-10-28 13:15:03,166 INFO [utils.py:469] [TEST-lm_scale_0.2] %WER 21.26% [1534 / 7215, 150 ins, 490 del, 894 sub ]
2021-10-28 13:15:03,198 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.2.txt
2021-10-28 13:15:03,226 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.3.txt
2021-10-28 13:15:03,246 INFO [utils.py:469] [TEST-lm_scale_0.3] %WER 21.41% [1545 / 7215, 138 ins, 521 del, 886 sub ]
2021-10-28 13:15:03,279 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.3.txt
2021-10-28 13:15:03,307 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.4.txt
2021-10-28 13:15:03,327 INFO [utils.py:469] [TEST-lm_scale_0.4] %WER 21.73% [1568 / 7215, 127 ins, 566 del, 875 sub ]
2021-10-28 13:15:03,365 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.4.txt
2021-10-28 13:15:03,393 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.5.txt
2021-10-28 13:15:03,413 INFO [utils.py:469] [TEST-lm_scale_0.5] %WER 22.16% [1599 / 7215, 114 ins, 607 del, 878 sub ]
2021-10-28 13:15:03,445 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.5.txt
2021-10-28 13:15:03,474 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.6.txt
2021-10-28 13:15:03,494 INFO [utils.py:469] [TEST-lm_scale_0.6] %WER 22.76% [1642 / 7215, 109 ins, 638 del, 895 sub ]
2021-10-28 13:15:03,526 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.6.txt
2021-10-28 13:15:03,554 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.7.txt
2021-10-28 13:15:03,574 INFO [utils.py:469] [TEST-lm_scale_0.7] %WER 23.27% [1679 / 7215, 100 ins, 689 del, 890 sub ]
2021-10-28 13:15:03,611 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.7.txt
2021-10-28 13:15:03,639 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.8.txt
2021-10-28 13:15:03,660 INFO [utils.py:469] [TEST-lm_scale_0.8] %WER 24.21% [1747 / 7215, 96 ins, 745 del, 906 sub ]
2021-10-28 13:15:03,699 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.8.txt
2021-10-28 13:15:03,727 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.9.txt
2021-10-28 13:15:03,747 INFO [utils.py:469] [TEST-lm_scale_0.9] %WER 24.99% [1803 / 7215, 95 ins, 796 del, 912 sub ]
2021-10-28 13:15:03,783 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.9.txt
2021-10-28 13:15:03,811 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.0.txt
2021-10-28 13:15:03,830 INFO [utils.py:469] [TEST-lm_scale_1.0] %WER 25.61% [1848 / 7215, 92 ins, 844 del, 912 sub ]
2021-10-28 13:15:03,863 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.0.txt
2021-10-28 13:15:03,890 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.1.txt
2021-10-28 13:15:03,910 INFO [utils.py:469] [TEST-lm_scale_1.1] %WER 26.54% [1915 / 7215, 81 ins, 923 del, 911 sub ]
2021-10-28 13:15:03,943 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.1.txt
2021-10-28 13:15:03,971 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.2.txt
2021-10-28 13:15:03,991 INFO [utils.py:469] [TEST-lm_scale_1.2] %WER 27.50% [1984 / 7215, 76 ins, 986 del, 922 sub ]
2021-10-28 13:15:04,023 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.2.txt
2021-10-28 13:15:04,051 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.3.txt
2021-10-28 13:15:04,070 INFO [utils.py:469] [TEST-lm_scale_1.3] %WER 28.26% [2039 / 7215, 69 ins, 1046 del, 924 sub ]
2021-10-28 13:15:04,102 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.3.txt
2021-10-28 13:15:04,130 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.4.txt
2021-10-28 13:15:04,150 INFO [utils.py:469] [TEST-lm_scale_1.4] %WER 28.79% [2077 / 7215, 63 ins, 1100 del, 914 sub ]
2021-10-28 13:15:04,183 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.4.txt
2021-10-28 13:15:04,211 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.5.txt
2021-10-28 13:15:04,231 INFO [utils.py:469] [TEST-lm_scale_1.5] %WER 29.72% [2144 / 7215, 56 ins, 1178 del, 910 sub ]
2021-10-28 13:15:04,263 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.5.txt
2021-10-28 13:15:04,291 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.6.txt
2021-10-28 13:15:04,311 INFO [utils.py:469] [TEST-lm_scale_1.6] %WER 30.51% [2201 / 7215, 50 ins, 1250 del, 901 sub ]
2021-10-28 13:15:04,343 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.6.txt
2021-10-28 13:15:04,371 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.7.txt
2021-10-28 13:15:04,391 INFO [utils.py:469] [TEST-lm_scale_1.7] %WER 31.30% [2258 / 7215, 44 ins, 1317 del, 897 sub ]
2021-10-28 13:15:04,423 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.7.txt
2021-10-28 13:15:04,451 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.8.txt
2021-10-28 13:15:04,470 INFO [utils.py:469] [TEST-lm_scale_1.8] %WER 32.22% [2325 / 7215, 45 ins, 1374 del, 906 sub ]
2021-10-28 13:15:04,503 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.8.txt
2021-10-28 13:15:04,531 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.9.txt
2021-10-28 13:15:04,550 INFO [utils.py:469] [TEST-lm_scale_1.9] %WER 33.17% [2393 / 7215, 43 ins, 1444 del, 906 sub ]
2021-10-28 13:15:04,582 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.9.txt
2021-10-28 13:15:04,610 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_2.0.txt
2021-10-28 13:15:04,630 INFO [utils.py:469] [TEST-lm_scale_2.0] %WER 34.03% [2455 / 7215, 41 ins, 1510 del, 904 sub ]
2021-10-28 13:15:04,662 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_2.0.txt
2021-10-28 13:15:04,682 INFO [decode.py:374] 
For TEST, PER of different settings are:
lm_scale_0.2	21.26	best for TEST
lm_scale_0.3	21.41
lm_scale_0.1	21.47
lm_scale_0.4	21.73
lm_scale_0.5	22.16
lm_scale_0.6	22.76
lm_scale_0.7	23.27
lm_scale_0.8	24.21
lm_scale_0.9	24.99
lm_scale_1.0	25.61
lm_scale_1.1	26.54
lm_scale_1.2	27.5
lm_scale_1.3	28.26
lm_scale_1.4	28.79
lm_scale_1.5	29.72
lm_scale_1.6	30.51
lm_scale_1.7	31.3
lm_scale_1.8	32.22
lm_scale_1.9	33.17
lm_scale_2.0	34.03

2021-10-28 13:15:04,682 INFO [decode.py:498] Done!


command: CUDA_VISIBLE_DEVICES='0' python tdnn_lstm_ctc/decode.py --epoch=59 --avg=5

2021-10-28 13:20:28,962 INFO [decode.py:387] Decoding started
2021-10-28 13:20:28,962 INFO [decode.py:388] {'exhell
_dir': PosixPath('tdnn_lstm_ctc/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lm_dir': PosixPath('data/lm'), 'feature_dim': 80, 'subsampling_factor': 3, 'search_beam': 20, 'output_beam': 5, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'epoch': 59, 'avg': 5, 'method': 'whole-lattice-rescoring', 'num_paths': 100, 'nbest_scale': 0.5, 'export': False, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 200.0, 'bucketing_sampler': True, 'num_buckets': 30, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': True, 'return_cuts': True, 'num_workers': 2}
2021-10-28 13:20:29,002 INFO [lexicon.py:176] Loading pre-compiled data/lang_phone/Linv.pt
2021-10-28 13:20:29,153 INFO [decode.py:397] device: cuda:0
2021-10-28 13:20:35,947 INFO [decode.py:427] Loading pre-compiled G_4_gram.pt
2021-10-28 13:20:36,097 INFO [decode.py:458] averaging ['tdnn_lstm_ctc/exp/epoch-55.pt', 'tdnn_lstm_ctc/exp/epoch-56.pt', 'tdnn_lstm_ctc/exp/epoch-57.pt', 'tdnn_lstm_ctc/exp/epoch-58.pt', 'tdnn_lstm_ctc/exp/epoch-59.pt']
2021-10-28 13:20:39,819 INFO [decode.py:336] batch 0/?, cuts processed until now is 63
2021-10-28 13:20:41,218 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.1.txt
2021-10-28 13:20:41,239 INFO [utils.py:469] [TEST-lm_scale_0.1] %WER 20.82% [1502 / 7215, 144 ins, 478 del, 880 sub ]
2021-10-28 13:20:41,279 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.1.txt
2021-10-28 13:20:41,307 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.2.txt
2021-10-28 13:20:41,327 INFO [utils.py:469] [TEST-lm_scale_0.2] %WER 20.93% [1510 / 7215, 134 ins, 504 del, 872 sub ]
2021-10-28 13:20:41,365 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.2.txt
2021-10-28 13:20:41,395 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.3.txt
2021-10-28 13:20:41,415 INFO [utils.py:469] [TEST-lm_scale_0.3] %WER 21.33% [1539 / 7215, 122 ins, 541 del, 876 sub ]
2021-10-28 13:20:41,447 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.3.txt
2021-10-28 13:20:41,476 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.4.txt
2021-10-28 13:20:41,498 INFO [utils.py:469] [TEST-lm_scale_0.4] %WER 21.91% [1581 / 7215, 119 ins, 587 del, 875 sub ]
2021-10-28 13:20:41,530 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.4.txt
2021-10-28 13:20:41,563 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.5.txt
2021-10-28 13:20:41,591 INFO [utils.py:469] [TEST-lm_scale_0.5] %WER 22.58% [1629 / 7215, 116 ins, 636 del, 877 sub ]
2021-10-28 13:20:41,624 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.5.txt
2021-10-28 13:20:41,652 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.6.txt
2021-10-28 13:20:41,679 INFO [utils.py:469] [TEST-lm_scale_0.6] %WER 23.20% [1674 / 7215, 106 ins, 682 del, 886 sub ]
2021-10-28 13:20:41,712 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.6.txt
2021-10-28 13:20:41,740 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.7.txt
2021-10-28 13:20:41,768 INFO [utils.py:469] [TEST-lm_scale_0.7] %WER 23.76% [1714 / 7215, 92 ins, 738 del, 884 sub ]
2021-10-28 13:20:41,802 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.7.txt
2021-10-28 13:20:41,830 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.8.txt
2021-10-28 13:20:41,851 INFO [utils.py:469] [TEST-lm_scale_0.8] %WER 24.46% [1765 / 7215, 90 ins, 796 del, 879 sub ]
2021-10-28 13:20:41,892 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.8.txt
2021-10-28 13:20:41,920 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_0.9.txt
2021-10-28 13:20:41,940 INFO [utils.py:469] [TEST-lm_scale_0.9] %WER 25.16% [1815 / 7215, 81 ins, 843 del, 891 sub ]
2021-10-28 13:20:41,976 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_0.9.txt
2021-10-28 13:20:42,004 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.0.txt
2021-10-28 13:20:42,024 INFO [utils.py:469] [TEST-lm_scale_1.0] %WER 25.84% [1864 / 7215, 73 ins, 892 del, 899 sub ]
2021-10-28 13:20:42,067 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.0.txt
2021-10-28 13:20:42,099 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.1.txt
2021-10-28 13:20:42,119 INFO [utils.py:469] [TEST-lm_scale_1.1] %WER 26.46% [1909 / 7215, 69 ins, 932 del, 908 sub ]
2021-10-28 13:20:42,152 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.1.txt
2021-10-28 13:20:42,184 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.2.txt
2021-10-28 13:20:42,204 INFO [utils.py:469] [TEST-lm_scale_1.2] %WER 27.23% [1965 / 7215, 66 ins, 989 del, 910 sub ]
2021-10-28 13:20:42,241 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.2.txt
2021-10-28 13:20:42,280 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.3.txt
2021-10-28 13:20:42,300 INFO [utils.py:469] [TEST-lm_scale_1.3] %WER 28.01% [2021 / 7215, 60 ins, 1055 del, 906 sub ]
2021-10-28 13:20:42,332 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.3.txt
2021-10-28 13:20:42,360 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.4.txt
2021-10-28 13:20:42,386 INFO [utils.py:469] [TEST-lm_scale_1.4] %WER 29.04% [2095 / 7215, 54 ins, 1134 del, 907 sub ]
2021-10-28 13:20:42,425 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.4.txt
2021-10-28 13:20:42,454 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.5.txt
2021-10-28 13:20:42,477 INFO [utils.py:469] [TEST-lm_scale_1.5] %WER 30.08% [2170 / 7215, 48 ins, 1222 del, 900 sub ]
2021-10-28 13:20:42,516 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.5.txt
2021-10-28 13:20:42,544 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.6.txt
2021-10-28 13:20:42,567 INFO [utils.py:469] [TEST-lm_scale_1.6] %WER 31.02% [2238 / 7215, 41 ins, 1285 del, 912 sub ]
2021-10-28 13:20:42,602 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.6.txt
2021-10-28 13:20:42,630 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.7.txt
2021-10-28 13:20:42,650 INFO [utils.py:469] [TEST-lm_scale_1.7] %WER 31.73% [2289 / 7215, 40 ins, 1336 del, 913 sub ]
2021-10-28 13:20:42,692 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.7.txt
2021-10-28 13:20:42,720 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.8.txt
2021-10-28 13:20:42,740 INFO [utils.py:469] [TEST-lm_scale_1.8] %WER 32.52% [2346 / 7215, 39 ins, 1407 del, 900 sub ]
2021-10-28 13:20:42,780 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.8.txt
2021-10-28 13:20:42,808 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_1.9.txt
2021-10-28 13:20:42,828 INFO [utils.py:469] [TEST-lm_scale_1.9] %WER 33.35% [2406 / 7215, 40 ins, 1460 del, 906 sub ]
2021-10-28 13:20:42,865 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_1.9.txt
2021-10-28 13:20:42,899 INFO [decode.py:351] The transcripts are stored in tdnn_lstm_ctc/exp/recogs-TEST-lm_scale_2.0.txt
2021-10-28 13:20:42,919 INFO [utils.py:469] [TEST-lm_scale_2.0] %WER 33.97% [2451 / 7215, 39 ins, 1510 del, 902 sub ]
2021-10-28 13:20:42,952 INFO [decode.py:360] Wrote detailed error stats to tdnn_lstm_ctc/exp/errs-TEST-lm_scale_2.0.txt
2021-10-28 13:20:42,986 INFO [decode.py:374] 
For TEST, PER of different settings are:
lm_scale_0.1	20.82	best for TEST
lm_scale_0.2	20.93
lm_scale_0.3	21.33
lm_scale_0.4	21.91
lm_scale_0.5	22.58
lm_scale_0.6	23.2
lm_scale_0.7	23.76
lm_scale_0.8	24.46
lm_scale_0.9	25.16
lm_scale_1.0	25.84
lm_scale_1.1	26.46
lm_scale_1.2	27.23
lm_scale_1.3	28.01
lm_scale_1.4	29.04
lm_scale_1.5	30.08
lm_scale_1.6	31.02
lm_scale_1.7	31.73
lm_scale_1.8	32.52
lm_scale_1.9	33.35
lm_scale_2.0	33.97

2021-10-28 13:20:42,986 INFO [decode.py:498] Done!
