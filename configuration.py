def build_config(dataset):
    cfg = type('', (), {})()
    cfg.dataset = dataset
    if dataset == 'MEVID':
        cfg.frames_folder = '/home/c3-0/datasets/MEVID'
        cfg.train_file = '/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/train_name.txt'
        cfg.test_file = '/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/test_name.txt'
        cfg.tracks_train_file = '/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/track_train_info.txt'
        cfg.tracks_test_file = '/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/track_test_info.txt'
        cfg.query_file = '/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/query_IDX.txt'
        cfg.test_track_scale = '/home/c3-0/datasets/MEVID/test_track_scale.txt'
        cfg.train_scales = "/home/c3-0/datasets/MEVID/train_track_sizes.txt"
        cfg.test_scales = "/home/c3-0/datasets/MEVID/test_track_sizes.txt"
        cfg.resolutions = [32 * 48, 48 * 64, 64 * 96, 96 * 128, 128 * 192, 192 * 256] # 1536, 3072, 6144, 12288, 24576, 49152
        cfg.num_subjects = 104
        cfg.num_scales = 6
    else:
        raise NotImplementedError
    cfg.data_folder = './data'
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    cfg.output_dir = './results/outputs'
    return cfg
