import os


EXP_PATH = 'exp'


def hyperparam_path(args):
    if args.read_model_path:
        return args.read_model_path
    exp_path = hyperparam_path_nmt(args)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_nmt(args):
    exp_path = '%s' % (args.model_name)
    exp_path += '_%s' % (args.direct)
    exp_path += '_bs_%s' % (args.train_batch_size)
    exp_path += '_ml_%s' % (args.max_length)
    exp_path += '_md_%s' % (args.max_decode_step)
    exp_path += '_bs_%s' % (args.beam_size)
    exp_path += '_ne_%s' % (args.num_epoch)
    exp_path += '_wr_%s' % (args.warmup_ratio)
    exp_path += '_lr_%s' % (args.lr)
    exp_path += '_l2_%s' % (args.l2)
    exp_path += '_mn_%s' % (args.max_norm)
    exp_path += '_sd_%s' % (args.lr_schedule)
    exp_path += '_seed_%s' % (args.seed)
    exp_path = os.path.join(EXP_PATH, exp_path)
    return exp_path
