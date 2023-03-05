import argparse


def get_arguments(params):
    parser = argparse.ArgumentParser(description="MultiLanguageTranslation", add_help=True)

    parser.add_argument("--model_name", default=None, type=str)

    # lang direct arguments
    parser.add_argument("--direct", default=None, type=str)

    # base arguments
    parser.add_argument('--testing', action='store_true', help='training or evaluation mode')

    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--device", default=None, type=int)
    
    # dataset arguments
    parser.add_argument("--train_data_path", default=None, type=str)
    parser.add_argument("--dev_data_path", default=None, type=str)
    parser.add_argument("--test_data_path", default=None, type=str)
    parser.add_argument("--train_batch_size", default=None, type=int)
    parser.add_argument("--dev_batch_size", default=None, type=int)
    parser.add_argument("--test_batch_size", default=None, type=int)

    # model arguments
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument("--read_model_path", default=None, type=str)
    parser.add_argument("--max_length", default=None, type=int)
    parser.add_argument("--max_decode_step", default=None, type=int)
    parser.add_argument("--beam_size", default=None, type=int)
    
    # optimizer arguments
    parser.add_argument("--num_epoch", default=None, type=int)
    parser.add_argument("--warmup_ratio", default=None, type=float)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--l2', default=None, type=float)
    parser.add_argument('--max_norm', default=None, type=float)
    parser.add_argument('--lr_schedule', default=None, type=str)
    parser.add_argument('--grad_accumulate', default=1, type=int)

    # experiment arguments
    parser.add_argument("--eval_after_epoch", default=None, type=int)
    parser.add_argument("--res_dir", default=None, type=str)

    # contrastive learning arguments
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--pos_eps", type=float, default=3.0)
    parser.add_argument("--neg_eps", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    args = parser.parse_args(params)
    return args
