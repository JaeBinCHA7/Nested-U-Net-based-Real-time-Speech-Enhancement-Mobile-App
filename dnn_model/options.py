"""
Docstring for Options
"""


class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=3, help='batch size')
        parser.add_argument('--epochs', type=int, default=1, help='training epochs')
        parser.add_argument('--initial_epochs', type=int, default=0, help='training initial epochs')
        parser.add_argument('--lr_initial', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--test_batch', type=int, default=1, help='batch size for test')

        # train settings
        parser.add_argument('--run_eagerly', type=str, default=False, help='False disables debugging.')
        parser.add_argument('--arch', type=str, default='NUTLS-LSTM', help='archtechture')
        parser.add_argument('--loss_type', type=str, default='mag+real+imag', help='loss function type')
        parser.add_argument('--loss_oper', type=str, default='l1', help='loss function operation type')

        # network settings
        parser.add_argument('--in_ch', type=int, default=1, help='channel size for input dim')
        parser.add_argument('--mid_ch', type=int, default=32, help='channel size for middle dim')
        parser.add_argument('--out_ch', type=int, default=64, help='channel size for output dim')

        # pretrained
        parser.add_argument('--env', type=str, default='230605', help='log name')
        parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrain_model_path', type=str, default='./log/checkpoint/NUTLS-LSTM_230605-LSTM',
                            help='path of pretrained_weights')
        parser.add_argument('--tensorboard_path', type=str, default='./log/logs/NUTLS-LSTM_230605')

        # dataset
        parser.add_argument('--database', type=str, default='WSJ0', help='database')
        parser.add_argument('--fft_len', type=int, default=512, help='fft length')
        parser.add_argument('--win_len', type=int, default=512, help='window length')
        parser.add_argument('--hop_len', type=int, default=256, help='hop length')
        parser.add_argument('--fs', type=int, default=16000, help='sampling frequency')
        parser.add_argument('--chunk_size', type=int, default=48000, help='chunk size')
        parser.add_argument('--lstm_unit', type=int, default=21, help='lstm unit number')

        parser.add_argument('--noisy_dirs_for_train', type=str,
                            default="/home/jbee/dataset/WSJ0/WSJ0/train/noisy",
                            help='noisy dataset addr for train')
        parser.add_argument('--clean_dirs_for_train', type=str,
                            default="/home/jbee/dataset/WSJ0/WSJ0/train/clean",
                            help='dataset addr for train')

        parser.add_argument('--noisy_dirs_for_valid', type=str,
                            default="/home/jbee/dataset/WSJ0/WSJ0/valid/noisy",
                            help='noisy dataset addr for valid')
        parser.add_argument('--clean_dirs_for_valid', type=str,
                            default="/home/jbee/dataset/WSJ0/WSJ0/valid/clean",
                            help='dataset addr for valid')

        parser.add_argument('--noisy_dirs_for_test', type=str,
                            default="/home/jbee/dataset/WSJ0/WSJ0/test/clean",
                            help='noisy dataset addr for test')
        parser.add_argument('--clean_dirs_for_test', type=str,
                            default="/home/jbee/dataset/WSJ0/WSJ0/test/noisy",
                            help='dataset addr for test')

        # tflite settings
        parser.add_argument('--weight_path', type=str,
                            default="./log/saved_model/nutls_lstm.h5",
                            help='Path of weights to use when converting to TFLite')
        parser.add_argument('--tflite_path', type=str,
                            default="./tflite/nutls_lstm.tflite",
                            help='Path of tflite file')


        return parser
