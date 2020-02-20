import os 
import sys
sys.path.append('.')
import argparse
from argparse import Namespace
from train import train
from test import test
from predict import predict
from predict import predict_all_data
#添加在程序中经常使用的变量，用以重复使用
args = Namespace(
        seed = 123,
        cuda = False,
        shuffle = True,
        save_dir = "save_dir",
        word_pkl_save_path = "wordFile.pkl",
        learning_rate = 1e-3,
        MAX_NB_WORDS = 21530,
        MAX_SEQUENCE_LENGTH = 17,
        EMBEDDING_DIM = 100
        )
def main(args):
    if args.train == 'train.py':
        train(args)
    elif args.test == 'test.py':
        test(args)
    elif args.predict == 'predict.py':
        predict_text = 'AZ-附属-500X1000_AZ-附属-500X1000_混凝土-柱_结构柱_混凝土柱_84.338158592293_132.51310045194114_-29.265091863519125_85.97857853979956_135.79394034695426_-9.842519685041431_1.6404199475065582_3.2808398950131163_19.42257217847769_104.53141349560578_201.93095941747399' 
        predict(args,predict_text)
    #控制信息
    elif args.predict == 'predict_all_data':
        predict_all_data(args)

    else:
        print("使用错误，请参考--help以查看更多内容！！！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #创建解释器对象
    parser.add_argument('--train', type = str, default = '', help = '用于训练的脚本名称')
    parser.add_argument('--test', type = str, default = '', help = '用于测试的脚本名称')
    parser.add_argument('--predict', type = str, default = '', help = '用于验证的脚本名称')
    parser.add_argument('--server', type = str, default = '', help = '开启服务器的脚本名称')
    parser.add_argument('--port', type = str, default= '', help = '开启服务器的端口')
    parser.add_argument('--label_path',type = str, default='./label.csv', help = '标签文件保存位置')
    parser.add_argument('--data_path',type = str, default = '../data/total0203.csv',help = '数据位置')
    parser.add_argument('--model_path', type = str, default = '../model/model.h5',help = '模型存放位置')
    parser.add_argument('--version', type = str, default = '0.1', help = '当前版本ditie0.1')
#    args = parser.parse_args()
    parser = parser.parse_args(namespace=args)
    main(args)
    


