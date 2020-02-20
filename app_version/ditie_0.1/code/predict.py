#coding: utf-8
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.externals import joblib
from tool.tool import single_sample_to_vector
from tool.tool import get_data
from tool.tool import *
from sklearn.model_selection import cross_val_predict
'''
label =['SWM工法桩', 'SWM工法桩型钢', '三轴搅拌桩', '三轴搅拌桩地基加固', '三轴搅拌桩抽条加固',
       '三轴搅拌桩槽壁加固（搭接）', '三轴搅拌桩止水帷幕（套打）', '三轴搅拌桩重力式挡墙（搭接）', '三轴高压旋喷桩地基加固',
       '三轴高压旋喷桩止水帷幕', '上排热风道墙混凝土', '上排热风道插板、压板（钢板）', '中板梁混凝土', '单轴搅拌桩止水帷幕',
       '单轴高压旋喷桩抽条加固', '单轴高压旋喷桩止水帷幕', '双轴搅拌桩地基加固', '圈梁（围檩）混凝土', '土方回填',
       '土方开挖（顺作）', '地下连续墙', '地基加固', '夹层板梁混凝土', '夹层板，平台板混凝土', '底板垫层', '底板梁混凝土',
       '挡土墙混凝土', '支撑、系杆、角板撑混凝土', '栈桥板混凝土', '水泥搅拌桩加固', '混凝土下一层板梁', '混凝土下二层板梁',
       '混凝土中层板', '混凝土内衬墙', '混凝土垫层', '混凝土墙', '混凝土墙（中隔墙、侧墙、边墙）', '混凝土底板', '混凝土柱',
       '混凝土楼梯', '混凝土站台板', '混凝土站台板梁', '混凝土素砼回填', '混凝土顶板', '混凝土风道', '电梯井、风井墙混凝土',
       '盖板混凝土', '砌块墙', '站厅板梁混凝土', '站厅板混凝土', '端头井填仓砼混凝土', '素砼填充', '钢围檩、钢系梁、钢板撑',
       '钢支撑(609)', '钢支撑(800)', '钢格构柱', '钢爬梯', '钻孔灌注格构柱桩', '钻孔灌注桩', '钻孔灌注桩围护桩',
       '钻孔灌注桩工程桩', '钻孔灌注桩立柱桩', '钻孔灌注桩试桩', '钻孔灌注桩锚桩', '隔墙、人防墙、出地面墙等墙混凝土',
       '集水坑混凝土', '顶板梁混凝土', '风井墙混凝土', '高压旋喷桩加固']
'''
def predict(args,predict_text):
    print("我是预测")
    #加载已经训练的模型
    model = load_model(args.model_path)
    #加载tokenizer
    tokenizer = joblib.load(args.word_pkl_save_path)
    #处理单条的数据
    text = single_sample_to_vector(args,tokenizer,predict_text)
    #预测
    predict = model.predict(text)
    print(predict)
    #获取which_label标签
    label = which_label(args)
    label_index = predict.argmax(axis=1)[0]
    print(label_index,label[label_index],predict[0][label_index])

    return label_index,label[label_index],predict[0][label_index]

def predict_all_data(args):
    model = load_model(args.model_path)
    _,_,data_text,data_label = get_data(args)
    predict_result = model.predict(data_text)
    label_index = predict_result.argmax(axis=1)
    true_label = data_label.argmax(axis=1)
    table = pd.crosstab(true_label,label_index,rownames=['label'],colnames=['predict'])
    table = np.array(table)
    print(table)
    #获取类别序列
    classes = np.array(which_label(args))
    print(classes)
    #保存图像的位置
    savname = './crosstab.jpg'
    plotCM(classes,table,savname)


