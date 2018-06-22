# coding:utf-8
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm

import argparse
import resnet
import ipdb
import time
import os
import cv2
import json
import numpy as np
import tensorflow as tf


def timeit(f):
    # 功能：装饰器，用来计算函数运行时间
    # 输入：要计时的函数
    # 返回：函数运行的时间
    def timed(*args, **kwargs):
        start_time = time.time()    # 开始时间
        result = f(*args, **kwargs) 
        end_time = time.time()      # 结束时间
        print("   [-] %s : %2.5f sec"%(f.__name__, end_time - start_time))
        return result
    return timed


def read_images_food11(paths, dim, x=[], y=[]):
    # 功能：读取Food-11文件夹中的图片
    # 输入：图片的路径
    # 返回：数据集x和标签y
    for root, dirs, files in os.walk(paths):
        for i in tqdm(range(len(files))):
            if files[i].endswith('jpg'): 
                img = cv2.imread(root + files[i]) # 读取当前路径下的jpg图片
                resized = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA) # 将图片resize为固定大小
                x.append(resized)   # 存储图片矩阵
                y.append([int(files[i].split('_')[0])]) # 存储标签
    return np.array(x), np.array(y)


def read_images_food172(paths, dim, nb_classes=170, nb_pics_per_class=1000):
    # 功能：读取ready_chinese_food文件夹中的图片
    # 输入：图片路径，resize后的(正方形)图片维度，食物的种类数量，每种食物的图片数量
    # 返回：数据集x和标签y
    X_train, X_valid, X_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    for i in range(1, nb_classes+1):
        for root, dirs, files in os.walk(paths + '%d/'%i):
            for j, file in enumerate(files):
                if j >= nb_pics_per_class: # 超过数量的图片，跳过
                    break
                if file.endswith('jpg'):
                    class_id, pic_ind = i, j
                    img = cv2.imread(root + files[i])
                    resized = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA) # 将图片resize为固定大小
                    X_test.append(resized)
                    y_test.append([class_id])
                elif pic_ind % 10 == 1:
                    X_valid.append(resized)
                    y_valid.append([class_id])
                else:
                    X_train.append(resized)
                    y_train.append([class_id])
    return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid), np.array(X_test), np.array(y_test)

  
def read_images_food80(paths, dim, nb_classes=80, nb_pics_per_class=1000):
    # 功能：读取Food-80文件夹中的图片
    # 输入：图片路径，resize后的(正方形)图片维度，食物的种类数量，每种食物的图片数量
    # 返回：数据集x和标签y
    X_train, X_valid, X_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    for root, dirs, files in os.walk(paths):
        for i in tqdm(range(len(files))):
            if files[i].endswith('jpg'):
                try:
                    class_id = int(files[i].split('_')[0])
                    pic_ind = int(files[i].split('_')[1].split('.')[0])
                except:
                    continue
                if pic_ind >= nb_pics_per_class or class_id >= nb_classes: # 超过种类或者数量的图片，跳过
                    continue
                img = cv2.imread(root + files[i])
                if type(img).__name__ == 'NoneType': # 图片读取错误，跳过
                    continue
                resized = clip_img(img, dim)
                # cv2.imwrite('./downloads/img_%s'%(files[i]), img)
                # cv2.imwrite('./downloads/resized_%s'%(files[i]), resized)
                if pic_ind % 10 == 0:
                    X_test.append(resized)
                    y_test.append([class_id])
                elif pic_ind % 10 == 1:
                    X_valid.append(resized)
                    y_valid.append([class_id])
                else:
                    X_train.append(resized)
                    y_train.append([class_id])
  
    return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid), np.array(X_test), np.array(y_test)


def clip_img(img, dim):
    # 切割图片，先等比例缩放，然后将长方形的图片切割为正方形的图片
    width, height, _ = img.shape
    if height <= width:
        res_shape = (int(np.ceil(1.0 * dim / height * width)), dim)
        tmp_img = cv2.resize(img, res_shape, interpolation=cv2.INTER_AREA)
        gap = int((res_shape[0] - res_shape[1]) / 2)
    else:
        res_shape = (dim, int(np.ceil(1.0 * dim / width * height)))
        tmp_img = cv2.resize(img, res_shape, interpolation=cv2.INTER_AREA)
        gap = int((res_shape[1] - res_shape[0]) / 2)
    # 将图片resize为固定大小
    resized = tmp_img[:, gap: gap+dim, :] if tmp_img.shape[0] <= tmp_img.shape[1] else tmp_img[gap:gap+dim, :, :]
    return resized


@timeit
def main(args):
    # 使用50层的resnet构建模型，并进行训练和保存模型
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('recorded_results.csv')

    gpu_fraction = args.gpu_fraction # GPU使用率
    batch_size = args.batch_size    # 批量
    nb_classes = args.nb_classes    # 食物种类数量
    nb_pics_per_class = args.nb_pics_per_class # 每种食物图片数量
    nb_epoch = args.nb_epoch        # 训练回合
    data_augmentation = args.data_augmentation # True

    # input image dimensions
    img_rows, img_cols = args.pic_dim, args.pic_dim # resize后的图片维度
    img_channels = 3 # 彩色图片，三个通道

    if nb_classes != 172:
        X_train, y_train, X_valid, y_valid, X_test, y_test = read_images_food80(
            args.data_dir, args.pic_dim, nb_classes, nb_pics_per_class)
    else:
        X_train, y_train, X_valid, y_valid, X_test, y_test = read_images_food172(
            args.data_dir, args.pic_dim, nb_classes, nb_pics_per_class)
    print('training: {}\t validation: {}\t testing: {}\n'.format(
        X_train.shape, X_valid.shape, X_test.shape))

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')

    # 获取均值图片，将所有图片减去均值，再归一化到正负一之间
    mean_image = np.mean(X_train, axis=0)
    image_dir = './models/%s_mean_image.jpg'%args.model_name
    if not os.path.exists(image_dir):
        print('\nSaving mean_image ...\n')
        cv2.imwrite(image_dir, mean_image)

    X_train -= mean_image
    X_valid -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_valid /= 128.
    X_test /= 128.

    if 0.0 < gpu_fraction < 1.0:  
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    # 使用resnet50网络来训练模型
    if args.start_epoch > 0:
        model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights('./models/%s.h5'%args.model_name)
        args.model_name = 'pd%d_bs%d_food%d_%d_%d'%(args.pic_dim, args.batch_size, 
            args.nb_classes, args.start_epoch + args.nb_epoch, args.nb_pics_per_class)
        image_dir = './models/%s_mean_image.jpg'%args.model_name
        print('\nLoaded mode: ./models/%s.h5\n'%args.model_name)
        print('New model name: %s\n'%args.model_name)
        print('Saving mean_image ...\n')
        cv2.imwrite(image_dir, mean_image)
        
    else:
        model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  validation_data=(X_valid, Y_valid),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, csv_logger])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                channel_shift_range=0, # 颜色抖动，默认值为0
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        print('\nStart training ...\n')
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_data=(X_valid, Y_valid),
                            validation_steps=X_valid.shape[0] // batch_size,
                            epochs=nb_epoch, initial_epoch=args.start_epoch,
                            verbose=2, max_q_size=100, workers=4,
                            callbacks=[lr_reducer, early_stopper, csv_logger])

    # save model
    print('\nSaving model ...\n')
    model.save_weights('./models/%s.h5'%args.model_name)

    # testing 
    y_predict = model.predict(X_test, batch_size=args.batch_size, verbose=0)
    right = 0
    for j in range(len(y_predict)):
        if np.argmax(y_predict[j]) == y_test[j][0]:
            right += 1.0
    accuracy = right / len(y_predict)
    print('\nTest accuracy: %.4f\n'%accuracy) 
    for k, v in args.__dict__.items():
        print(k, v)


def load_model(args):
    print('Loading model %s.h5 ...'%args.model_name)
    dim = args.pic_dim
    nb_classes = args.nb_classes
    img_rows, img_cols = dim, dim
    img_channels = 3
    model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('./models/%s.h5'%args.model_name)
    mean_image = cv2.imread('./models/%s_mean_image.jpg'%args.model_name)
    return model, mean_image

#@timeit
def predict(pic_dir, model, mean_image, args):
    img = cv2.imread(pic_dir)
    if type(img).__name__ == 'NoneType':
        return 'NULL', []
    #ipdb.set_trace()
    dim = args.pic_dim
    if args.test_mode == 'normal':
        if img.shape[0] > dim and img.shape[1] > dim:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        if args.is_clip_img:
            resized = clip_img(img, dim)
        else:
            resized = cv2.resize(img, (dim, dim), interpolation=interpolation)
        x = (resized.astype('float32') - mean_image) / 128. # 减去均值，然后归一化到+-1之间
        result = model.predict(np.array([x]), batch_size=1, verbose=0)
        # class_id = np.argmax(result[0])
        class_ids = np.argsort(-result[0])[:args.top_n]
        rank_probs = result[0][class_ids]
        print('{}\n{}'.format(class_ids, rank_probs))
        return [menu[k] for k in class_ids], rank_probs
    
    else:
        box_dim = (args.nb_photo_boxes + 1) * args.margin + dim # args.nb_photo_boxes + 2
        if img.shape[0] >= box_dim and img.shape[1] >= box_dim:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        box_img = cv2.resize(img, (box_dim, box_dim), interpolation=interpolation)
        mid = int((box_dim - dim) / 2) 
        nums = args.nb_photo_boxes
        # test the resized picture except the margin
        if args.test_pic_block == 'middle':
            tmp_shape = (dim + 2 * args.margin,  dim + 2 * args.margin)
            tmp_img = cv2.resize(img, tmp_shape, interpolation=interpolation)
            x = [ tmp_img[args.margin: args.margin + dim, args.margin: args.margin + dim] ] * nums
        # test most inner part of the resized picture
        elif args.test_pic_block == 'inner':
            x = [ box_img[mid: mid + dim, mid: mid + dim] ] * nums
        # test full resized picture
        else: 
            x = [cv2.resize(img, (dim, dim), interpolation=interpolation)] * nums
        for i in range(nums):
            for j in range(nums):
                left = int(i * args.margin) + args.margin
                right = left + dim
                down = int(j * args.margin) + args.margin
                up = down + dim
                x.append(box_img[left: right, down: up, :])
        
        for i, img in enumerate(x):
            # cv2.imwrite('./tests/%d_%s'%(i, pic_dir.split('/')[-1]), img)
            x[i] = normalize(x[i], mean_image)
        
        votes = {}
        result = model.predict(np.array(x), batch_size=len(x), verbose=0)
        if args.beam_size == 0:
            class_ids = np.argmax(result, axis=1)
            weights = np.max(result, axis=1)
            for i, k in enumerate(class_ids):
                if k not in votes:
                    votes[k] = [1, weights[i]/len(weights)]
                else:
                    votes[k][0] += 1
                    votes[k][1] += weights[i]/len(weights)
        else:
            class_ids = np.argsort(-result, axis=1)[:,:args.beam_size] # reverse = True
            total_weight = args.beam_size * len(x)
            for i in range(len(class_ids)):
                for j in range(len(class_ids[i])):
                    k = class_ids[i, j]
                    if k not in votes:
                        votes[k] = [1, result[i, k] / total_weight]
                    else:
                        votes[k][0] += 1
                        votes[k][1] += result[i, k] / total_weight
        rank_votes = sorted(votes.items(), key=lambda a:a[1][0], reverse=True)[:args.top_n]
        rank_weights = sorted(votes.items(), key=lambda a:a[1][1], reverse=True)[:args.top_n]
        # print('{}\n{}'.format(class_ids, rank_votes))
        return [menu[k] for k,v in rank_weights], rank_weights


def robust_test(model, mean_image, args, top_n=1):
    # 鲁棒性测试，即用真实照片，除了百度图片外的网站的实物照片来做测试，计算模型准确度
    real_photos = './tests/real_photos/' # 自己拍的真实照片
    web_photos = './tests/web_photos/' # 其他网站上的照片
    test_wrong = './tests/wrong/' # 用于存放被识别错的食物照片
    total_real = total_web = real_right = web_right = real_error = web_error = 0
    files = os.listdir(real_photos)
    # ipdb.set_trace()
    for i in tqdm(range(len(files))): # 测试真实照片
        pic_dir = files[i]
        flag = False
        try:
            results, probs = predict(real_photos + pic_dir, model, mean_image, args)
            for name in results[:top_n]:
                if name in pic_dir:
                    real_right += 1
                    flag = True
                    break
            total_real += 1
            if not flag:
                os.system('cp %s %s%s__%s'%(real_photos+pic_dir, test_wrong, results[0], pic_dir))
        except:
            real_error += 1

    files = os.listdir(web_photos) # 测试网站上的照片
    for i in tqdm(range(len(files))):
        pic_dir = files[i]
        flag = False
        try:
            results, probs = predict(web_photos + pic_dir, model, mean_image, args)
            for name in results[:top_n]:
                if name in pic_dir:
                    web_right += 1
                    flag = True
                    break
            total_web += 1
            if not flag:
                os.system('cp %s %s%s__%s'%(web_photos+pic_dir, test_wrong, results[0], pic_dir))
        except:
            web_error += 1
    print('real_photos acc: {} \t real_error: {}'.format(1.0*real_right/total_real, real_error))
    print('web_photos acc: {} \t web_error: {}'.format(1.0*web_right/total_web, web_error))
    args.printer.write(' test_mode: {}\n test_pic_block: {}\n clip_img: {}\n'.format(
                        args.test_mode, args.test_pic_block, args.is_clip_img))
    args.printer.write('real_photos acc: {} \t real_error: {}\n'.format(1.0*real_right/total_real, real_error))
    args.printer.write('web_photos acc: {} \t web_error: {}\n\n'.format(1.0*web_right/total_web, web_error))


def normalize(img, mean_image):
    # 归一化，将0~255数值转换到+-1之间
    return (img.astype('float32') - mean_image) / 128.


def softmax(x):
    # softmax概率归一化
    return np.exp(x) / sum(np.exp(x))


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-robot_shut_up',     type=bool,  default=False)
    parser.add_argument('-data_augmentation', type=bool,  default=True)
    parser.add_argument('-gpu_fraction',      type=float, default=0.9)
    parser.add_argument('-batch_size',        type=int,   default=32)
    parser.add_argument('-pic_dim',           type=int,   default=224)
    parser.add_argument('-nb_epoch',          type=int,   default=100)
    parser.add_argument('-start_epoch',       type=int,   default=50)
    parser.add_argument('-nb_classes',        type=int,   default=136)
    parser.add_argument('-nb_pics_per_class', type=int,   default=1000)
    parser.add_argument('-nb_photo_boxes',    type=int,   default=3)  # 2
    parser.add_argument('-margin',            type=int,   default=20) # margin = 10 if pic_dim < 200 else 20
    parser.add_argument('-beam_size',         type=int,   default=0)
    parser.add_argument('-top_n',             type=int,   default=3)
    parser.add_argument('-is_clip_img',       type=int,   default=1)
    parser.add_argument('-test_mode',         type=str,   default='sliding') # 'normal', 'sliding'
    parser.add_argument('-test_pic_block',    type=str,   default='full') # 'middle', 'inner', 'full' 
    parser.add_argument('-data_dir',          type=str,   default='../../data/Food-80/')
    parser.add_argument('-model_name',        type=str,   default='pd224_bs32_food136_80_1k')
    parser.add_argument('-mode',              type=str,   default='online_test')
    parser.add_argument('-pic_dir',           type=str,   default='./tests/0_200.jpg')
    parser.add_argument('-dialog_dir',        type=str,   default='./results/dialog.log')
    args = parser.parse_args()
    if args.test_mode == 'normal':
        args.top_n = 3
    if args.pic_dim == 224:
        args.model_name = 'pd224_bs32_food136_80_1k'
    elif args.pic_dim == 256:
        args.model_name = 'pd256_bs16_food136_50_500'
    if args.nb_classes == 170:
        if args.nb_pics_per_class == 1000:
            args.model_name = 'pd224_bs32_food170_170_500'
        else:
            args.model_name = 'food170_50_500'
    args.calorie_table = {}
    for line in open('./quotes/calorie_table.txt', encoding='UTF-8').readlines():
        args.calorie_table[line.split()[0]] = ' '.join(line.split()[1:3])
    return args


menu = ['豆豉蒸排骨',  '西红柿炒蛋',    '葱油鸡',      '麻婆豆腐',     '酸辣土豆丝', 
        '宫保鸡丁',    '梅菜扣肉',      '水煮肉片',    '蚝油炒生菜',   '孜然羊肉', 
        '米饭',       '鱼香肉丝',      '糖醋里脊',    '松仁玉米',     '瓦罐汤',
        '炸鸡腿',      '麻辣鱼块',     '爆炒猪肝',     '椒盐鸡块',     '辣子鸡丁',
        '上汤菠菜',    '番茄牛肉饭',   '泰式咖喱鸡饭',  '墨西哥牛肉卷', '黄金咖喱猪扒',
        '香辣猪手焖锅', '小锅牛腩米线', '红烧牛肉面',   '香煎肉汁豆腐', '铁板菠萝炒饭', 
        '鱼香茄子',    '地三鲜',       '醋溜白菜',     '炒青菜',      '炒空心菜', 
        '凉拌木耳',    '炸花生米',      '拔丝山药',     '干煸四季豆',  '炒苦瓜', 
        '虎皮青椒',    '凉拌腐竹',      '清炒西兰花',   '椒盐蘑菇',    '芹菜香干', 
        '西芹百合',    '韭菜炒鸡蛋',    '鸡蛋羹',       '水果沙拉',    '叉烧', 
        '可乐鸡翅',    '泡椒凤爪',      '口水鸡',      '黄焖鸡',      '大盘鸡', 
        '剁椒鱼头',    '三杯鸡',       '红烧肉',      '啤酒鸭',      '铁板牛肉', 
        '锅包肉',      '杭椒牛柳',     '京酱肉丝',     '青椒肉丝',    '木须肉', 
        '葱爆羊肉',     '酸菜鱼',      '羊肉串',      '红烧狮子头',   '烤鱼', 
        '松鼠鳜鱼',     '红烧带鱼',    '芝士虾球',     '皮皮虾',      '扬州炒饭', 
        '小笼包',      '蛋炒饭',      '馒头',         '包子',        '皮蛋瘦肉粥',
        '哈密瓜',      '苹果',        '香蕉',         '桃子',        '西瓜', 
        '李子',       '菠萝',         '草莓',         '芒果',       '樱桃',
        '火龙果',      '橘子',        '荔枝',         '龙眼',        '葡萄', 
        '榴莲',       '甘蔗',         '木瓜',        '柚子',        '杨桃',
        '豆浆',       '油条',         '饺子',        '面包',        '肠粉', 
        '炒河粉',      '蛋挞',       '鸡蛋灌饼',      '回锅肉',      '粉蒸肉',
        '红烧排骨',   '红烧猪蹄',     '小鸡炖蘑菇',    '炸鸡翅',      '煎蛋',
        '姜汁皮蛋',   '炸虾',        '咖喱牛肉',      '牛排',        '红烧茄子肉末',
        '土豆泥',     '炒藕片',      '腌黄瓜',        '炒荷兰豆',    '炸臭豆腐',
        '粉丝蒸扇贝',  '大闸蟹',     '辣蛤蜊',        '云吞面',      '蓝莓山药',
        '炸春卷',     '小龙虾',     '豆豉凤爪',       '荷叶糯米鸡',   '烤乳猪',
        '肉酱意面',    '肉夹馍',    '油焖大虾',       '葱油饼',       '红薯',
        '大象',       '猫',        '仓鼠',          '虾',           '鸟',
        '乌龟',       '蚯蚓',       '蜥蜴',          '青蛙',         '蜈蚣',
        '蝴蝶',       '蜻蜓',       '蝙蝠',          '老鼠',         '牛',
        '老虎',       '兔子',       '龙',            '蛇',          '马',
        '羊',         '猴',        '鸡',            '狗',           '猪',
        '披萨',       '汉堡',       '可乐',          '薯条',         '三明治']


if __name__ == '__main__':
    args = arg_init()
    if args.mode == 'train':
        main(args)
    else: # test
        global model, mean_image
        model, mean_image = load_model(args)
        if args.mode == 'online_test': # 在线测试，接收图片的路径，输出结果
            #ipdb.set_trace()
            pic_dir = input() #"input pic_dir\n")
            while pic_dir:
                # pic_dir = '../../data/Food-80/%s.jpg'%pic_dir.strip()
                res, _ = predict(pic_dir, model, mean_image, args)
                print('{"path": %s, "result": %s}'%(pic_dir, res[0]))
                pic_dir = input()
        elif args.mode == 'robust_test': # 鲁棒性测试
            with open('./results/robust_test_dim%d_class%d.txt'%(args.pic_dim, args.nb_classes), 'a') as args.printer:
                for op in ['sliding', 'normal']:
                    if op == 'normal':
                        for clip in [0, 1]:
                            args.test_mode, args.is_clip_img = op, clip
                            robust_test(model, mean_image, args, top_n=1)
                    else:
                        for nb_block in [1, 2, 3]:
                            for block in ['full', 'middle', 'inner']:
                                args.test_mode, args.test_pic_block = op, block
                                robust_test(model, mean_image, args, top_n=1)
        else: # randomly test 随机测试
            while True:
                class_id = np.random.randint(args.nb_classes)
                pic_ind = np.random.randint(20) * 10 # + np.random.randint(2)
                res, _ = predict('../../data/Food-80/%d_%d.jpg'%(class_id, pic_ind), model, mean_image, args)
                print(class_id, pic_ind, res, menu[class_id])
                confirm = input('\nContinue?(Y/N)\n').lower()
                if confirm == 'n':
                    break

