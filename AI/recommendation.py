import os
import re
import ipdb
import json
import time
import jieba
import requests
import argparse
import pymysql
import numpy as np
from tqdm import tqdm
from lxml import etree
from sklearn.cluster import KMeans
from sklearn import metrics


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


class MeishijieSpider(object):
    """爬虫：从美食杰上爬取数据"""
    def __init__(self):
        self.home = args.home_url if args.start_page == 0 else '%s?page=%d'%(args.home_url, args.start_page)
        self.user_menu_dict, self.start_url, self.count_page, self.count_menu, self.count_food = self.load_data(args.dict_dir) 
        # {'user_id': [{'food_name': '',  'main_mater': [], 'aux_mater': [], 'flavor': '', 'method': ''}, ] }

    @timeit
    def load_data(self, dict_dir):
        # 加载之前保存了的文件
        if os.path.exists(dict_dir):
            print('\nLoading pre-saved data ...\n')
            return json.load(open(dict_dir, 'r'))
        else:
            return {}, '', 0, 0, 0


    def get_tree_page(self, url):
        # 输入url，返回url对应页面的tree结构内容
        page = requests.get(url).content
        tree = etree.HTML(page)
        return tree


    def get_urls(self, elements):
        # 获取一个元素的url
        return [e.get('href') for e in elements]

    @timeit
    def get_url_by_user(self):
        # ipdb.set_trace()
        current_url = self.home if not self.start_url else self.start_url

        try:
            print('Start spider ...')
            while self.count_page < args.max_page and self.count_menu < args.max_menu and current_url:
                try:
                    # 获取当前页面，一个current_url页面包括大概十个用户菜单
                    tree = self.get_tree_page(current_url)
                    next_page_url = tree.xpath('//div[@class="listtyle1_page_w"]/a[@href]')[-1].get('href')
                    self.count_page += 1
                    print('\nPage NO.%d: %s'%(self.count_page, current_url))

                    # 获取当前页所有用户菜单的url
                    users_menu = self.get_urls(tree.xpath('//div[@class="cdlist_item_style1 clearfix"]/h3/a[@href]'))
                    users_info = self.get_urls(tree.xpath('//div[@class="cdlist_item_style1 clearfix"]/div[@class="info"]/a[@href]'))
                    assert len(users_menu) == len(users_info)

                    for i, menu_url in enumerate(users_menu): 
                        user_id = users_info[i].split('=')[-1]
                        menu_id = menu_url.split('=')[-1]
                        # print('user_id: %s \t menu_id: %s'%(user_id, menu_id))
                        if user_id in self.user_menu_dict and menu_id in self.user_menu_dict[user_id]:
                            print('user_id: %s \t menu_id: %s'%(user_id, menu_id))
                            print('----- Skip duplicated menu -----')
                            continue
                        try:
                            # 进入一个用户的菜单，获取该菜单下的所有食物的信息
                            menu_tree = self.get_tree_page(menu_url)
                            foods_url = self.get_urls(menu_tree.xpath('//div[@class="info1"]/h3/a[@href]'))
                            recorede_food_urls = [menu_url + '&page=1'] # 该菜单下可能包含几页的食物，每页最多12个
                            next_food_page = menu_tree.xpath('//div[@class="listtyle1_page_w"]/a[@href]')[-1].get('href')
                            next_food_page_url = '%s%s'%(menu_url.split('?')[0], next_food_page)
                            while next_food_page: # 循环，获取该菜单所有页面的食物的url
                                if next_food_page_url in recorede_food_urls:
                                    break
                                menu_tree = self.get_tree_page(next_food_page_url)
                                recorede_food_urls.append(next_food_page_url)
                                tmp_foods_url = self.get_urls(menu_tree.xpath('//div[@class="info1"]/h3/a[@href]'))
                                if len(tmp_foods_url) > 0:
                                    foods_url.extend(tmp_foods_url)
                                    print('total foods: %d, foods of this page: %d'%(len(foods_url), len(tmp_foods_url)))
                                next_food_page = menu_tree.xpath('//div[@class="listtyle1_page_w"]/a[@href]')[-1].get('href')
                                next_food_page_url = '%s%s'%(menu_url.split('?')[0], next_food_page)
                                
                            a_menu = [] # 存储该菜单的内容，是由每个食物的字典组成的列表
                            for food_url in foods_url:
                                try:
                                    food_tree = self.get_tree_page(food_url) # 一种食物的页面
                                    food_name = food_tree.xpath('//div[@class="info1"]/h1[@class="title"]/a[@id]')[-1].text.strip()
                                    #print('food_name: %s \t url: %s'%(food_name, food_url))

                                    main = food_tree.xpath('//div[@class="yl zl clearfix"]/ul[@class="clearfix"]/li[@class]/div[@class="c"]/h4/a[@href]')
                                    auxiliary = food_tree.xpath('//div[@class="yl fuliao clearfix"]/ul[@class="clearfix"]/li[@class]/h4/a[@href]')
                                    main_materials = [str(re.sub(r'[（）()]', '', m.text.strip())) for m in main]
                                    aux_materials = [str(re.sub(r'[（）()]', '', m.text.strip())) for m in auxiliary]

                                    tmp_features = food_tree.xpath('//div[@class="info2"]/ul[@class="clearfix"]/li[@class]/a[@id]')
                                    tmp_classes = food_tree.xpath('//ul[@class="pathstlye1"]/li/a[@class="curzt"]')
                                    assert len(tmp_features) == 2
                                    features = [str(re.sub(r'[（）()]', '', f.text)) for f in tmp_features]
                                    classes = [str(re.sub(r'#', '', c.text)) for c in tmp_classes]

                                    # 保存食物名字，口味，工艺，食物页面的url，主料，辅料
                                    menu = {'food_name': str(food_name), 'flavor': features[1], 'method': features[0], 'class': classes,
                                            'food_url': food_url, 'main_mater': main_materials, 'aux_mater': aux_materials}
                                    # print(menu)
                                    a_menu.append(menu)
                                    self.count_food += 1
                                except KeyboardInterrupt:
                                    raise KeyboardInterrupt
                                except:
                                    continue
                        except KeyboardInterrupt:
                            raise KeyboardInterrupt
                        except:
                            continue
                        self.count_menu += 1
                        print('user_id: {}\t menu_id: {}\t pages: {}\t menus: {}\t foods: {}\n'.format(
                                user_id, menu_id, self.count_page, self.count_menu, self.count_food))
                        if user_id in self.user_menu_dict:
                            self.user_menu_dict[user_id][menu_id] = a_menu
                        else:
                            self.user_menu_dict[user_id] = {menu_id: a_menu}
                    current_url = next_page_url
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    continue
        except KeyboardInterrupt:
            print('\nManually KeyboardInterrupt the Program!') # 手动终止程序
        except Exception as e:
            # ipdb.set_trace()
            print('\n!Error:', e)
        finally:
            print('\nTrying to save file ... ')
            with open(args.dict_dir, 'w') as f: # 保存数据的json格式文件
                json.dump([self.user_menu_dict, current_url, self.count_page, self.count_menu, self.count_food], f, indent=4)
                
            with open(args.dict_dir+'.txt', 'w') as f: # 保存数据的txt格式文件
                for user_id in self.user_menu_dict:
                    f.write('\n\nuser_id: {}\n'.format(user_id))
                    for menu_id, menu in self.user_menu_dict[user_id].items():
                        for food in menu:
                            for k, v in food.items():
                                f.write('{}: {}\n'.format(k, v))
                            f.write('\n')
                        f.write('\n')

            print('Successfully save file %s\n'%args.dict_dir)


class DataProcessor(object):
    """docstring for DataProcessor"""
    def __init__(self):
        self.data = json.load(open(args.dict_dir))[0]
        self.food2int = {}
        self.flavor2int = {}
        self.method2int = {}
        self.class2int = {}
        self.main2int = {}
        self.aux2int = {}
        self.data2dict()
        self.food_array_dim = None
        self.db = pymysql.connect('10.66.109.99','root','im2hungry','MINI', charset='utf8')
        self.cursor = self.db.cursor()


    def data2dict(self):
        # 把食物名字，口味，工艺等文字特征全部用字典存起来，构成str->int的字典
        # 保存成类成员变量，便于之后将文字特征转换为向量表示
        keys = list(self.data.keys())
        for i in range(len(keys)):
            user_id = keys[i]
            for menu_id in self.data[user_id]:
                for food in self.data[user_id][menu_id]:
                    if len(food['food_name']) <= 3:  # 食物名称
                        if food['food_name'] not in self.food2int:
                            self.food2int[food['food_name']] = len(self.food2int)
                    else:
                        for word in jieba.cut(food['food_name']): # 超过三个汉字的，用中文分词
                            if word not in self.food2int:
                                self.food2int[word] = len(self.food2int)

                    if food['flavor'] not in self.flavor2int: # 口味
                        self.flavor2int[food['flavor']] = len(self.flavor2int)
                    if food['method'] not in self.method2int:
                        self.method2int[food['method']] = len(self.method2int)

                    for c in food['class']:  # 食物分类
                        if c not in self.class2int:
                            self.class2int[c] = len(self.class2int)

                    for main_mater in food['main_mater']: # 主料
                        if len(main_mater) <= 3:
                            if main_mater not in self.main2int:
                                self.main2int[main_mater] = len(self.main2int)
                        else:
                            for word in jieba.cut(main_mater): # 超过三个汉字的，用中文分词
                                if word not in self.main2int:
                                    self.main2int[word] = len(self.main2int)
                    for aux_mater in food['aux_mater']: # 辅料
                        if len(aux_mater) <= 3:
                            if aux_mater not in self.aux2int:
                                self.aux2int[aux_mater] = len(self.aux2int)
                        else:
                            for word in jieba.cut(aux_mater): # 超过三个汉字的，用中文分词
                                if word not in self.aux2int:
                                    self.aux2int[word] = len(self.aux2int)
        #ipdb.set_trace()
        print(' self.food2int: {}\n self.flavor2int: {}\n self.method2int: {}\n self.main2int: {}\n self.aux2int: {}\n'.format(
            len(self.food2int), len(self.flavor2int), len(self.method2int), len(self.main2int), len(self.aux2int)))


    def dict2matrix(self):
        # 将字典中的元素转换成向量表示，作为聚类模型的输入数据
        train_user_data = [] # 以用户为单位的数据，即每个用户的所有菜单下的所有食物特征统计
        train_menu_data = [] # 以菜单为单位的数据，即每个菜单下的所有食物统计
        train_food_data = [] # 以食物为单位的数据，即只保存每个食物的数据
        self.food_feature_dict = {}
        keys = list(self.data.keys())
        for i in range(len(keys)):
            user_id = keys[i]
            menus_array = [] # (nb_menu, nb_food, dim_feature)
            for menu_id in self.data[user_id]:
                menu_array = []
                for food in self.data[user_id][menu_id]:
                    flavor_array = np.zeros(len(self.flavor2int))
                    flavor_array[ self.flavor2int[ food['flavor'] ] ] = args.class_weights[0]

                    method_array = np.zeros(len(self.method2int))
                    method_array[ self.method2int[ food['method'] ] ] = args.class_weights[1]

                    class_array = np.zeros(len(self.class2int))
                    for c in food['class']:
                        class_array[ self.class2int[ c ] ] += args.class_weights[2]

                    main_array = np.zeros(len(self.main2int))
                    for main_mater in food['main_mater']:
                        if len(main_mater) <= 3:
                            main_array[ self.main2int[ main_mater ] ] += args.class_weights[3]
                        else:
                            for word in jieba.cut(main_mater):
                                main_array[ self.main2int[ word ] ] += args.class_weights[3]

                    aux_array = np.zeros(len(self.aux2int))
                    for aux_mater in food['aux_mater']:
                        if len(aux_mater) <= 3:
                            aux_array[ self.aux2int[ aux_mater ] ] += args.class_weights[4]
                        else:
                            for word in jieba.cut(aux_mater):
                                aux_array[ self.aux2int[ word ] ] += args.class_weights[4]
                    food_array = np.concatenate([flavor_array, method_array, class_array, main_array, aux_array])
                    if food['food_name'] not in self.food_feature_dict:
                        self.food_feature_dict[ food['food_name'] ] = food_array
                        if self.food_array_dim == None:
                            self.food_array_dim = len(food_array) + 1
                    menu_array.append(food_array)
                    train_food_data.append(food_array)

                avg_menu_array = np.mean( np.array(menu_array), axis=0 ) # 食物特征的平均值
                sum_menu_array = np.sum( np.array(menu_array), axis=0 ) # 食物特征的求和值，类似于词袋模型
                # menus_array.append(avg_menu_array)
                menus_array.append(sum_menu_array)
                train_menu_data.append(sum_menu_array) # 把第一维取平均，即行平均

            train_user_data.append( np.sum(np.array(menu_array), axis=0 ) )#np.mean( np.array(menus_array), axis=0 ) )
        return np.array(train_menu_data), np.array(train_user_data), np.array(train_food_data)


    def train_model(self):
        # 训练模型，实际上调用的是最简单的kmeans做一个聚类的模型
        # 分别训练以用户，菜单，食物为单位的数据，对比分析模型的优劣
        menu_data, user_data, food_data = self.dict2matrix()
        print(menu_data.shape, user_data.shape)
        print('Training model ...')
        menu_model = KMeans(n_clusters=args.nb_class, random_state=0).fit(menu_data)
        menu_labels = menu_model.labels_
        menu_centers = menu_model.cluster_centers_
        menu_score = metrics.calinski_harabaz_score(menu_data, menu_labels)

        user_model = KMeans(n_clusters=args.nb_class, random_state=0).fit(user_data)
        user_labels = user_model.labels_
        user_centers = user_model.cluster_centers_
        user_score = metrics.calinski_harabaz_score(user_data, user_labels)

        food_model = KMeans(n_clusters=args.nb_class, random_state=0).fit(food_data)
        food_labels = food_model.labels_
        food_centers = food_model.cluster_centers_
        food_score = metrics.calinski_harabaz_score(food_data, food_labels)

        print('menu_score: {} \t user_score: {} \t food_score: {}'.format(menu_score, user_score, food_score))
        print('Saving model ...')
        with open('%s_%d.json'%(args.model_dir, args.nb_class), 'w') as f:
            json.dump([food_centers.tolist(), menu_centers.tolist(), user_centers.tolist(), args.__dict__], f, indent=4)
        print('All done!\n')


    def load_model(self):
        # 加载已经训练好的模型的聚类中心的数值
        with open('%s_%d.json'%(args.model_dir, args.nb_class), 'w') as f:
            models = json.load(f)
        self.food_centers = np.array(models[0])
        self.menu_centers = np.array(models[1])
        self.user_centers = np.array(models[2])


    def recommend(self, user_id):
        # 用于线上推荐的函数，输入用户id，根据用户自己收藏菜单下的所有食物，求出其聚类中心
        # 从数据库中抽取所有post的数据，计算每个食物与当前用户的聚类中心的余弦相似度，
        # 返回以相似度从高到低排序的post id列表
        command = "select post_id from favorite where user_id = '%s' " % user_id
        self.cursor.execute(command) # 读取用户的收藏菜单数据
        result = self.cursor.fetchall()
        user_menu = []
        for row in result:
            command = "select food_name, food_cal from post where post_id = '%s' " % row[0]
            self.cursor.execute(command)
            r = self.cursor.fetchall()[0]
            user_menu.append(['%s'%r[0], float(r[1])])

        user_food_array = []  # 将用户的收藏菜单中的食物，转换为向量表示
        for row in user_menu:
            food_name, food_cal = row
            tmp_features = []
            if food_name in self.food_feature_dict:
                tmp_features = self.food_feature_dict[food_name]
            else:
                for key in self.food_feature_dict:
                    if food_name in key or key in food_name:
                        tmp_features = self.food_feature_dict[key]
            if len(tmp_features) > 0:
                tmp_features = np.concatenate( [tmp_features, np.array([float(food_cal)])], axis=0 )
                user_food_array.append(tmp_features)
        if len(user_food_array) == 0: # 如果用户菜单为空，则直接返回
            return 'NULL'

        command = "select post_id, food_name, food_cal from post"
        self.cursor.execute(command) # 抽取所有post的数据
        posts = self.cursor.fetchall()
        posts_food_array = []
        posts_ids = []
        for row in posts: # 将所有post中的食物数据转换为向量表示
            post_id, food_name, food_cal = row
            posts_ids.append(post_id)
            tmp_features = []
            if food_name in self.food_feature_dict:
                tmp_features = self.food_feature_dict[food_name]
            else:
                for key in self.food_feature_dict:
                    if food_name in key or key in food_name:
                        tmp_features = self.food_feature_dict[key]
                        break
            if len(tmp_features) > 0:   
                tmp_features = np.concatenate( [tmp_features, np.array([float(food_cal)])], axis=0 )
                posts_food_array.append(tmp_features)
        posts_food_array = np.array(posts_food_array)
        if len(posts_food_array) == 0: # 如果所有post为空，则直接返回
            return 'NULL'

        user_food_center = np.mean(user_food_array, axis=0) # 计算用户菜单的聚类中心
        similarity = np.sqrt( np.sum( np.square(posts_food_array - user_food_center), axis=1 ) )
        rank = np.argsort(-similarity) # 根据相似度排序
        rank_post_ids = [posts_ids[i] for i in rank]
        return '{"result": %s}' % '{}'.format(rank_post_ids)

        

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-max_page',       type=int,   default=52)
    parser.add_argument('-max_menu',       type=int,   default=10000)
    parser.add_argument('-start_page',     type=int,   default=0)
    parser.add_argument('-nb_class',       type=int,   default=3)
    parser.add_argument('-home_url',       type=str,   default='https://i.meishi.cc/recipe_list/') 
    parser.add_argument('-dict_dir',       type=str,   default='./web_data/user_menu_dict0_2.json')
    parser.add_argument('-class_weights',  type=list,  default=[1, 1, 1, 1, 1])
    parser.add_argument('-model_dir',      type=str,   default='./web_data/kmeans_weights')
    args = parser.parse_args()

    # spider = MeishijieSpider()
    # spider.get_url_by_user()

    data_processor = DataProcessor()
    data_processor.dict2matrix()
    while True:
        user_id = input()
        if not user_id:
            break
        print(data_processor.recommend(user_id.strip()))
    # for i in range(2, 11):
    #     args.nb_class = i
    #     print('\nnb_class: %d\n'%(i))
    #     data_processor.train_model()

