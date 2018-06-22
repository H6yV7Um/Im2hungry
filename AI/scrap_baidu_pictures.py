# -*- coding:utf8 -*-
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import requests
import urllib
import ipdb
import time
import sys
import os

class FoodsImSpider(object):
    """从百度图片中根据食物名字，搜索和保存图片"""
    def __init__(self):
        self.count_pictures = 0
        self.recorded_pictures = []
        self.home_dir = '../data/Food-80/'
        self.home_url = 'http://image.baidu.com'
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'} 
        # 打开浏览器，最大化浏览器，进入百度图片主页
        self.driver = webdriver.Chrome('./chromedriver')
        self.driver.maximize_window()
        self.driver.get(self.home_url)
        time.sleep(3)

    def search(self, search_keys):
        # 找到搜索框，并输入搜索的菜肴名称
        inputElement = self.driver.find_element_by_name('word')
        inputElement.send_keys(search_keys)
        inputElement.submit()
        time.sleep(3)

    def download(self, search_keys, class_id, start_num=0):
        self.search(search_keys)
        class_name = '%s%d'%(self.home_dir, class_id)
        # ipdb.set_trace()
        self.count_pictures = 0
        self.repeated_pictures = 0
        self.recorded_pictures = []
        while self.count_pictures < 1000:
            print('食物名称：%s \t 类别：%d \t 图片数量：%d'%(search_keys, class_id, self.count_pictures))
            # 找到当前一页的全部图片元素，放在elements里面
            elements = self.driver.find_elements_by_xpath('//ul/li/div/a/img')
            self.save_picture(elements, class_name, start_num)
            # 移动到本页最后一张图
            action = ActionChains(self.driver).move_to_element(elements[-1])
            action.perform()
        print('因为重复而略过的图片数量：%d\n'%self.repeated_pictures)


    def save_picture(self, elements, class_name, start_num):
        # 保存图片
        for element in elements:
            self.count_pictures += 1
            if self.count_pictures < start_num:
                continue
            img_url = element.get_attribute('data-imgurl')
            if img_url not in self.recorded_pictures: # 只保存不重复的图片
                urllib.request.urlretrieve(img_url, class_name + '_%d.jpg'%self.count_pictures)
                time.sleep(1)
            else:
                self.repeated_pictures += 1
            if self.count_pictures >= 1000: # 每种菜肴只保存1000张图片
                return


if __name__ == "__main__":
    # spider = BaiduSpider("saber")
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
    print('\nNumber of food: %d'%len(menu)) # 输出总菜肴数量
    if len(sys.argv) >= 3: # 从命令行输入三个参数，最后一个参数可省，默认为0
        start_class, end_class = int(sys.argv[1]), int(sys.argv[2]) # 开始的菜肴类别，结束的菜肴类别
        start_num = 0 if len(sys.argv) == 3 else int(sys.argv[3]) # 菜肴start从start_num张图片开始保存
        class_id = start_class
        while class_id < end_class:
            try:
                spider = FoodsImSpider()
                if class_id != start_class: # 只有第一个菜肴（即上次中断的菜肴才需要start_num > 0）
                    start_num = 0 # 否则从0开始保存
                spider.download(menu[class_id], class_id, start_num)
                class_id += 1
                time.sleep(2)
            except Exception as e:
                print('Error:', e)
                start_num = spider.count_pictures
                start_class = class_id
            finally:
                spider.driver.quit()
    else:
        print('\nWrong input arguments!\n')

