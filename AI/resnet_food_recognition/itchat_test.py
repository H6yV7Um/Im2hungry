# coding: utf-8
import ipdb
import itchat
import requests
import numpy as np
#import tensorflow as tf

from itchat.content import *
from food_recognition import *
#from keras.backend.tensorflow_backend import set_session


@itchat.msg_register(PICTURE, isFriendChat=True, isGroupChat=True)
def get_wechat_pic(msg):
    # 群聊或者私聊中，好友发了图片，将会触发该函数，用于食物识别
    group_name = ''
    try:
        group_name = itchat.search_chatrooms(userName=msg['FromUserName'])['NickName']
        if group_name not in args.shut_up_status:
            args.shut_up_status[group_name] = True
            memberdict = {m['UserName']: m for m in itchat.update_chatroom(msg['FromUserName'])['MemberList']}
            args.memberlist[group_name] = memberdict
        find_name = args.memberlist[group_name][msg['ActualUserName']]
    except:
        find_name = itchat.search_friends(userName=msg['FromUserName'])
        if not find_name:
            find_name = args.memberlist[group_name][msg['ActualUserName']]
    
    if not args.robot_shut_up:
        if group_name and args.shut_up_status[group_name]:
            return
        if msg['FileName'].endswith('gif'):
            if not args.shut_up_status[group_name]:
                ind = np.random.rand()
                if 0.0 < ind < 0.2:
                    num = np.random.randint(len(args.quotes[0]))
                    response = args.quotes[0][num]
                elif 0.2 <= ind <= 0.7:
                    num = np.random.randint(len(args.quotes[1]))
                    response = args.quotes[1][num]
                else:
                    response = '%s童鞋, 乱发表情是不对滴！我都看不懂。 '%find_name['NickName']
                itchat.send_msg(response, toUserName=msg['FromUserName'])
        else:
            pic_dir = './downloads/%s'%msg['FileName']
            msg['Text'](pic_dir)
            results, weights = predict(pic_dir, model, mean_image, args)
            if args.test_mode == 'normal':
                probs = weights
            else:
                # sum_w = sum( [ w[1][1] for w in weights ] )
                probs = [ w[1][1]  for w in weights ] # [ w[1][1] / sum_w for w in weights ] 
            res = ''
            for i, food_name in enumerate(results):
                res += '%.6f的概率为 %s [%s]\n'%(probs[i], food_name, args.calorie_table[food_name])
            print(res, find_name['NickName'])
            itchat.send_msg('@%s: %s'%(find_name['NickName'], res), toUserName=msg['FromUserName'])


@itchat.msg_register(TEXT, isGroupChat=True)
def auto_response(msg):
    # 群聊中发了文字，触发该函数，采用图灵机器人自动回复
    group_name = itchat.search_chatrooms(userName=msg['FromUserName'])['NickName']
    find_name = itchat.search_friends(userName=msg['FromUserName'])
    # ipdb.set_trace()
    if group_name not in args.shut_up_status:
        args.shut_up_status[group_name] = True
        memberdict = {m['UserName']: m for m in itchat.update_chatroom(msg['FromUserName'])['MemberList']}
        args.memberlist[group_name] = memberdict
    if not find_name:
        find_name = args.memberlist[group_name][msg['ActualUserName']]
    input_text = msg['Text']                                 # 从好友发过来的消息
    name = '主人' if find_name['NickName'] in args.root_names else find_name['NickName']
    if ('isAt' in msg and msg['isAt']) or ('IsAt' in msg and msg['IsAt']):
        if len(input_text.split()) == 1:
            if name == '主人':
                threshold = 10
                nick = ''
            else:
                if np.random.randint(10) < 5:
                    nick = '大佬'
                else:
                    nick = '小哥哥' if find_name['Sex'] == 1 else '小姐姐'
                threshold = 4
            ind = 0 if np.random.randint(10) < threshold else 1
            num = np.random.randint(len(args.quick_res[ind]))
            itchat.send_msg('%s%s，%s'%(name, nick, args.quick_res[ind][num]), toUserName=msg['FromUserName'])
            args.shut_up_status[group_name] = False
            return

        if msg['Text'].split()[-1] in ['闭嘴', '别吵', '安静']:
            args.shut_up_status[group_name] = True
            itchat.send_msg('好的，%s'%name, toUserName=msg['FromUserName'])
            return

        if '出来' in msg['Text']:#.split()[-1] in ['出来', '说话']:
            args.shut_up_status[group_name] = False
            itchat.send_msg('好的，%s'%name, toUserName=msg['FromUserName'])
            return

    if not args.robot_shut_up and not args.shut_up_status[group_name]:
        if '6' == msg['Text'] or (msg['Text'].isdigit() and '6' in msg['Text']):
            itchat.send_msg('6'*np.random.randint(99), toUserName=msg['FromUserName'])
        else:
            get_response(msg, name)
    else:
        args.logger.write('%s: %s\n'%(name, msg['Text']))


@itchat.msg_register(TEXT, isFriendChat=True)
def auto_response(msg):
    # 私聊，发送文字，触发该函数，调用图灵机器人回答
    name = itchat.search_friends(userName=msg['FromUserName'])['NickName']
    if name in args.root_names:
        if not args.robot_shut_up:
            if msg['Text'].split()[-1] in ['闭嘴', '别吵', '安静']:
                args.robot_shut_up = True
                itchat.send_msg('好的，%s'%name, toUserName=msg['FromUserName'])
                return
        else:
            if msg['Text'].split()[-1] in ['出来', '说话']:
                args.robot_shut_up = False
                itchat.send_msg('好的，%s'%name, toUserName=msg['FromUserName'])
                return
    if not args.robot_shut_up:
        get_response(msg, name)
    else:
        args.logger.write('%s: %s\n'%(name, msg['Text']))


def get_response(msg, name):
    # 调用图灵机器人回答
    api_url = 'http://openapi.tuling123.com/openapi/api'   # 图灵机器人网址
    proxy = {'https': 'https://web-proxy.oa.com:8080'}
    data = {"key": "3aff49dbc51b4f4c9e57c64ca8ab0658", "info": msg['Text'], "userid": 12345}

    #ipdb.set_trace()
    response=requests.post(api_url, data).content
    r = json.loads(response, encoding='utf-8')
    response = r['text']                                   # 机器人回复给好友的消息
    itchat.send_msg(response, toUserName=msg['FromUserName'])

    args.logger.write('%s: %s\n'%(name, msg['Text']))
    args.logger.write('%s: %s\n'%('robot', response))       
    print('%s: %s'%(name, msg['Text']))
    print('%s: %s'%('robot', response))


if __name__ == '__main__':
    global args, model, mean_image
    args = arg_init()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    # set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    model, mean_image = load_model(args)
    with open(args.dialog_dir, 'a') as args.logger:
        args.quotes = [open('./quotes/quotes0.txt', encoding='UTF-8').readlines()] # 保存的是名言
        args.quotes.append(open('./quotes/quotes1.txt', encoding='UTF-8').readlines()) # 同上
        args.quick_res = [open('./quotes/quick_response0.txt', encoding='UTF-8').readlines()] # 保存了常用的快速回复
        args.quick_res.append(open('./quotes/quick_response1.txt', encoding='UTF-8').readlines()) # 同上
        args.root_names = ['Fence']
        args.shut_up_status = {}
        args.memberlist = {}
        print('Logging in ...')
        itchat.auto_login(True)  
        itchat.run()