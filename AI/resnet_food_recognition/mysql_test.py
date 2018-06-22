import ipdb
import time
import random
import pymysql
import numpy as np


def timeit(f):
    # 功能：装饰器，用来计算函数运行时间
    # 输入：要计时的函数
    # 返回：函数运行的时间
    def timed(*args, **kwargs):
        start_time = time.time()    # 开始时间
        result = f(*args, **kwargs) 
        end_time = time.time()      # 结束时间
        print("   [-] {} : {} sec".format(f.__name__, end_time - start_time))
        return result
    return timed

command = """create table user(
    user_id varchar(255) not null,
    username varchar(255) not null,
    img varchar(1000) not null,
    sex int(11),
    primary key (user_id) 
    )"""

chars = ['z','y','x','w','v','u','t','s','r','q','p','o','n','m','l','k','j','i','h',
        'g','f','e','d','c','b','a','!','@','#','$','%','^','&','*']
users = []


@timeit
def test_insert(count):
    for i in range(count):
        user_id = ''.join(random.sample(chars, 32))
        username = ''.join(random.sample(chars, np.random.randint(5, 20)))
        img = ''.join(random.sample(chars, np.random.randint(30)))
        sex = np.random.randint(2)
        users.append(user_id)

        command = """insert into user 
        (user_id, username, img, sex)
        values 
        ("%s", "%s", "%s", "%d");
        """ % (user_id, username, img, sex)
        cur.execute(command)
        db.commit()

@timeit
def test_query():
    command = """select * from user where user_id = "%s"  """ % users[0]
    cur.execute(command)
    db.commit()

@timeit
def test_delete(num=0):
    if num == 0:
        command = """delete from user where user_id = "%s" """ % users[0]
    else:
        command = """delete from user"""
    cur.execute(command)
    db.commit()

@timeit
def test_update():
    command = """update user set img = "test_update" where user_id = "%s"  """ % users[0]
    cur.execute(command)
    db.commit()


if __name__ == '__main__':
    # 测试数据库的插入，查询，更新，删除的性能
    try:
        db = pymysql.connect('10.66.109.99','root','im2hungry','test')
        cur = db.cursor()
        for i in [1, 2, 4, 8, 16, 32]:     
            count = i * 1000
            print('\n number of data: %d' % count)
            test_insert(count)
            test_query()
            test_update()
            test_delete()
            test_delete(1)
    finally:
        db.close()


