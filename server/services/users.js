let client = require('../storage/mysql_client');
let users = {
    async getUser(userId){
        let result = await client.query('SELECT * FROM user WHERE user_id = ?', [userId]);
        if(result.err){
            console.error(result.err);
            return null;
        }else{
            return result.length == 0 ? null : result[0];
        }
    },
    async addUser(user){
        if(!user){
            return {
                success: false,
                err: 'user is null'
            };
        }
        let result = await client.query('INSERT INTO user VALUES (?, ?, ?, ?)',
             [user.user_id, user.username, user.img, user.sex]);
        //console.log("addUser: user=%o, result=%o", user, result);
        if(result.err){
            return {
                success: false,
                err: result.err
            };
        }else{
            return {
                success: true
            };
        }
    },
    async addQQUserIfNeeded(userInfo){
        let user = qqUserInfoToUser(userInfo);
        if(!await users.getUser(user.user_id)){
            let result = await users.addUser(user);
            //console.log("addQQUserIfNeeded: userInfo=%o", userInfo);    
            return result;        
        }else{
            return false;
        }
    },
    
};

function qqUserInfoToUser(userInfo){
    return {
        user_id: userInfo.userId,
        img: userInfo.figureurl_2,
        sex: userInfo.gender == '男' ? 0 : 1,
        username: userInfo.nickname
    }       
}


module.exports = users;