
var express = require('express');
var request = require('request');
var querystring = require("querystring");
var config = require("../config");
var users = require('../services/users');
var router = express.Router();
var appId = config["auth.qq.app_id"];

const debug = false;
const DEBUG_USER = {
  figureurl_2: "http://www.baidu.com",
  gender: "男",
  nickname: "Test"
}

router.post('/qq', async function(req, res, next){
  res.writeHead(200, 'Content-Type', 'application/json');
  if(!checksParmas(req)){
    return;
  }
  let data = {
    user_id: req.body.user_id,
    access_token: req.body.access_token,
    app_id: appId,
    expires: req.body.expires
  };
  auth(data, req, res);
});

async function auth(data, req, res){
  try{
      let userInfo = await getUserInfo(data.user_id, data.access_token, appId);
      if(userInfo.ret < 0){
          res.end(JSON.stringify({
            success: false,
            err: {
              code: userInfo.ret,
              msg: userInfo.msg
            }
          }));
          return;
      }
      saveSession(data, req, res);
      userInfo.userId = req.body.user_id;
      await users.addQQUserIfNeeded(userInfo);
      console.log('auth for %o success: ', req.body, userInfo);
      res.end(JSON.stringify({
         success: true
      }));
  }catch(err){
    console.log('auth for %o error: ', data, err);
    res.end(JSON.stringify({
      success: false,
      err: err
   }));
  }
}

function checksParmas(req){
  if(!req.body){
    req.end(JSON.stringify({
      success: false,
      err: {
        msg: '缺少参数'
      }
    }));
    return false;
  }
  return true;
}

function saveSession(data, req, res){
    let expires = data.expires == undefined? 1000 * 3600: data.expires;
    req.session.cookie.expires = expires;
    req.session.userId = data.user_id;
    req.session.accessToken = data.access_token;
    req.session.save()
    console.log("save sessions: expires = %d, session = ", expires, req.session);
}



function getUserInfo(userId, accessToken, appId){
  return new Promise(function(resolve, reject){
    if(debug){
      resolve(DEBUG_USER);
      return;
    }
    let url = 'https://graph.qq.com/user/get_user_info?' + querystring.stringify({
      access_token: accessToken,
      oauth_consumer_key: appId,
      openid: userId
    });
    console.log(url);
    request.get(url, function(err, res, body){
      if(err){
        reject(err);
      }else{
        resolve(JSON.parse(body));
      }
    });
  });
  
}


module.exports = router;
