var express = require('express');
var router = express.Router();
var users = require('../services/users');

router.get('/me', function(req, res, next){
  var userId = req.session.userId;
  if(userId == null || userId == undefined){
    res.writeHeader(401);
    res.end();
  }else{
    getUser(userId, res);
  }
});

router.get('/:user_id', function(req, res, next){
  getUser(req.params.user_id, res);
});

async function getUser(userId, res) {
  let user = await users.getUser(userId);        
  res.writeHeader(user ? 200 : 404, {'Content-Type': 'application/json'});
  res.end(user ? JSON.stringify(user) : undefined);
};



module.exports = router;
