var express = require('express');
var router = express.Router();
var posts = require('../services/posts')
var multer  = require('multer')
var config = require('../config')
var upload = multer({ dest: config['dir.uploads'] + '/uploads'})
var users = require('../services/users')
var favorites = require('../services/favorites');

router.post('/', upload.single('img'), async function(req, res, next){
  var userId = req.session.userId;
  if(!userId){
      res.status(401);
      res.end();
      return;
  }
  if(!req.file){
    res.end(JSON.stringify({
        success: false,
        err: {
            msg: "missing `img` file params"
        }
    }));
    return;
  }
  req.body.user_id = userId;
  req.body.post_time = new Date()
  req.body.pic_url = req.protocol + '://' + req.get('host') + '/uploads/' + req.file.filename;
  let result = await posts.addPost(req.body);
  res.status(200);
  res.end(JSON.stringify(result));
});

router.get('/:post_id', function(req, res, next){
  getPost(req.params.post_id, req, res);
});

router.get('/', async function(req, res, next){
    let max = req.query.max;
    var params = {
        user_id: req.session.userId,
        distance: parseFloat(req.query.distance),
        loc_lat: parseFloat(req.query.loc_lat),
        loc_lng: parseFloat(req.query.loc_lng),
        post_user_id: req.query.post_user_id,
        favored_by_user_id: req.query.favored_by_user_id
    }
    if(req.query.favored != undefined){
        if(req.session.userId){
            params.favored_by_user_id = req.session.userId;
        }else{
            res.status(401);
            res.end();
            return;
        }
    }
    
    let result = await posts.queryPosts(params, max);
    //console.log("query posts: params = %o, result = ", params, result);
    res.status(200);
    res.end(JSON.stringify(await fillPosts(req.session.userId, result)));
});

async function getPost(postId, req, res) {
  let post = await fillPost(req.session.userId, await posts.getPost(postId));        
  res.writeHeader(post ? 200 : 404, {'Content-Type': 'application/json'});
  res.end(post ? JSON.stringify(post) : undefined);
};

async function fillPost(myUserId, post){
    if(post == null)
        return null;
    let user = await users.getUser(post.user_id);
    post.user = user;
    if(myUserId){
        post.favored = await favorites.isFavored(myUserId, post.post_id);
    }
    return post;
}

async function fillPosts(myUserId, posts){
    if(posts == null){
        return [];
    }
    for(let i = 0; i < posts.length; i++){
        posts[i] = await fillPost(myUserId, posts[i]);
    }
    return posts;
}

module.exports = router;
