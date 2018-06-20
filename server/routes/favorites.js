var express = require('express');
var router = express.Router();
var favorites = require('../services/favorites');

router.get('/:post_id', async function(req, res, next){
    let userId = req.session.userId;
    if(!userId){
        res.status(401);
        res.end();
        return;
    }
    let postId = req.params.post_id;
    res.writeHeader(200, {'Content-Type': 'application/json'});
    try{
        let favored = await favorites.isFavored(userId, postId);
        res.end(JSON.stringify({
            success: true,
            favored: favored
        }));
    }catch(e){
        console.error(e);
        res.end(JSON.stringify({
            success: false,
            err: e
        }));
    }
});

router.post('/:post_id',async function(req, res, next){
    let userId = req.session.userId;
    if(!userId){
        res.status(401);
        res.end();
        return;
    }
    let postId = req.params.post_id;
    let favored = req.body.favored == 'true';
    console.log("favored:", favored);
    res.writeHeader(200, {'Content-Type': 'application/json'});
    try{
        let success = favored ? await favorites.favor(userId, postId):
            await favorites.unfavor(userId, postId);
        res.end(JSON.stringify({
            success: true
        }));
    }catch(e){
        res.end(JSON.stringify({
            success: false,
            err: e
        }));
    }
});



module.exports = router;
