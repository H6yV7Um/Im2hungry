var express = require('express');
var router = express.Router();
var foods = require('../services/foods');

router.get('/:food_name', async function(req, res, next){
    try{
        let food = await foods.getFoodByName(req.params.food_name);        
        res.writeHeader(food ? 200 : 404, {'Content-Type': 'application/json'});
        res.end(JSON.stringify({
            success: food != null,
            food: food
        }));
    }catch(e){
        console.error(e);
        res.end(JSON.stringify({
            success: false,
            err: e
        }));
    }
    
});


module.exports = router;
