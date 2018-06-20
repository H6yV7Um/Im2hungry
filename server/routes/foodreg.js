var express = require('express');
var router = express.Router();
var foodreg = require('../services/foodreg');
var express = require('express')
var multer  = require('multer')
var config = require('../config');
var upload = multer({ dest: config['dir.uploads'] + '/uploads'})
var foods = require('../services/foods')


router.post('/', upload.single('img'), async function(req, res, next) {
    console.log(req.file) // form files
    if(!req.file){
        res.end(JSON.stringify({
            success: false, 
            err: {
                msg: "missing params"
            }
        }));
        return;
    }
    let img_url = req.protocol + '://' + req.get('host') +'/uploads/' + req.file.filename;
    let foodId = await foodreg.reg(req.file.path);
    let food = await foods.getFoodById(foodId);
    res.end(JSON.stringify({
        success: true,
        data: {
            img_url: img_url,
            food: food
       }
    }));
})

module.exports = router;
