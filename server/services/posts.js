
let client = require('../storage/mysql_client');
let util = require('util')
let favorites =  require('../services/favorites');
let posts = {

    /**
     * 
     * @param {*} params 查询参数
     *      * user_id 执行查询的用户的id
     *      * post_user_id 动态发布者的id
     *      * distance 动态位置和执行查询的用户的最大距离
     *      * loc_lat 执行查询的用户的经度 
     *      * loc_lng 执行查询的用户的维度
     * @param {number} max  查询的最大数量
     */
    async queryPosts(params, max){
        let args = [];
        let sql = "SELECT * FROM post WHERE ";
        //加上距离判断
        if(isNumber(params.distance) && isNumber(params.loc_lat) && 
            isNumber(params.loc_lng)){
            sql += "(POWER(loc_lat - ?, 2) + POWER(loc_lng - ?, 2) <= POWER(?, 2)) AND ";
            args = args.concat([params.loc_lat, params.loc_lng, params.distance])
        }
        //加上动态发布者的条件
        if(params.post_user_id){
            sql += "user_id = ? AND ";
            args.push(params.post_user_id);
        }
        if(params.favored_by_user_id){
            sql += "post_id in (SELECT post_id FROM favorite WHERE user_id = ?)";
            args.push(params.favored_by_user_id);
        }
        
        sql = trimSql(sql);
        //加上查询数量的限制
        sql += " LIMIT " + parseInt(max);
        console.log("query posts: sql = ", sql);
        let result = await client.query(sql, args);
        if(!result.err){
            reuslt = await sortPosts(params.loc_lat, params.loc_lng, params.user_id, result);
        }
        return result;
    },
    async addPost(post){
        if(!post){
            return {
                success: false,
                err: 'user is null'
            };
        }
        let result = await client.query('INSERT INTO post (user_id, description, food_name, food_cal, pic_url, loc_lat, ' +
                'loc_lng, loc_name, post_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [post.user_id, post.description, post.food_name, post.food_cal, post.pic_url, post.loc_lat, post.loc_lng, 
                    post.loc_name, post.post_time]);
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
    async getPost(postId){
        let result = await client.query('SELECT * FROM post WHERE post_id = ?', [postId]);
        console.log("getPost by id %d: ", postId, result);
        if(result.err){
            console.error(result.err);
            return null;
        }else{
            let post = result.length == 0 ? null : result[0];
            if(post){
                post.favoriteCount = await favorites.getFavoriteCount(post.post_id);                
            }
            return post;
        }
    }
}

function trimSql(sql){
    if(sql.endsWith("AND ")){
        sql = sql.substring(0, sql.length - 4);
    }
    if(sql.endsWith("WHERE ")){
        sql = sql.substring(0, sql.length - 6);
    }
    return sql;
}

function isNumber(num){
    return util.isNumber(num) && !isNaN(num);
}


async function sortPosts(lat, lng, userId, posts){
    if(!isNumber(lat) || !isNumber(lng)){
        return Array.prototype.sort.call(posts, (p1, p2) =>{
            console.log(Date.parse(p1.post_time));
            return Date.parse(p2.post_time) - Date.parse(p1.post_time);
        });
    }
    let time = new Date().getTime();
    for(let i = 0; i < posts.length; i++){
        let post = posts[i];
        let distance = getDistance(lat, lng, post.loc_lat, post.loc_lng); 
        let timeDiff = (time - Date.parse(post.post_time)) / 1000;
        let favoriteCount = post.favoriteCount = await favorites.getFavoriteCount(post.post_id);
        let rank = 0;
        console.log("rank for " + post.post_id + ": dis = %d, td = %d, f = %d, puid = %s, uid = %s", distance, timeDiff, favoriteCount,
            post.user_id, userId);
        if(post.user_id == userId && timeDiff < 300 * 1000){
            console.log("just now post: " + timeDiff);
            rank += 4294967296;
        }
        if(distance < 1000){

            rank += 2000 - distance;
        }else{
            rank += (2000 - distance) / 10;
        }
        console.log("after distance rank = " + rank);
        if(timeDiff < 1800){
            timeDiff += 3600 - timeDiff;
        }
        timeDiff += Math.min(20 * favoriteCount, 2000);
        post.rank = rank;
    }
    return Array.prototype.sort.call(posts, (p1, p2) =>{
        return p2.rank - p1.rank;
    });
}

function getDistance(lng1, lat1, lng2, lat2){
    let a = rad(lat1) - rad(lat2);
    let b = rad(lng1) - rad(lng2);
    let s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a / 2), 2) + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.pow(Math.sin(b / 2), 2)));
    s = s * 6378137.0;
    s = Math.round(s * 10000) / 10000;
    return s;
}

function rad(d) {
    return d * Math.PI / 180.00; 
}

module.exports = posts;