let client = require('../storage/mysql_client');
let favorites = {
    async isFavored(userId, postId){
        let result = await client.query('SELECT * FROM favorite WHERE user_id = ? AND post_id = ?', [userId, postId]);
        console.log("getUser by id %d: ", userId, result);
        if(result.err){
            console.error(result.err);
            throw result.err;
        }else{
            return result.length > 0;
        }
    },
    async favor(userId, postId){
        //todo 使用事务
        if(await this.isFavored(userId, postId)){
            return {
                success: false
            };
        }
        let sql = "INSERT INTO favorite (user_id, post_id) VALUES (?, ?)";
        let result = await client.query(sql, [userId, postId, userId, postId]);
        console.log("favor user_id = %s, post_id = %d: ", userId, postId, result);
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
    async unfavor(userId, postId){
        //todo 使用事务
        if(!(await this.isFavored(userId, postId))){
            return {
                success: false
            };
        }
        let sql = "DELETE FROM favorite WHERE user_id = ? AND post_id = ?";
        let result = await client.query(sql, [userId, postId]);
        console.log("unfavor user_id = %s, post_id = %d: ", userId, postId, result);
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
    async getFavoriteCount(postId){
        let sql = "SELECT COUNT(*) FROM favorite WHERE post_id = ?";
        let result = await client.query(sql, [postId]);
        if(result.err){
            throw result.err;
        }else{
            return result[0]["COUNT(*)"];
        }
    }
};



module.exports = favorites;