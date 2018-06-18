let client = require('../storage/mysql_client');

let foods = {
    async getFoodByName(foodName) {
        let result = await client.query('SELECT * FROM menu WHERE name = ?', [foodName]);
        if(result.err){
            throw err;
        }
        return result.length == 0 ? null : result[0];
    },
    async getFoodById(foodId){
        let result = await client.query('SELECT * FROM menu WHERE menu_id = ?', [foodId]);
        if(result.err){
            throw err;
        }
        return result.length == 0 ? null : result[0];
    }
}

module.exports = foods;