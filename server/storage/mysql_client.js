var mysql      = require('mysql');
var config = require('../config');

var connection = mysql.createConnection({
  host     : config['mysql.host'],
  user     : config['mysql.yser'],
  password : config['mysql.password'],
  database : config['mysql.database']
});
connection.connect();

var mysql_client = {
   async query(sql, args){
      return new Promise(function(resolve, reject){
          connection.query(sql, args, function(err, result){
              if(result == null || result == undefined){
                result = {};
              }
              result.err = err;
              resolve(result);
          });
      });
   }
};


Object.defineProperty(mysql_client, "conn", {
    value: connection,
    writable: false
});



module.exports = mysql_client;