let config = require('../config');
let EventEmitter = require('events').EventEmitter;
const pythonFilePath = config['py.foodreg.path'];

let spawn = require("child_process").spawn;
if(config['py.foodreg.path'] != '?'){
    var pythonProcess = startFoodRegProcess(pythonFilePath);
}
let emmiter = new EventEmitter();

let foodreg = {
    async reg(path){
        pythonProcess.stdin.write(path + "\n");
        return new Promise(function(resolve, reject){
            emmiter.on(path, result => resolve(result));
        });
    }
};


function startFoodRegProcess(path){
    var pythonProcess = spawn('python', [pythonFilePath], {
        cwd: config['py.foodreg.dir']
    });
    pythonProcess.stderr.on('data', data => {
        console.error(data.toString('utf-8'));
    });
    pythonProcess.stdout.on('data', str => {
        str = str.toString('utf-8').trim();
        if(str.startsWith("b'") && str.endsWith("'")){
            str = str.substring(2, str.length - 1);
        }
	    console.log(str);
        try{
            let data = JSON.parse(str);
	    console.log("emit ", data);
            emmiter.emit(data['path'], data['result']);
        }catch(e){console.error(e);}
    });
    return pythonProcess;
}


module.exports = foodreg;
