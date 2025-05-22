const mongoose = require('mongoose');

const retrainCounterSchema = new mongoose.Schema({
    _id: {
        type: String,
        default: 'retrain'
    },
    count: {
        type: Number,
        required: true,
        default: 0
    }
});

module.exports = mongoose.model('RetrainCounter', retrainCounterSchema);