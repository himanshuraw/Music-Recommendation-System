const mongoose = require('mongoose');

const likeSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
    },
    track_id: {
        type: Number,
        required: true,
    },
    artist_id: {
        type: Number,
        required: true,
    }
})


module.exports = mongoose.model('Like', likeSchema);
