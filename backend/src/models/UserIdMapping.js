const mongoose = require('mongoose');

const UserIdMappingSchema = new mongoose.Schema({
    mongo_user_id: { type: mongoose.Schema.Types.ObjectId, required: true, ref: 'User' },
    numerical_user_id: { type: Number, required: true }
});

module.exports = mongoose.model('UserIdMapping', UserIdMappingSchema);
