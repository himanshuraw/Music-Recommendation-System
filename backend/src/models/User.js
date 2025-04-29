const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

const userSchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, 'Name is required'],
        trim: true,
        minlength: [5, 'Name must be 5 characters long'],
        maxlength: [20, 'Name must be less than 20 characters long'],
        match: [/^[a-zA-Z]+$/, 'Name must only contain alphabets']

    },
    username: {
        type: String,
        required: [true, 'Username is required'],
        unique: true,
        minlength: [5, 'Username must be 5 characters long'],
        maxlength: [20, 'Username must be less than 20 characters long'],
        match: [/^[a-zA-Z0-9]+$/, 'Username must only contain alphanumerics']

    },
    password: {
        type: String,
        required: true,
    }
})

userSchema.pre('save', async function (next) {
    if (!this.isModified('password')) return next();
    this.password = await bcrypt.hash(this.password, process.env.SALT_ROUNDS || 10);
    next();
})

module.exports = mongoose.model('User', userSchema);