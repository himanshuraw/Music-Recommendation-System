const router = require('express').Router();
const User = require('../models/User');

router.get('/', async (request, response) => {
    return response.json({ message: "User router is working" })
})

router.get('/:username', async (request, response) => {
    // Get user by his username
    const username = request.params.username;

    try {
        const user = await User.findOne({ username: username })
            .select('-password');
        if (!user) {
            return response.status(404).json({ message: `${username} do not exist` });
        }

        return response.status(200).json(user);
    } catch (error) {
        return response.status(500).json({ message: error.message });
    }
});

router.post('/register', async (request, response) => {
    // Register a new user
    const { name, username, password } = request.body;
    if (!name || !username || !password) {
        return response.status(400).json({ error: 'All fields are required' });
    }
    try {
        const newUser = await User.create({
            name,
            username,
            password
        });

        const userResponse = newUser.toObject();
        delete userResponse.password;

        return response.status(201).json(newUser);
    } catch (error) {
        if (error.code === 11000) {
            return response.status(409).json({ error: 'Username already exists' });
        }
        if (error.name === 'ValidationError') {
            const messages = Object.values(error.errors).map(val => val.message);
            return response.status(400).json({ error: messages });
        }
        return response.status(500).json({ error: error.message });
    }
})

router.post('/login', async (request, response) => {
    try {
        const { username, password } = request.body;

        if (!username || !password) {
            return response.status(400).json({ error: 'Username and password required' });
        }

        const user = await User.findOne({ username });

        if (!user || !(await bcrypt.compare(password, user.password))) {
            return response.status(401).json({ error: 'Invalid credentials' });
        }

        const userResponse = user.toObject();
        delete userResponse.password;

        return response.status(200).json(userResponse);

    } catch (error) {
        return response.status(500).json({ error: error.message });
    }
});

module.exports = router;