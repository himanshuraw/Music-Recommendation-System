const jwt = require('jsonwebtoken')

const auth = async (request, response, next) => {
    const token = request.header('Authorization').replace('Bearer ', '').trim();
    if (!token) {
        return response.status(401).json({
            success: false,
            message: 'No token provided'
        })
    }
    try {
        if (!process.env.JWT_SECRET) {
            console.error('❌  Missing JWT_SECRET environment variable');
            process.exit(1);
        }
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        request.user = decoded;
        next();

    } catch (error) {
        return response.status(401).json({
            success: false,
            message: 'Invalid token',
        })
    }
}

module.exports = { auth };