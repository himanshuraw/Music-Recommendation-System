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
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        console.log(decoded)
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