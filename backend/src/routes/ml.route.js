const router = require('express').Router();
const axios = require('axios');
const User = require('../models/User');
const { auth } = require('../middleware/auth');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';


router.get('/recommend', auth, async (request, response) => {
    const n = parseInt(request.query.n, 10) || 16;

    try {
        const userid = request.user.id;

        const mlResponse = await axios.get(`${ML_SERVICE_URL}/recommend/${userid}`, {
            params: { n }
        });

        return response
            .status(200)
            .json(mlResponse.data);

    } catch (error) {
        return handleRecommendationError(error, response);
    }
})

function handleRecommendationError(error, response) {
    if (error.response) {
        return response
            .status(error.response.status)
            .json({
                success: false,
                message: error.response.data.message || error.message
            });
    }
    return response
        .status(500)
        .json({
            success: false,
            message: error.message || 'An unknown error occurred'
        });
}


module.exports = router;