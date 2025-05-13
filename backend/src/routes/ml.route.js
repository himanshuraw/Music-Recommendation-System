const router = require('express').Router();
const axios = require('axios');
const User = require('../models/User');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';


router.get('/recommend/:username', async (request, response) => {
    const username = request.params.username;
    const n = parseInt(request.query.n, 10) || 10;
    try {
        const user = await User.findOne({ username });

        const userid = user._id;

        const mlResponse = await axios.get(`${ML_SERVICE_URL}/recommend/${userid}`, {
            params: { n }
        });

        return response
            .status(200)
            .json({
                success: true,
                data: mlResponse.data,
                source: 'ml service'
            });

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