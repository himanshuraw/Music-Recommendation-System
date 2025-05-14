const router = require('express').Router();
const axios = require('axios');
const User = require('../models/User');
const { auth } = require('../middleware/auth');
const Like = require('../models/Like');
const UserIdMapping = require('../models/UserIdMapping');
const { default: mongoose } = require('mongoose');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';


router.get('/recommend', auth, async (request, response) => {
    const n = parseInt(request.query.n, 10) || 16;

    try {
        const userid = request.user.id;

        const mapping = await UserIdMapping.findOne({ mongo_user_id: new mongoose.Types.ObjectId(userid) });
        console.log(mapping)

        numericalUserId = userid;

        if (mapping) {
            numericalUserId = mapping.numerical_user_id;
        }

        const mlResponse = await axios.get(`${ML_SERVICE_URL}/recommend/${numericalUserId}`, {
            params: { n }
        });

        const recommendations = mlResponse.data.recommendations || [];

        if (!recommendations || recommendations.length === 0) {
            return response.status(200).json(mlResponse.data);
        }

        const trackArtistPairs = recommendations.map(rec => ({
            track_id: rec.track_id,
            artist_id: rec.artist_id
        }));

        const likes = await Like.find({
            userId: userid,
            $or: trackArtistPairs.map(pair => ({
                track_id: pair.track_id,
                artist_id: pair.artist_id
            }))
        });

        const likedMap = new Map();
        likes.forEach(like => {
            likedMap.set(`${like.track_id}-${like.artist_id}`, true);
        });

        const enhancedRecommendations = recommendations.map(rec => ({
            ...rec,
            liked: likedMap.has(`${rec.track_id}-${rec.artist_id}`) || false
        }));

        return response.status(200).json({
            ...mlResponse.data,
            recommendations: enhancedRecommendations
        });

    } catch (error) {
        return handleRecommendationError(error, response);
    }
});

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