const router = require('express').Router();
const axios = require('axios');
const Like = require('../models/Like');
const { auth } = require('../middleware/auth');
const RetrainCounter = require('../models/RetrainCounter');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

router.post('/', auth, async (request, response) => {
    const { track_id, artist_id } = request.body;

    const userId = request.user.id;

    try {
        const like = await Like.findOne({
            userId,
            artist_id,
            track_id
        })

        if (like) {
            await like.deleteOne();
            return response
                .status(200)
                .json({ message: "Like removed" })
        }

        const newLike = await Like.create({
            userId,
            track_id,
            artist_id
        })

        const counter = await RetrainCounter.findOneAndUpdate(
            { _id: 'retrain' },
            { $inc: { count: 1 } },
            { new: true, upsert: true }
        )

        if (counter.count >= 5) {
            try {
                await axios.post(`${ML_SERVICE_URL}/retrain`);
                await RetrainCounter.updateOne(
                    { _id: 'retrain' },
                    { $set: { count: 0 } }
                );
            } catch (err) {
                console.error('Retrain trigger failed:', err.message);
            }
        }

        return response
            .status(201)
            .json({ message: "Like added" });
    } catch (error) {
        return response
            .status(500)
            .json({ message: error.message })
    }
})

module.exports = router;