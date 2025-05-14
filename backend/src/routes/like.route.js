const router = require('express').Router();
const { request } = require('express');
const Like = require('../models/Like');
const { auth } = require('../middleware/auth');

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