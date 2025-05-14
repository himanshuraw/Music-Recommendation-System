import React, { useEffect, useState } from 'react';
import { IoMdHeart, IoMdHeartEmpty } from 'react-icons/io';
import { privateAPI } from '../services/api';

const Card = ({ item }) => {
    const [liked, setLiked] = useState(item.liked);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        setLiked(item.liked);
    }, [item.liked]);

    const handleLike = async () => {
        if (isLoading) return;

        try {
            setIsLoading(true);
            const newLikedState = !liked;
            setLiked(newLikedState);

            const response = await privateAPI.post('/like', {
                track_id: item.track_id,
                artist_id: item.artist_id
            });

            console.log(response)

            if ((response.data.message === 'Like removed')) {
                setLiked(false);
            } else if (response.data.message === "Like added") {
                setLiked(true);
            }
        } catch (error) {
            console.error('Like action failed:', error);
            setLiked(!liked);
        } finally {
            setIsLoading(false);
        }
    }
    return (
        <div className="rounded-2xl overflow-hidden p-6 shadow-lg 
        transition-all duration-300 bg-[radial-gradient(ellipse_at_top_left,_#1f1f1f,_#1f2937)]
         text-[#D1D5DC] border border-white/10 relative group">
            <div className="pt-16 md:pt-20 lg:pt-24 flex justify-between items-center">
                <button
                    onClick={handleLike}
                    disabled={isLoading}
                    className={`transition-transform ${isLoading ? 'cursor-wait' : 'hover:scale-110'}`}
                    aria-label={liked ? "Unlike track" : "Like track"}
                >
                    {liked ? (
                        <IoMdHeart className='text-3xl text-blue-300' />
                    ) : (
                        <IoMdHeartEmpty className='text-3xl hover:text-blue-200' />
                    )}
                </button>

                <div>
                    <h2 className="text-2xl font-bold mb-2 text-right group-hover:text-blue-400">
                        {item.track}
                    </h2>
                    <p className="text-right text-gray-400 group-hover:text-blue-300">
                        by {item.artist}
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Card;
