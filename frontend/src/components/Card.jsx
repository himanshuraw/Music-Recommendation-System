import React from 'react';

const Card = ({ item }) => {
    return (
        <div className="rounded-2xl overflow-hidden p-6 shadow-lg 
        transition-all duration-300 bg-[radial-gradient(ellipse_at_top_left,_#1f1f1f,_#1f2937)]
         text-[#D1D5DC] border border-white/10 relative group">
            <div className="pt-16 md:pt-20 lg:pt-24">
                <h2 className="text-3xl font-bold mb-2 text-right group-hover:text-blue-400">
                    {item.track}
                </h2>
                <p className="text-right text-gray-400 group-hover:text-blue-300">
                    by {item.artist}
                </p>
            </div>
        </div>
    );
};

export default Card;
