import React from 'react'
import useFetchData from '../hooks/useFetchData';
import Card from '../components/Card';

const Home = () => {
    const { data, loading, error } = useFetchData('/ml/recommend', true, { n: 20 });
    if (loading) {
        return <div>loading ....</div>
    }

    return (
        <div className="px-8 py-6">
            <h1 className="text-4xl font-bold mb-6">Recommendations</h1>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                {data?.recommendations.map((item) => (
                    <Card item={item} key={item.artist_id + item.track_id} />
                ))}
            </div>
        </div>
    )
}

export default Home;