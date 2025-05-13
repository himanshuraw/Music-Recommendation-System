import { useEffect, useState } from "react"
import { privateAPI, publicAPI } from "../services/api";

const useFetchData = (url, isPrivate, params = {}) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);

                const response = isPrivate
                    ? await privateAPI.get(url, { params })
                    : await publicAPI.get(url, { params });

                setData(response.data);
                console.log(response.data);
            } catch (err) {
                setError('Failed to fetch data');
                console.error(err);
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, [url, JSON.stringify(params)]);
    return { data, loading, error };
}

export default useFetchData;