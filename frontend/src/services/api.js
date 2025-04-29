import axios from 'axios';
import { isTokenExpired } from '../utils/authUtils';

const BASE_URL = import.meta.env.VITE_BASE_URL;

export const publicAPI = axios.create({
    baseURL: BASE_URL,
})

export const privateAPI = axios.create({
    baseURL: BASE_URL,
})

export const setupPrivateAPIInterceptor = (store) => {
    privateAPI.interceptors.request.use((config) => {
        const token = localStorage.getItem('token');
        if (token) {
            if (isTokenExpired(token)) {
                store.dispatch({ type: 'auth/logout' });
                return Promise.reject(new Error('Token expired'));
            }
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    }, (error) => Promise.reject(error));
};