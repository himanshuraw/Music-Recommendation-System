import axios from 'axios';
import { isTokenExpired } from '../utils/authUtils';
import { store } from '../store/store';
import { logout } from '../store/slices/authSlice';

const BASE_URL = import.meta.env.VITE_BASE_URL;

export const publicAPI = axios.create({
    baseURL: BASE_URL,
})

export const privateAPI = axios.create({
    baseURL: BASE_URL,
})

privateAPI.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');

    if (token) {
        if (isTokenExpired(token)) {
            store.dispatch(logout());
            return Promise.reject(new Error('Token expored'));
        }
        config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
}, (error) => {
    return Promise.reject(error);
})
