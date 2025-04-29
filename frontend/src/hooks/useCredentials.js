import React, { useState } from 'react'
import { useDispatch } from 'react-redux';
import { useNavigate } from 'react-router';
import { login, register } from '../store/slices/authSlice';


const useCredentials = (type) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [error, setError] = useState(null);

    const dispatch = useDispatch();
    const navigate = useNavigate();

    const handleSubmission = async (e) => {
        e.preventDefault();

        if (!username) {
            setError("Username is required");
            return;
        }

        if (!password) {
            setError("Password is required");
            return;
        }

        if (type == 'register' && !name) {
            setError('Name is required');
            return;
        }

        setError(null);

        try {
            switch (type) {
                case 'login':
                    await dispatch(login({ username, password })).unwrap();
                    break;
                case 'register':
                    console.log('Registering user:', { name, username, password });
                    await dispatch(register({ name, username, password })).unwrap();
                    break;
                default:
                    throw new Error("Invalid type");
            }
            navigate('/');
        } catch (error) {
            console.error(`${type.charAt(0).toUpperCase() + type.slice(1)} failed`, error);
            setError(`${type.charAt(0).toUpperCase() + type.slice(1)} failed. Please check your credentials: ${error.message}`);
        }
    }

    return {
        username,
        setUsername,
        password,
        setPassword,
        name,
        setName,
        error,
        handleSubmission
    }
}

export default useCredentials