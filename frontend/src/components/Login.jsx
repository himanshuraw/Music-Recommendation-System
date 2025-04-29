import React from 'react'
import useCredentials from '../hooks/useCredentials';
import LabeledInput from './LabeledInput';
import { useNavigate } from 'react-router';

const Login = () => {
    const { username, setUsername, password, setPassword, error, handleSubmission } = useCredentials('login');
    const navigate = useNavigate();
    return (
        <div className='w-full p-16'>
            <div className='text-6xl mb-5'>Login</div>
            <div className=' mb-8'>Log-in to continue your journey with endless songs..</div>
            <form onSubmit={handleSubmission} className='flex flex-col gap-6'>
                <LabeledInput
                    label='Username'
                    name='username'
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                />
                <LabeledInput
                    label='Password'
                    name='password'
                    type='password'
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
                {error && <p className='text-red-500'>{error}</p>}
                <button
                    type='submit'
                    className='bg-blue-500 text-white rounded-full my-4 p-2 hover:bg-blue-600 transition duration-200'>
                    Log in
                </button>
                <div className=''>Don't have an account? <span className='text-blue-500 hover:text-blue-600' onClick={(e) => navigate('/register')}>Sign up</span> now</div>
            </form>
        </div >
    )
}

export default Login;