import React from 'react'
import { useNavigate } from 'react-router';
import LabeledInput from './LabeledInput';
import useCredentials from '../hooks/useCredentials';

const Register = () => {
    const { username, setUsername, password, setPassword, name, setName, error, handleSubmission } = useCredentials('register');
    const navigate = useNavigate();
    return (
        <div className='w-full p-16'>
            <div className='text-6xl mb-5'>Register</div>
            <div className=' mb-8'>Register to start your quest of endless music..</div>
            <form onSubmit={handleSubmission} className='flex flex-col gap-6'>
                <LabeledInput
                    label='Name'
                    name='name'
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                />
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
                    Sign up
                </button>
                <div className=''>Already have an account? <span className='text-blue-500 hover:text-blue-600' onClick={(e) => navigate('/login')}>Log in</span> now</div>
            </form>
        </div >
    )
}

export default Register;