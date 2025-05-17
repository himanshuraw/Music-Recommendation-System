import React from 'react'
import NavigationLink from './NavigationLink';
import { useDispatch, useSelector } from 'react-redux';
import { logout } from '../store/slices/authSlice';

const NavigationBar = () => {
    const { user } = useSelector((state) => state.auth)
    const dispatch = useDispatch();
    return (
        <div className='flex gap-10 w-screen text-xl py-6 px-14 justify-end'>
            <NavigationLink
                to="/"
                text="Home"
            />
            <div
                className="px-4 py-2 rounded-full cursor-pointer border border-transparent hover:border-blue-600"
                onClick={() => dispatch(logout())}
            >
                {user?.name}
            </div>
        </div>
    )
}

export default NavigationBar;