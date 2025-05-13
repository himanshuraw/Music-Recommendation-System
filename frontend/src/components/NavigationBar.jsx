import React from 'react'
import NavigationLink from './NavigationLink';
import { useSelector } from 'react-redux';

const NavigationBar = () => {
    const { user } = useSelector((state) => state.auth)
    return (
        <div className='flex gap-10 w-screen text-xl py-6 px-14 justify-end'>
            <NavigationLink
                to="/"
                text="Home"
            />
            <NavigationLink
                to="/playlist"
                text={user?.name}
            />
        </div>
    )
}

export default NavigationBar;