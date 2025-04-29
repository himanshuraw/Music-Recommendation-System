import React from 'react'
import NavigationLink from './NavigationLink';

const NavigationBar = () => {
    return (
        <div className='flex gap-10 w-screen text-xl py-6 px-14 justify-end'>
            <NavigationLink
                to="/"
                text="Home"
            />
            <NavigationLink
                to="/playlist"
                text="Playlist"
            />
        </div>
    )
}

export default NavigationBar;