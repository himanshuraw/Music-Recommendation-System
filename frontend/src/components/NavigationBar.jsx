import React from 'react'
import NavigationLink from './NavigationLink';

const NavigationBar = () => {
    return (
        <div style={{ display: 'flex', gap: '20px' }}>
            <NavigationLink
                to="/"
                text="Home"
            />
            <NavigationLink
                to="/playlist"
                text="Playlist"
            />
            <NavigationLink
                to="/login"
                text="Login"
            />
            <NavigationLink
                to="/register"
                text="Register"
            />
        </div>
    )
}

export default NavigationBar;