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
        </div>
    )
}

export default NavigationBar;