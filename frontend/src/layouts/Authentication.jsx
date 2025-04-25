import React from 'react'
import { Outlet } from 'react-router';
import NavigationBar from '../components/NavigationBar';

const Authentication = () => {
    return (
        <>
            {/* Remove NavigationBar later */}
            <NavigationBar />
            <div>Authentication</div>
            <Outlet />
        </>
    )
}

export default Authentication;