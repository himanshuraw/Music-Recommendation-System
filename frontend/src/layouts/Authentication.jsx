import React from 'react'
import { Outlet } from 'react-router';

const Authentication = () => {
    return (
        <>
            <div>Authentication</div>
            <Outlet />
        </>
    )
}

export default Authentication;