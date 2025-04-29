import React from 'react'
import { Outlet } from 'react-router';

const Authentication = () => {
    return (
        <>
            <div className='h-screen w-screen flex justify-end'>
                <div className="p-20 border-l border-white/20 flex flex-col w-2/5">
                    <Outlet />
                </div>

            </div>
        </>
    )
}

export default Authentication;

