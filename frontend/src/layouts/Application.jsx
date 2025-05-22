import React from 'react'
import { Navigate, Outlet } from 'react-router';
import NavigationBar from '../components/NavigationBar';
import { useSelector } from 'react-redux';

const Application = () => {

    const { user, loading } = useSelector((state) => state.auth)

    if (loading) {
        return <div>loading ...</div>
    }

    if (!user) {
        return <Navigate to={"/login"} />
    }
    return (
        <>
            <NavigationBar />
            <Outlet />
        </>
    )
}

export default Application;