import React from 'react'
import { Navigate, Outlet } from 'react-router';
import NavigationBar from '../components/NavigationBar';

const Application = () => {
    // Replace with redux implementation later 
    // Using static user data for now

    const user = "User1";
    const loading = false;

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