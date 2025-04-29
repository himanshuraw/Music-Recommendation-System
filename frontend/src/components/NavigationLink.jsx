import React from 'react'
import { Link, useLocation } from 'react-router'

const NavigationLink = ({ to, text }) => {
    const location = useLocation();
    const isActive = location.pathname === to;
    return (
        <Link to={to}>
            <div className={`${isActive ? "bg-blue-500 hover:bg-blue-600" : "border border-transparent hover:border-blue-600"} px-4 py-2 rounded-full`}>{text}</div>
        </Link>
    )
}

export default NavigationLink