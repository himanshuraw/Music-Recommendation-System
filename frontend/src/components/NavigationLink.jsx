import React from 'react'
import { Link, useLocation } from 'react-router'

const NavigationLink = ({ to, text }) => {
    const location = useLocation();
    const isActive = location.pathname === to;
    return (
        <Link to={to}>
            {isActive
                ? <div> --{text}-- </div>
                : <div> {text} </div>
            }
        </Link>
    )
}

export default NavigationLink