import { jwtDecode } from "jwt-decode"

export const isTokenExpired = (token) => {
    try {
        const decodedToken = jwtDecode(token);
        const curTime = Date.now() / 1000;
        return decodedToken.exp && decodedToken.exp < curTime;
    } catch (error) {
        console.error(`Error decoding token ${token}: `, error)
        return true;
    }
}

export const setAutoLogout = (token, dispatch) => {
    try {
        const decoded = jwtDecode(token);
        const expiryTime = decoded.exp * 1000 - Date.now();

        if (expiryTime > 0) {
            setTimeout(() => {
                dispatch({ type: 'auth/logout' });
                console.log("Auto-logout triggered (token expired)");
            }, expiryTime);
        }
    } catch {
        dispatch(logout());
    }
}