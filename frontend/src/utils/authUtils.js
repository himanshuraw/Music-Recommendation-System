import { jwtDecode } from "jwt-decode"
import { store } from "../store/store";
import { logout } from "../store/slices/authSlice";

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

export const setAutoLogout = (token) => {
    try {
        const decoded = jwtDecode(token);
        const expiryTime = decoded.exp * 1000 - Date.now();

        if (expiryTime > 0) {
            setTimeout(() => {
                store.dispatch(logout());
                console.log("Auto-logout triggered (token expired)");
            }, expiryTime);
        }
    } catch {
        store.dispatch(logout());
    }
}