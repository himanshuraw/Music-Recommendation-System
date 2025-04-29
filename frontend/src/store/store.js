import { configureStore } from "@reduxjs/toolkit";
import authReducer from "./slices/authSlice"
import { setupPrivateAPIInterceptor } from "../services/api";


export const store = configureStore({
    reducer: {
        auth: authReducer,
    }
})

setupPrivateAPIInterceptor(store);