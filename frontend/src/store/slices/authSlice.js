import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { publicAPI } from "../../services/api";
import { isTokenExpired } from "../../utils/authUtils";

const loadInitialState = () => {
    const token = localStorage.getItem('token');
    const user = localStorage.getItem('user');

    return {
        token: !!token,
        user: user ? JSON.parse(user) : null,
        loading: false,
        error: null
    }
}

const initialState = loadInitialState();

export const login = createAsyncThunk(
    'auth/login',
    async (credentials, { rejectWithValue }) => {
        try {
            const response = await publicAPI.post(`/login`, credentials);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
)

export const register = createAsyncThunk(
    'auth/register',
    async (userData, { rejectWithValue }) => {
        try {
            const response = await publicAPI.post('/register', userData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
)

const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        logout: (state) => {
            state.token = null;
            state.user = null;

            localStorage.removeItem('token');
            localStorage.removeItem('user');
            state.loading = false;
            state.error = null;
        },
        initializeAuth: (state) => {
            const token = localStorage.getItem('token');
            const user = localStorage.getItem('user');

            if (token && user && !isTokenExpired(token)) {
                state.token = token;
                state.user = JSON.parse(user);
            } else {
                state.token = null;
                state.user = null;
                localStorage.removeItem('token');
                localStorage.removeItem('user');
            }
        }
    },
    extraReducers: (builder) => {
        builder
            .addCase(login.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(login.fulfilled, (state, action) => {
                state.loading = false;
                state.token = action.payload.token;
                state.user = action.payload.user;

                localStorage.setItem('token', action.payload.token);
                localStorage.setItem('user', JSON.stringify(action.payload.user));
            })
            .addCase(login.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })

            .addCase(register.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(register.fulfilled, (state, action) => {
                state.loading = false;
                state.token = action.payload.token;
                state.user = action.payload.user;

                localStorage.setItem('token', action.payload.token);
                localStorage.setItem('user', JSON.stringify(action.payload.user));
            })
            .addCase(register.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
    }
})

export const { logout, initializeAuth } = authSlice.actions;
export default authSlice.reducer;