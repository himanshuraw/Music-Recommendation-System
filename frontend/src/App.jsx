import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router'
import './App.css'
import Login from './components/Login'
import Authentication from './layouts/Authentication'
import Application from './layouts/Application'
import Home from './pages/Home'
import Register from './components/Register'

function App() {

  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route element={<Authentication />}>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Route>
          <Route element={<Application />}>
            <Route path='/' element={<Home />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App
