import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Features from './pages/Feature';
import About from './pages/About';
import LoginRegister from './pages/LoginRegister';
import Profile from './pages/Profile';

const App = () => {
  const [user, setUser] = useState(null); // Manage user state

  return (
    <Router>
      <Navbar user={user} setUser={setUser} />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/features" element={<Features />} />
        <Route path="/about" element={<About />} />
        <Route path="/login" element={<LoginRegister setUser={setUser} />} />
        <Route path="/profile" element={<Profile user={user} />} />
      </Routes>
    </Router>
  );
};

export default App;
