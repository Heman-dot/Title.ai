// src/components/Navbar.js
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import logo from '../assets/main-logo.png';
import './Navbar.css';

const Navbar = ({ user, setUser }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    setUser(null);
    navigate('/');
  };

  return (
    <Nav>
      <Logo><img className='logo' src={logo} alt="Logo" onClick={() => navigate('/')} /></Logo>
      <Menu>
        <MenuItem><StyledLink to="/">Home</StyledLink></MenuItem>
        <MenuItem><StyledLink to="/features">Features</StyledLink></MenuItem>
        <MenuItem><StyledLink to="/about">About Us</StyledLink></MenuItem>
        {user ? (
          <>
            <MenuItem>
              <ProfileIcon onClick={() => navigate('/profile')}>
                {user.name.charAt(0)}
              </ProfileIcon>
            </MenuItem>
            <MenuItem><LogoutButton onClick={handleLogout}>Logout</LogoutButton></MenuItem>
          </>
        ) : (
          <MenuItem><StyledButton onClick={() => navigate('/login')}>Login/Register</StyledButton></MenuItem>
        )}
      </Menu>
    </Nav>
  );
};

export default Navbar;

// Styled Components for Navbar
const Nav = styled.nav`
  background: #ef233c;
  padding: 0.05rem 2rem;
  height: 80px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky; 
  top: 0;           
  z-index: 1000;    
`;

const Logo = styled.div`
  color: #edf2f4;
  cursor: pointer;
`;

const Menu = styled.ul`
  display: flex;
  list-style: none;
`;

const MenuItem = styled.li`
  margin-left: 2rem;
`;

const StyledLink = styled(Link)`
  color: #edf2f4;
  text-decoration: none;
  font-size: 1.2rem;

  &:hover {
    text-decoration: underline;
  }
`;

const StyledButton = styled.button`
  background-color: #edf2f4;
  color: #ef233c;
  margin-top:-50px;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 5px;
  font-size: 1rem;
  cursor: pointer;

  &:hover {
    background-color: #d90429;
    color: #fff;
  }
`;

const LogoutButton = styled.button`
  background: #ef233c;
  color: white;
  border: none;
  cursor: pointer;
  padding: 0.5rem 1rem;
  border-radius: 5px;

  &:hover {
    background: #d90429;
  }
`;

const ProfileIcon = styled.div`
  background-color: #edf2f4;
  color: #ef233c;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  font-size: 1.5rem;
  font-weight: bold;
`;
