import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import styled from 'styled-components';
import axios from 'axios';

const LoginRegister = () => {
  const [isLogin, setIsLogin] = useState(true); // State to toggle between Login and Register
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const navigate = useNavigate(); // Initialize navigate

  const toggleForm = () => {
    setIsLogin(!isLogin);
  };

  // Function to handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);  // Show loading state
    setError(null);    // Clear error state

    try {
      const endpoint = isLogin ? "/api/login" : "/api/register";
      const data = isLogin ? { email, password } : { username, email, password };
      
      // Send request to backend for login/register
      const res = await axios.post(`http://localhost:8080${endpoint}`, data);
      
      if (res.data.success) {
        alert(isLogin ? "Login successful!" : "Registration successful!");
        // Clear form fields
        setEmail("");
        setPassword("");
        setUsername("");

        // Navigate to home page on successful login/register
        navigate('/');
      } else {
        setError(res.data.message || "Something went wrong. Please try again.");
      }
    } catch (err) {
      setError("Error in authentication. Please try again.");
    } finally {
      setLoading(false);  // Stop loading
    }
  };

  return (
    <AuthSection>
      <FormContainer>
        <FormTitle>{isLogin ? 'Login' : 'Register'}</FormTitle>
        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <InputWrapper>
              <label htmlFor="username">Username</label>
              <Input
                type="text"
                id="username"
                placeholder="Enter username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </InputWrapper>
          )}
          <InputWrapper>
            <label htmlFor="email">Email</label>
            <Input
              type="email"
              id="email"
              placeholder="Enter email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </InputWrapper>
          <InputWrapper>
            <label htmlFor="password">Password</label>
            <Input
              type="password"
              id="password"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </InputWrapper>

          {error && <p style={{ color: 'red' }}>{error}</p>}
          <SubmitButton type="submit" disabled={loading}>
            {loading ? (isLogin ? "Logging in..." : "Registering...") : (isLogin ? 'Login' : 'Register')}
          </SubmitButton>
        </form>

        <ToggleText onClick={toggleForm}>
          {isLogin ? "Don't have an account? Register here" : "Already have an account? Login here"}
        </ToggleText>
      </FormContainer>
    </AuthSection>
  );
};

export default LoginRegister;

// Styled Components (same as before)

const AuthSection = styled.section`
  background: #edf2f4;
  padding: 4rem 1rem;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
`;

const FormContainer = styled.div`
  background: #fff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  max-width: 400px;
  width: 100%;
  text-align: center;
  margin-bottom: 10%;
`;

const FormTitle = styled.h2`
  font-size: 2rem;
  color: #ef233c;
  margin-bottom: 1.5rem;
`;

const InputWrapper = styled.div`
  margin-bottom: 1rem;
  text-align: left;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.55rem;
  margin-top: 0.5rem;
  border: 1px solid #8d99ae;
  border-radius: 4px;
  font-size: 1rem;
  color: #2b2d42;
  margin-left: -3%;

  &:focus {
    border-color: #ef233c;
    outline: none;
  }
`;

const SubmitButton = styled.button`
  width: 100%;
  padding: 0.75rem;
  background: #ef233c;
  color: white;
  font-size: 1rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;

  &:hover {
    background: #d90429;
  }
`;

const ToggleText = styled.p`
  margin-top: 1rem;
  font-size: 0.9rem;
  color: #2b2d42;
  cursor: pointer;

  &:hover {
    color: #ef233c;
  }
`;
