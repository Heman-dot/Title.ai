import React, { useState } from 'react';
import axios from 'axios';
import styled, { keyframes } from 'styled-components';

// Styled Components for styling the form and elements
const Container = styled.div`
  padding: 20px;
  max-width: 600px;
  margin: 0 auto;
  margin-top: 10%;
  background-color: #f7f7f7;
  border-radius: 10px;
  box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h2`
  text-align: center;
  color: #333;
  font-family: 'Arial', sans-serif;
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
`;

const Label = styled.label`
  font-size: 16px;
  color: #555;
  margin-bottom: 5px;
`;

const Input = styled.input`
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 16px;
  margin-bottom: 15px;
  transition: border-color 0.3s;

  &:focus {
    border-color: #ef233c;
    outline: none;
  }
`;

const Button = styled.button`
  padding: 12px;
  background-color: ${props => (props.loading ? '#ccc' : '#ef233c')};
  color: white;
  border: none;
  border-radius: 5px;
  font-size: 16px;
  cursor: ${props => (props.loading ? 'not-allowed' : 'pointer')};
  transition: background-color 0.3s;

  &:hover {
    background-color: ${props => (props.loading ? '#ccc' : '#d90429')};
  }
`;

const LoadingSpinner = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const Spinner = styled.div`
  width: 24px;
  height: 24px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #ef233c;
  border-radius: 50%;
  animation: ${LoadingSpinner} 1s linear infinite;
  margin: 10px auto;
`;

const ErrorMessage = styled.p`
  color: red;
  text-align: center;
  font-size: 14px;
`;

const ResponseBox = styled.div`
  margin-top: 20px;
  background-color: ${props => (props.isValid ? '#c7f3c4' : '#f3c4c4')};
  padding: 15px;
  border-radius: 10px;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
`;

const Score = styled.p`
  font-size: 16px;
  text-align: center;
`;

const PredictionText = styled.h3`
  text-align: center;
  color: ${props => (props.isValid ? 'green' : 'red')};
`;

// Main Form Component
const TitleForm = () => {
  const [title, setTitle] = useState('');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Function to handle form submission
  const handleSubmit = async event => {
    event.preventDefault(); // Prevent page reload on form submission
    setLoading(true); // Show loading state

    try {
      // Send title to the backend using Axios
      const res = await axios.post('http://localhost:8080/api/check-title', { title });
      setResponse(res.data); // Save the response from the backend
      setError(null); // Clear any errors
    } catch (err) {
      setError('Error in checking title. Please try again.'); // Show error message
      setResponse(null); // Clear previous response
    } finally {
      setLoading(false); // Stop the loading state
    }
  };

  return (
    <Container>
      <Title>Check Title Similarity</Title>
      <Form onSubmit={handleSubmit}>
        <Label htmlFor="title">Enter Title:</Label>
        <Input
          type="text"
          id="title"
          name="title"
          value={title}
          onChange={e => setTitle(e.target.value)} // Update state when input changes
          required
          placeholder="Enter the title here..."
        />
        <Button type="submit" loading={loading}>
          {loading ? <Spinner /> : 'Check Title'}
        </Button>
      </Form>

      {error && <ErrorMessage>{error}</ErrorMessage>}
      {response && (
        <ResponseBox isValid={response.is_valid}>
          <PredictionText isValid={response.is_valid}>
            {response.is_valid ? 'Title is Valid' : 'Title is Invalid'}
          </PredictionText>
          <Score>Prediction Score: {parseFloat((response.prediction_score * 100).toFixed(2))}%</Score>
        </ResponseBox>
      )}
    </Container>
  );
};

export default TitleForm;
