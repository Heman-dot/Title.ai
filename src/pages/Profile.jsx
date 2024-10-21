// src/components/Profile.js
import React from 'react';
import styled from 'styled-components';

const Profile = ({ user }) => {
  if (!user) {
    return <Message>Please log in to view your profile.</Message>;
  }

  return (
    <ProfileContainer>
      <Header>Welcome, {user.name}!</Header>
      <ProfileDetails>
        <DetailItem><strong>Name:</strong> {user.name}</DetailItem>
        <DetailItem><strong>Email:</strong> {user.email}</DetailItem>
        {/* Add more fields as needed */}
      </ProfileDetails>
    </ProfileContainer>
  );
};

export default Profile;

// Styled Components for Profile Page
const ProfileContainer = styled.div`
  padding: 2rem;
  max-width: 600px;
  margin: 0 auto;
  background-color: #edf2f4;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
`;

const Header = styled.h2`
  text-align: center;
  color: #2b2d42;
`;

const ProfileDetails = styled.div`
  margin-top: 2rem;
`;

const DetailItem = styled.p`
  font-size: 1.2rem;
  color: #2b2d42;
  margin: 0.5rem 0;
`;

const Message = styled.div`
  text-align: center;
  padding: 2rem;
  font-size: 1.2rem;
  color: #ef233c;
`;
