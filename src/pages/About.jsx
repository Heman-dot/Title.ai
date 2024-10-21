// src/pages/About.js
import React from 'react';
import styled from 'styled-components';

const About = () => {
  return (
    <AboutPage>
      {/* Hero Section */}
      <HeroSection>
        <h1>Welcome to TitleVerifier.AI</h1>
        <p>
          Where innovation meets precision. We are a team of dedicated professionals who believe in the power of artificial
          intelligence to enhance accuracy, efficiency, and trust in digital titles across industries.
        </p>
      </HeroSection>

      {/* What We Do Section */}
      <Section>
        <SectionTitle>What We Do</SectionTitle>
        <SectionContent>
          <Feature>
            <h3>Check Phonetic Similarity</h3>
            <p>Ensure that variations in title names don't compromise accuracy.</p>
          </Feature>

          <Feature>
            <h3>Handle Prefixes and Suffixes</h3>
            <p>Automatically process common variations to avoid mismatches.</p>
          </Feature>

          <Feature>
            <h3>Disallow Unapproved Words</h3>
            <p>Maintain a predefined standard of word usage for title authenticity.</p>
          </Feature>

          <Feature>
            <h3>ML-Driven Predictions</h3>
            <p>Predict and flag potential errors with precision using machine learning.</p>
          </Feature>
        </SectionContent>
      </Section>

      {/* Our Vision Section */}
      <Section>
        <SectionTitle>Our Vision</SectionTitle>
        <SectionText>
          We believe in creating a future where technology simplifies everyday tasks and enables businesses to focus on what truly
          matters. By integrating AI into title verification, we aim to set new standards for accuracy and efficiency in the
          industry.
        </SectionText>
      </Section>

      {/* Footer */}
      <Footer>
        <SocialLinks>
          <h4>Follow Us</h4>
          <ul>
            <li><a href="#facebook">Facebook</a></li>
            <li><a href="#twitter">Twitter</a></li>
            <li><a href="#linkedin">LinkedIn</a></li>
          </ul>
        </SocialLinks>
        <FeedbackForm>
          <h4>Feedback</h4>
          <form>
            <input type="text" placeholder="Your Name" required />
            <input type="email" placeholder="Your Email" required />
            <textarea placeholder="Your Feedback" required></textarea>
            <button type="submit">Submit</button>
          </form>
        </FeedbackForm>
      </Footer>
    </AboutPage>
  );
};

export default About;

// Styled Components
const AboutPage = styled.div`
  font-family: 'Arial', sans-serif;
  padding: 2rem;
  color: #2b2d42;
`;

const HeroSection = styled.section`
  text-align: center;
  padding: 3rem;
  background-color: #f7f9fc;

  h1 {
    font-size: 2.5rem;
    color: #ef233c;
  }

  p {
    font-size: 1.2rem;
    margin-top: 1rem;
    line-height: 1.6;
  }
`;

const Section = styled.section`
  padding: 3rem 1.5rem;
  background-color: #fff;
  text-align: center;

  &:nth-child(even) {
    background-color: #f7f9fc;
  }
`;

const SectionTitle = styled.h2`
  font-size: 2rem;
  color: #2b2d42;
  margin-bottom: 2rem;
`;

const SectionContent = styled.div`
  display: flex;
  justify-content: center;
  gap: 2rem;
  flex-wrap: wrap;

  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

const Feature = styled.div`
  background-color: #edf2f4;
  padding: 1.5rem;
  border-radius: 10px;
  max-width: 300px;
  text-align: left;

  h3 {
    color: #ef233c;
    margin-bottom: 1rem;
  }

  p {
    color: #2b2d42;
  }
`;

const SectionText = styled.p`
  font-size: 1.2rem;
  color: #2b2d42;
  line-height: 1.6;
`;

const Footer = styled.footer`
  display: flex;
  justify-content: space-between;
  padding: 2rem;
  background-color: #2b2d42;
  color: #fff;

  @media (max-width: 768px) {
    flex-direction: column;
    text-align: center;
  }
`;

const SocialLinks = styled.div`
  padding-left:30%;
  h4 {
    margin-bottom: 1rem;
  }

  ul {
    list-style: none;
    padding: 0;

    li {
      margin: 0.5rem 0;

      a {
        color: #fff;
        text-decoration: none;
        transition: color 0.3s;

        &:hover {
          color: #ef233c;
        }
      }
    }
  }
`;

const FeedbackForm = styled.div`
  padding-right:30%;

  h4 {
    margin-bottom: 1rem;
  }

  form {
    display: flex;
    flex-direction: column;
    gap: 1rem;

    input, textarea {
      padding: 0.75rem;
      border-radius: 5px;
      border: none;
    }

    button {
      padding: 0.75rem;
      background-color: #ef233c;
      color: #fff;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      transition: background-color 0.3s;

      &:hover {
        background-color: #d90429;
      }
    }
  }
`;
