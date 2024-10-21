// src/pages/Home.js
import React from 'react';
import styled from 'styled-components';

const Home = () => {
  return (
    <HomePage>
      {/* Hero Section */}
      <HeroSection>
        <HeroText>
          <h1>Welcome to AI Title Validator</h1>
          <p>
            Ensure your titles are valid, impactful, and ready for success with our AI-driven title verification system.
            Whether you're naming a product, service, or project, we help you verify and refine your title to meet
            industry standards and best practices.
          </p>
          <ActionButton >Start Verifying Now</ActionButton>
        </HeroText>
        <HeroImage>
          <img src="https://media.licdn.com/dms/image/C5612AQESQvdOzZQK_w/article-cover_image-shrink_720_1280/0/1633939642359?e=2147483647&v=beta&t=nO-auZXvskNCSn4oheCbLsdRi1tUKq7rOI_BbXf5stA" alt="AI Title Validator" />
        </HeroImage>
      </HeroSection>

      {/* Why Use AI Title Validator Section */}
      <WhyUseSection>
        <SectionTitle>Why Use AI Title Validator?</SectionTitle>
        <SectionDescription>
          Our AI-powered tool analyzes titles across multiple dimensions to ensure they’re not just catchy, but also compliant and aligned with your needs.
        </SectionDescription>

        <FeatureCards>
          <FeatureCard>
            <CardTitle>Phonetics Check</CardTitle>
            <CardText>
              Ensure your title sounds right! Our AI evaluates phonetic patterns to detect how well your title rolls off
              the tongue, making it easy to remember.
            </CardText>
          </FeatureCard>

          <FeatureCard>
            <CardTitle>Prefix & Suffix Analysis</CardTitle>
            <CardText>
              Identify if your title has the right structure by examining commonly accepted prefixes and suffixes. We
              check if your title fits linguistic norms while staying original.
            </CardText>
          </FeatureCard>

          <FeatureCard>
            <CardTitle>Disallowed Words</CardTitle>
            <CardText>
              Our system flags inappropriate or restricted terms, ensuring your title adheres to the necessary guidelines and regulations.
            </CardText>
          </FeatureCard>

          <FeatureCard>
            <CardTitle>AI-Powered Predictions</CardTitle>
            <CardText>
              Receive a prediction score for your title, giving you insights into its suitability. Whether it’s a creative
              name or a technical one, get AI-based feedback on the success potential of your title.
            </CardText>
          </FeatureCard>
        </FeatureCards>
      </WhyUseSection>

      {/* How It Works Section */}
      <HowItWorksSection>
        <SectionTitle>How It Works</SectionTitle>
        <Steps>
          <Step>
            <StepNumber>1</StepNumber>
            <StepText>Enter Your Title - Type or paste the title you want to verify in our simple, user-friendly form.</StepText>
          </Step>
          <Step>
            <StepNumber>2</StepNumber>
            <StepText>Get Real-Time Results - Instantly receive a prediction score and detailed feedback on your title’s validity.</StepText>
          </Step>
          <Step>
            <StepNumber>3</StepNumber>
            <StepText>Refine and Improve - Based on our AI’s suggestions, you can make edits and improvements to craft the perfect title.</StepText>
          </Step>
        </Steps>
      </HowItWorksSection>
    </HomePage>
  );
};

export default Home;

// Styled Components
const HomePage = styled.div`
  font-family: 'Arial', sans-serif;
`;

const HeroSection = styled.section`
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #f7f9fc;
  padding: 3rem 1.5rem;

  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

const HeroText = styled.div`
  max-width: 50%;
  color: #2b2d42;

  h1 {
    font-size: 3rem;
    color: #ef233c;
    margin-bottom: 1rem;
  }

  p {
    font-size: 1.25rem;
    margin-bottom: 2rem;
    line-height: 1.6;
  }

  @media (max-width: 768px) {
    max-width: 100%;
    text-align: center;

    h1 {
      font-size: 2rem;
    }
  }
`;

const HeroImage = styled.div`
  img {
    max-width: 60%;
    border-radius: 10px;
    margin-left:30%;
  }

  @media (max-width: 768px) {
    margin-top: 2rem;
  }
`;

const ActionButton = styled.button`
  padding: 0.75rem 1.5rem;
  background-color: #ef233c;
  color: #fff;
  border: none;
  border-radius: 5px;
  font-size: 1.2rem;
  cursor: pointer;
  transition: background-color 0.3s ease;

  &:hover {
    background-color: #d90429;
  }
`;

const WhyUseSection = styled.section`
  padding: 4rem 1.5rem;
  background-color: #fff;
  text-align: center;
`;

const SectionTitle = styled.h2`
  font-size: 2.5rem;
  color: #2b2d42;
  margin-bottom: 1rem;
`;

const SectionDescription = styled.p`
  font-size: 1.2rem;
  margin-bottom: 2rem;
  color: #8d99ae;
`;

const FeatureCards = styled.div`
  display: flex;
  justify-content: center;
  gap: 2rem;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 1.5rem;
  }
`;

const FeatureCard = styled.div`
  background-color: #edf2f4;
  padding: 1.5rem;
  border-radius: 10px;
  max-width: 300px;
  text-align: left;

  @media (max-width: 768px) {
    max-width: 100%;
  }
`;

const CardTitle = styled.h3`
  font-size: 1.5rem;
  color: #ef233c;
  margin-bottom: 1rem;
`;

const CardText = styled.p`
  font-size: 1rem;
  color: #2b2d42;
`;

const HowItWorksSection = styled.section`
  padding: 4rem 1.5rem;
  background-color: #f7f9fc;
  text-align: center;
`;

const Steps = styled.div`
  display: flex;
  justify-content: center;
  gap: 2rem;

  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

const Step = styled.div`
  background-color: #fff;
  padding: 1.5rem;
  border-radius: 10px;
  text-align: left;
  max-width: 300px;

  @media (max-width: 768px) {
    max-width: 100%;
  }
`;

const StepNumber = styled.div`
  font-size: 2.5rem;
  color: #ef233c;
  margin-bottom: 1rem;
`;

const StepText = styled.p`
  font-size: 1rem;
  color: #2b2d42;
`;
