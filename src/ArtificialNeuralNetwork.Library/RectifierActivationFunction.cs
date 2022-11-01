﻿using ArtificialNeuralNetwork.Abstractions;

namespace ArtificialNeuralNetwork.Library
{
    public class RectifierActivationFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return input > 0 ? input : 0;
        }
    }
}