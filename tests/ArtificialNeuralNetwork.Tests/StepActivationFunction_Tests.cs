using ArtificialNeuralNetwork.Abstractions;
using ArtificialNeuralNetwork.Library.Functions;
using FluentAssertions;

namespace ArtificialNeuralNetwork.Tests;

public class SigmoidActivationFunction_Tests
{
    [Theory]
    [InlineData(1, 1, 0.73105)]
    [InlineData(1, 2, 0.88079)]
    [InlineData(-1, -3, 0.9525)]
    [InlineData(10000, 1, 1)]
    [InlineData(-10000, 1, 0)]
    [Trait("Category", "Unit")]
    public void CalculateOutput_Should_Return_ExpectedValues(double input, double coefficient, double expected)
    {
        IActivationFunction stepActivationFunction = new SigmoidActivationFunction(coefficient);

        var output = stepActivationFunction.CalculateOutput(input);

        output.Should().BeInRange(expected - 0.01, expected + 0.01);
    }
}