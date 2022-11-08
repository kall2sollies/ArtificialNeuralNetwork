using ArtificialNeuralNetwork.Abstractions;
using ArtificialNeuralNetwork.Library.Functions;
using FluentAssertions;

namespace ArtificialNeuralNetwork.Tests.Functions;

public class StepActivationFunction_Tests
{
    [Theory]
    [InlineData(1.23, 0.98, 1)]
    [InlineData(0.23, 0.98, 0)]
    [InlineData(0.98, 0.98, 0)]
    [Trait("Category", "Unit")]
    public void CalculateOutput_Should_Return_ExpectedValues(double input, double threshold, double expected)
    {
        IActivationFunction stepActivationFunction = new StepActivationFunction(threshold);

        var output = stepActivationFunction.CalculateOutput(input);

        output.Should().Be(expected);
    }
}