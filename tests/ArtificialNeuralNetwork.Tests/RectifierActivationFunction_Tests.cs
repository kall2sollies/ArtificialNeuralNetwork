using ArtificialNeuralNetwork.Abstractions;
using ArtificialNeuralNetwork.Library.Functions;
using FluentAssertions;

namespace ArtificialNeuralNetwork.Tests
{
    public class RectifierActivationFunction_Tests
    {
        [Theory]
        [InlineData(0, 0)]
        [InlineData(1, 1)]
        [InlineData(2.456, 2.456)]
        [InlineData(-1, 0)]
        [InlineData(-12.085, 0)]
        [Trait("Category", "Unit")]
        public void CalculateOutput_Should_Return_ExpectedValues(double input, double expected)
        {
            IActivationFunction stepActivationFunction = new RectifierActivationFunction();

            var output = stepActivationFunction.CalculateOutput(input);

            output.Should().BeApproximately(expected, 0.01d);
        }
    }
}