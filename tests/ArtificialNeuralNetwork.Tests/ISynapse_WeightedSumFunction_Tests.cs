using ArtificialNeuralNetwork.Abstractions;
using ArtificialNeuralNetwork.Library.Functions;
using FluentAssertions;
using Moq;

namespace ArtificialNeuralNetwork.Tests
{
    public class ISynapse_WeightedSumFunction_Tests
    {
        [Fact]
        [Trait("Category", "Unit")]
        public void CalculateInput_Should_Return_Nothing_With_NoInput()
        {
            var synapses = new List<ISynapse>();

            IInputFunction inputFunction = new WeightedSumFunction();

            var input = inputFunction.CalculateInput(synapses);

            input.Should().Be(0);
        }

        [Theory]
        [InlineData(12, 42, 12 * 42)]
        [Trait("Category", "Unit")]
        public void CalculateInput_Should_Return_ExpectedValues_With_OneSynapse(double weight, double output, double expected)
        {
            Mock<ISynapse> synapse = new();
            synapse.Setup(s => s.Weight).Returns(weight);
            synapse.Setup(s => s.GetOutput()).Returns(output);
            
            var synapses = new List<ISynapse> {synapse.Object};

            IInputFunction inputFunction = new WeightedSumFunction();

            var input = inputFunction.CalculateInput(synapses);

            input.Should().Be(expected);
        }

        [Theory]
        [InlineData(12, 42, 14, 54, (12 * 42) + (14 * 54))]
        [Trait("Category", "Unit")]
        public void CalculateInput_Should_Return_ExpectedValues_With_TwoSynapses(double weight1, double output1, double weight2, double output2, double expected)
        {
            Mock<ISynapse> synapse1 = new();
            synapse1.Setup(s => s.Weight).Returns(weight1);
            synapse1.Setup(s => s.GetOutput()).Returns(output1);

            Mock<ISynapse> synapse2 = new();
            synapse2.Setup(s => s.Weight).Returns(weight2);
            synapse2.Setup(s => s.GetOutput()).Returns(output2);
            
            var synapses = new List<ISynapse> {synapse1.Object, synapse2.Object};

            IInputFunction inputFunction = new WeightedSumFunction();

            var input = inputFunction.CalculateInput(synapses);

            input.Should().Be(expected);
        }
    }
}