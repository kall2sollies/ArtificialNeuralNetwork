using ArtificialNeuralNetwork.Abstractions;
using ArtificialNeuralNetwork.Library;
using FluentAssertions;
using Moq;

namespace ArtificialNeuralNetwork.Tests;

public class Synapse_Tests
{
    [Fact]
    [Trait("Category", "Unit")]
    public void Constructor_Should_CreateSynapse_WithWeightParameter()
    {
        // Arrange
        Mock<INeuron> fromNeuron = new Mock<INeuron>();
        Mock<INeuron> toNeuron = new Mock<INeuron>();
        var inputWeight = 0.42d;

        ISynapse sut = new Synapse(
            from: fromNeuron.Object,
            to: toNeuron.Object,
            weight: inputWeight);

        // Act
        var weight = sut.Weight;

        // Assert
        weight.Should().Be(inputWeight);
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void Constructor_Should_CreateSynapse_WithRandomWeight()
    {
        // Arrange
        Mock<INeuron> fromNeuron = new Mock<INeuron>();
        Mock<INeuron> toNeuron = new Mock<INeuron>();

        // Act
        ISynapse sut = new Synapse(
            from: fromNeuron.Object,
            to: toNeuron.Object);

        ISynapse sut2 = new Synapse(
            from: fromNeuron.Object,
            to: toNeuron.Object);

        // Assert
        sut.Weight.Should().BeGreaterThan(0);
        sut.Weight.Should().BeLessThan(1);
        sut2.Weight.Should().BeGreaterThan(0);
        sut2.Weight.Should().BeLessThan(1);
        sut2.Weight.Should().NotBe(sut.Weight);
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void UpdateWeight_Should_ChangeAndSetPrevious()
    {
        // Arrange
        Mock<INeuron> fromNeuron = new Mock<INeuron>();
        Mock<INeuron> toNeuron = new Mock<INeuron>();
            
        var initialWeight = 0.42d;
        var learningRate = 0.02d;
        var delta = 0.99d;

        ISynapse sut = new Synapse(
            from: fromNeuron.Object,
            to: toNeuron.Object,
            weight: initialWeight);

        // Act
        sut.UpdateWeight(learningRate, delta);

        // Assert
        sut.Weight.Should().NotBe(initialWeight);
        sut.PreviousWeight.Should().Be(initialWeight);
        sut.Weight.Should().Be(sut.PreviousWeight + learningRate * delta);
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void IsFromNeuron_Should_MatchFromNeuron()
    {
        // Arrange
        Mock<INeuron> fromNeuron = new Mock<INeuron>();
        Mock<INeuron> toNeuron = new Mock<INeuron>();
        Guid fromNeuronId = Guid.NewGuid();
        Guid toNeuronId = Guid.NewGuid();
        fromNeuron.Setup(x => x.Id).Returns(fromNeuronId);
        toNeuron.Setup(x => x.Id).Returns(toNeuronId);

        ISynapse sut = new Synapse(
            from: fromNeuron.Object,
            to: toNeuron.Object);

        // Act & Assert
        sut.IsFromNeuron(fromNeuron.Object).Should().BeTrue();
        sut.IsFromNeuron(fromNeuron.Object.Id).Should().BeTrue();
        sut.IsFromNeuron(toNeuron.Object).Should().BeFalse();
        sut.IsFromNeuron(toNeuron.Object.Id).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void IsToNeuron_Should_MatchToNeuron()
    {
        // Arrange
        Mock<INeuron> fromNeuron = new Mock<INeuron>();
        Mock<INeuron> toNeuron = new Mock<INeuron>();
        Guid fromNeuronId = Guid.NewGuid();
        Guid toNeuronId = Guid.NewGuid();
        fromNeuron.Setup(x => x.Id).Returns(fromNeuronId);
        toNeuron.Setup(x => x.Id).Returns(toNeuronId);

        ISynapse sut = new Synapse(
            from: fromNeuron.Object,
            to: toNeuron.Object);

        // Act & Assert
        sut.IsToNeuron(toNeuron.Object).Should().BeTrue();
        sut.IsToNeuron(toNeuron.Object.Id).Should().BeTrue();
        sut.IsToNeuron(fromNeuron.Object).Should().BeFalse();
        sut.IsToNeuron(fromNeuron.Object.Id).Should().BeFalse();
    }

    [Fact]
    [Trait("Category", "Unit")]
    public void GetOutput_Should_ReturnFromNeuronOutput()
    {
        // Arrange
        Mock<INeuron> fromNeuron = new Mock<INeuron>();
        Mock<INeuron> toNeuron = new Mock<INeuron>();

        var mockOutput1 = -12.42f;
        var mockOutput2 = 7.69d;

        fromNeuron.Setup(x => x.CalculateOutput()).Returns(mockOutput1);
        toNeuron.Setup(x => x.CalculateOutput()).Returns(mockOutput2);

        ISynapse sut = new Synapse(
            from: fromNeuron.Object,
            to: toNeuron.Object);

        // Act
        var output = sut.GetOutput();

        // Assert
        output.Should().Be(mockOutput1);
        output.Should().NotBe(mockOutput2);
    }
}