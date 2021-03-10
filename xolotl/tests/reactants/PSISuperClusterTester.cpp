#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <boost/test/unit_test.hpp>
#include <PSICluster.h>
#include <PSISuperCluster.h>
#include <HDF5NetworkLoader.h>
#include <PSIHeCluster.h>
#include <PSIMixedCluster.h>
#include <XolotlConfig.h>
#include <DummyHandlerRegistry.h>
#include <Constants.h>
#include <Options.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace xolotlCore;

/**
 * This suite is responsible for testing the PSISuperCluster.
 */
BOOST_AUTO_TEST_SUITE (PSISuperCluster_testSuite)

/**
 * This operation checks the ability of the PSISuperCluster to describe
 * its connectivity to other clusters.
 */
BOOST_AUTO_TEST_CASE(checkConnectivity) {
	// Create the parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 5 0" << std::endl << "grid=100 0.5"
			<< std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char **argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array
	// Initialize MPI for HDF5
	MPI_Init(&argc, &argv);

	// Read the options
	Options opts;
	opts.readParams(argc, argv);

	// Create the loader
	PSIClusterNetworkLoader loader = PSIClusterNetworkLoader(
			std::make_shared<xolotlPerf::DummyHandlerRegistry>());
	// Set grouping parameters
	loader.setVMin(4);
	loader.setWidth(4, 0);
	loader.setWidth(1, 3);

	// Generate the network from the options
	auto network = loader.generate(opts);

	// Recompute Ids and network size and redefine the connectivities
	network->reinitializeConnectivities();

	// Get the super cluster
	IReactant::Composition composition;
	composition[toCompIdx(Species::He)] = 18;
	composition[toCompIdx(Species::V)] = 5;
	composition[toCompIdx(Species::I)] = 4;
	auto reactant = network->get(ReactantType::PSISuper, composition);

	// Check the type name
	BOOST_REQUIRE(ReactantType::PSISuper == reactant->getType());
	auto reactionConnectivity = reactant->getConnectivity();

	// Check the connectivity for He, V, and I
	int connectivityExpected[] = { 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			1, 0, 0, 1, 1, 1, 0 };
	for (unsigned int i = 0; i < reactionConnectivity.size(); i++) {
		BOOST_REQUIRE_EQUAL(reactionConnectivity[i], connectivityExpected[i]);
	}

	return;
}

/**
 * This operation checks the ability of the PSISuperCluster to compute the total flux.
 */
BOOST_AUTO_TEST_CASE(checkTotalFlux) {
	// Create the parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 5 0" << std::endl << "grid=100 0.5"
			<< std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char **argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array

	// Read the options
	Options opts;
	opts.readParams(argc, argv);

	// Create the loader
	PSIClusterNetworkLoader loader = PSIClusterNetworkLoader(
			std::make_shared<xolotlPerf::DummyHandlerRegistry>());
	// Set grouping parameters
	loader.setVMin(4);
	loader.setWidth(4, 0);
	loader.setWidth(1, 3);

	// Generate the network from the options
	auto network = loader.generate(opts);
	// Add a grid point for the rates
	network->addGridPoints(1);

	// Set the temperature in the network
	int networkSize = network->size();
	double temperature = 1000.0;
	network->setTemperature(temperature, 0);
	// Recompute Ids and network size and redefine the connectivities
	network->reinitializeConnectivities();

	// Get the super cluster
	IReactant::Composition composition;
	composition[toCompIdx(Species::He)] = 18;
	composition[toCompIdx(Species::V)] = 5;
	composition[toCompIdx(Species::I)] = 4;
	auto cluster = network->get(ReactantType::PSISuper, composition);
	// Get one that it combines with (He)
	auto secondCluster = (PSICluster*) network->get(Species::He, 1);

	// Set the concentrations
	cluster->setConcentration(0.5);
	secondCluster->setConcentration(0.5);

	// Get and check the flux
	double flux = cluster->getTotalFlux(0);
	BOOST_REQUIRE_CLOSE(8284790005.657, flux, 0.1);

	return;
}

/**
 * This operation checks the PSISuperCluster get*PartialDerivatives methods.
 */
BOOST_AUTO_TEST_CASE(checkPartialDerivatives) {
	// Create the parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 5 0" << std::endl << "grid=100 0.5"
			<< std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char **argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array

	// Read the options
	Options opts;
	opts.readParams(argc, argv);

	// Create the loader
	PSIClusterNetworkLoader loader = PSIClusterNetworkLoader(
			std::make_shared<xolotlPerf::DummyHandlerRegistry>());
	// Set grouping parameters
	loader.setVMin(4);
	loader.setWidth(4, 0);
	loader.setWidth(1, 3);

	// Generate the network from the options
	auto network = loader.generate(opts);
	// Add a grid point for the rates
	network->addGridPoints(1);

	// Set the temperature in the network
	int networkSize = network->size();
	double temperature = 1000.0;
	network->setTemperature(temperature, 0);
	// Recompute Ids and network size and redefine the connectivities
	network->reinitializeConnectivities();

	// Get the super cluster
	IReactant::Composition composition;
	composition[toCompIdx(Species::He)] = 18;
	composition[toCompIdx(Species::V)] = 5;
	composition[toCompIdx(Species::I)] = 4;
	auto cluster = network->get(ReactantType::PSISuper, composition);

	// The vector of partial derivatives to compare with
	double knownPartials[] = { 1.65696e+10, 5.64406e+09, 1.16845e+09, 0, 0, 0,
			0, 0, 0, 7.02156e-20, 0, 0, 0, -1.04991e-10 };

	// Set the concentration
	cluster->setConcentration(0.5);

	// Set up the network to be able to compute the partial derivatives
	xolotlCore::IReactionNetwork::SparseFillMap dfill;
	network->getDiagonalFill(dfill);
	// Get the dof
	const int dof = network->getDOF();
	// Initialize the arrays for the reaction partial derivatives
	std::vector<long int> reactionSize;
	reactionSize.resize(dof);
	std::vector<size_t> reactionStartingIdx;
	reactionStartingIdx.resize(dof);
	auto nPartials = network->initPartialsSizes(reactionSize,
			reactionStartingIdx);
	std::vector<long int> reactionIndices;
	reactionIndices.resize(nPartials);
	network->initPartialsIndices(reactionSize, reactionStartingIdx,
			reactionIndices);
	std::vector<double> reactionVals;
	reactionVals.resize(nPartials);
	// Compute the partial derivatives
	network->computeAllPartials(reactionStartingIdx, reactionIndices,
			reactionVals, 0);

	// Check the size of the partials
	int id = cluster->getId() - 1;
	BOOST_REQUIRE_EQUAL(reactionSize[id], 14U);

	// Check all the values
	for (unsigned int i = 0; i < reactionSize[id]; i++) {
		BOOST_REQUIRE_CLOSE(reactionVals[reactionStartingIdx[id] + i],
				knownPartials[i], 0.1);
	}

	return;
}

/**
 * This operation checks the reaction radius for PSISuperCluster.
 */
BOOST_AUTO_TEST_CASE(checkReactionRadius) {
	// Create the parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 5 0" << std::endl << "grid=100 0.5"
			<< std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char **argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array

	// Read the options
	Options opts;
	opts.readParams(argc, argv);

	// Create the loader
	PSIClusterNetworkLoader loader = PSIClusterNetworkLoader(
			std::make_shared<xolotlPerf::DummyHandlerRegistry>());
	// Set grouping parameters
	loader.setVMin(4);
	loader.setWidth(4, 0);
	loader.setWidth(1, 3);

	// Generate the network from the options
	auto network = loader.generate(opts);

	// Get the super cluster
	IReactant::Composition composition;
	composition[toCompIdx(Species::He)] = 18;
	composition[toCompIdx(Species::V)] = 5;
	composition[toCompIdx(Species::I)] = 4;
	auto cluster = network->get(ReactantType::PSISuper, composition);

	// Check the radius
	BOOST_REQUIRE_CLOSE(0.248079, cluster->getReactionRadius(), 0.001);

	return;
}

/**
 * This operation checks the get concentration methods for PSISuperCluster.
 */
BOOST_AUTO_TEST_CASE(checkGetConcentrations) {
	// Create the parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 5 0" << std::endl << "grid=100 0.5"
			<< std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char **argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array

	// Read the options
	Options opts;
	opts.readParams(argc, argv);

	// Create the loader
	PSIClusterNetworkLoader loader = PSIClusterNetworkLoader(
			std::make_shared<xolotlPerf::DummyHandlerRegistry>());
	// Set grouping parameters
	loader.setVMin(4);
	loader.setWidth(4, 0);
	loader.setWidth(1, 3);

	// Generate the network from the options
	auto network = loader.generate(opts);
	// Add a grid point for the rates
	network->addGridPoints(1);

	// Get the super cluster
	IReactant::Composition composition;
	composition[toCompIdx(Species::He)] = 18;
	composition[toCompIdx(Species::V)] = 5;
	composition[toCompIdx(Species::I)] = 4;
	auto superCluster = (PSISuperCluster*) network->get(ReactantType::PSISuper,
			composition);

	// Set the concentration in the cluster
	superCluster->setZerothMoment(4.5);
	superCluster->setMoment(1.0, 0); // He
	superCluster->setMoment(1.0, 3); // V

	// Check the different concentrations
	BOOST_REQUIRE_CLOSE(18.0, superCluster->getTotalConcentration(), 0.001);
	BOOST_REQUIRE_CLOSE(336.33, superCluster->getTotalAtomConcentration(0),
			0.001); // He
	BOOST_REQUIRE_CLOSE(90.0, superCluster->getTotalVacancyConcentration(),
			0.001);
	BOOST_REQUIRE_CLOSE(0.0, superCluster->getIntegratedVConcentration(1),
			0.001);

	return;
}

/**
 * This operation checks the boundary methods for PSISuperCluster.
 */
BOOST_AUTO_TEST_CASE(checkBoundaries) {
	// Create the parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 5 0" << std::endl << "grid=100 0.5"
			<< std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char **argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array

	// Read the options
	Options opts;
	opts.readParams(argc, argv);

	// Create the loader
	PSIClusterNetworkLoader loader = PSIClusterNetworkLoader(
			std::make_shared<xolotlPerf::DummyHandlerRegistry>());
	// Set grouping parameters
	loader.setVMin(4);
	loader.setWidth(4, 0);
	loader.setWidth(1, 3);

	// Generate the network from the options
	auto network = loader.generate(opts);

	IReactant::Composition composition;
	composition[toCompIdx(Species::He)] = 18;
	composition[toCompIdx(Species::V)] = 5;
	composition[toCompIdx(Species::I)] = 4;
	auto superCluster = (PSISuperCluster*) network->get(ReactantType::PSISuper,
			composition);

	// Check the different numbers
	BOOST_REQUIRE_CLOSE(5.0, superCluster->getNumV(), 0.001);
	BOOST_REQUIRE_CLOSE(4.0, superCluster->getNTot(), 0.001);
	auto bounds = superCluster->getBounds(0); // He
	BOOST_REQUIRE_EQUAL(17, *(bounds.begin()));
	BOOST_REQUIRE_EQUAL(21, *(bounds.end()));
	bounds = superCluster->getBounds(1); // D
	BOOST_REQUIRE_EQUAL(0, *(bounds.begin()));
	BOOST_REQUIRE_EQUAL(1, *(bounds.end()));
	bounds = superCluster->getBounds(2); // T
	BOOST_REQUIRE_EQUAL(0, *(bounds.begin()));
	BOOST_REQUIRE_EQUAL(1, *(bounds.end()));
	bounds = superCluster->getBounds(3); // V
	BOOST_REQUIRE_EQUAL(5, *(bounds.begin()));
	BOOST_REQUIRE_EQUAL(6, *(bounds.end()));

	// Remove the created file
	std::string tempFile = "param.txt";
	std::remove(tempFile.c_str());

	// Finalize MPI
	MPI_Finalize();

	return;
}

BOOST_AUTO_TEST_SUITE_END()
