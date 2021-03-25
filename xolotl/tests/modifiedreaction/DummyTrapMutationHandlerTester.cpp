#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <boost/test/unit_test.hpp>
#include <DummyTrapMutationHandler.h>
#include <HDF5NetworkLoader.h>
#include <XolotlConfig.h>
#include <Options.h>
#include <DummyHandlerRegistry.h>
#include <DummyAdvectionHandler.h>
#include <mpi.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace xolotlCore;

/**
 * This suite is responsible for testing the DummyTrapMutationHandler.
 */
BOOST_AUTO_TEST_SUITE(DummyTrapMutationHandler_testSuite)

/**
 * Method checking the initialization and the compute modified trap-mutation methods.
 */
BOOST_AUTO_TEST_CASE(checkModifiedTrapMutation) {
	// Create the option to create a network
	xolotlCore::Options opts;
	// Create a good parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=8 0 0 10 6" << std::endl;
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
	opts.readParams(argc, argv);

	// Create the network loader
	HDF5NetworkLoader loader = HDF5NetworkLoader(
			make_shared<xolotlPerf::DummyHandlerRegistry>());
	// Create the network
	auto network = loader.generate(opts);
	// Get its size
	const int dof = network->getDOF();
	// Initialize the rates
	network->addGridPoints(1);
	network->setTemperature(1000.0, 0);

	// Suppose we have a grid with 13 grip points and distance of
	// 0.1 nm between grid points
	std::vector<double> grid;
	for (int l = 0; l < 13; l++) {
		grid.push_back((double) l * 0.1);
	}
	// Set the surface position
	int surfacePos = 0;

	// Create the modified trap-mutation handler
	DummyTrapMutationHandler trapMutationHandler;

	// Create the advection handlers needed to initialize the trap mutation handler
	std::vector<xolotlCore::IAdvectionHandler*> advectionHandlers;
	advectionHandlers.push_back(new DummyAdvectionHandler());

	// Initialize it
	trapMutationHandler.initialize(*network, 11);
	trapMutationHandler.initializeIndex1D(surfacePos, *network,
			advectionHandlers, grid, 11, 0);

	// The arrays of concentration
	double concentration[13 * dof];
	double newConcentration[13 * dof];

	// Initialize their values
	for (int i = 0; i < 13 * dof; i++) {
		concentration[i] = (double) i * i;
		newConcentration[i] = 0.0;
	}

	// Get pointers
	double *conc = &concentration[0];
	double *updatedConc = &newConcentration[0];

	// Get the offset for the second grid point
	double *concOffset = conc + dof;
	double *updatedConcOffset = updatedConc + dof;

	// Compute the modified trap mutation at the second grid point
	trapMutationHandler.computeTrapMutation(*network, concOffset,
			updatedConcOffset, 1);

	// Check the new values of updatedConcOffset
	BOOST_REQUIRE_CLOSE(updatedConcOffset[0], 0.0, 0.01); // Create I
	BOOST_REQUIRE_CLOSE(updatedConcOffset[8], 0.0, 0.01); // He3
	BOOST_REQUIRE_CLOSE(updatedConcOffset[17], 0.0, 0.01); // Create He3V
	BOOST_REQUIRE_CLOSE(updatedConcOffset[10], 0.0, 0.01); // He5
	BOOST_REQUIRE_CLOSE(updatedConcOffset[19], 0.0, 0.01); // Create He5V

	// Initialize the indices and values to set in the Jacobian
	int nHelium = network->getAll(ReactantType::He).size();
	xolotl::IdType indices[3 * nHelium];
	double val[3 * nHelium];
	// Get the pointer on them for the compute modified trap-mutation method
	xolotl::IdType *indicesPointer = &indices[0];
	double *valPointer = &val[0];

	// Compute the partial derivatives for the modified trap-mutation at the grid point 1
	int nMutating = trapMutationHandler.computePartialsForTrapMutation(*network,
			valPointer, indicesPointer, 1);

	// Verify that no cluster is undergoing modified trap-mutation
	BOOST_REQUIRE_EQUAL(nMutating, 0);

	// Remove the created file
	std::string tempFile = "param.txt";
	std::remove(tempFile.c_str());

	// Finalize MPI
	MPI_Finalize();

	return;
}

BOOST_AUTO_TEST_SUITE_END()
