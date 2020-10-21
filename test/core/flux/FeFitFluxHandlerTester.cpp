#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <fstream>
#include <iostream>

#include <mpi.h>

#include <boost/test/unit_test.hpp>

#include <xolotl/config.h>
#include <xolotl/core/flux/FeFitFluxHandler.h>
#include <xolotl/options/Options.h>

using namespace std;
using namespace xolotl::core;
using namespace flux;

using Kokkos::ScopeGuard;
BOOST_GLOBAL_FIXTURE(ScopeGuard);

/**
 * The test suite is responsible for testing the FeFitFluxHandler.
 */
BOOST_AUTO_TEST_SUITE(FeFitFluxHandlerTester_testSuite)

BOOST_AUTO_TEST_CASE(checkComputeIncidentFlux)
{
	// Create the option to create a network
	xolotl::options::Options opts;
	// Create a good parameter file
	std::ofstream paramFile("param.txt");
	paramFile << "netParam=10 0 0 10 10" << std::endl;
	paramFile.close();

	// Create a fake command line to read the options
	int argc = 2;
	char** argv = new char*[3];
	std::string appName = "fakeXolotlAppNameForTests";
	argv[0] = new char[appName.length() + 1];
	strcpy(argv[0], appName.c_str());
	std::string parameterFile = "param.txt";
	argv[1] = new char[parameterFile.length() + 1];
	strcpy(argv[1], parameterFile.c_str());
	argv[2] = 0; // null-terminate the array
	// Initialize MPI
	MPI_Init(&argc, &argv);
	opts.readParams(argc, argv);

	// Create an empty grid because we want 0D
	std::vector<double> grid;
	// Specify the surface position
	int surfacePos = 0;

	// Create the network
	using NetworkType = network::FeReactionNetwork;
	NetworkType::AmountType maxV = opts.getMaxV();
	NetworkType::AmountType maxI = opts.getMaxI();
	NetworkType::AmountType maxHe = opts.getMaxImpurity();
	NetworkType network({maxHe, maxV, maxI}, grid.size(), opts);
	network.syncClusterDataOnHost();
	network.getSubpaving().syncZones(plsm::onHost);
	// Get its size
	const int dof = network.getDOF();

	// Create the iron flux handler
	auto testFitFlux = make_shared<FeFitFluxHandler>(opts);
	// Set the flux amplitude
	testFitFlux->setFluxAmplitude(1.0);
	// Initialize the flux handler
	testFitFlux->initializeFluxHandler(network, surfacePos, grid);

	// Create a time
	double currTime = 1.0;

	// The array of concentration
	double newConcentration[dof];

	// Initialize their values
	for (int i = 0; i < dof; i++) {
		newConcentration[i] = 0.0;
	}

	// The pointer to the grid point we want
	double* updatedConc = &newConcentration[0];
	double* updatedConcOffset = updatedConc;

	// Update the concentrations
	testFitFlux->computeIncidentFlux(
		currTime, updatedConcOffset, 0, surfacePos);

	// Check the value at some grid points
	BOOST_REQUIRE_CLOSE(newConcentration[0], 1.49e-05, 0.01); // I
	BOOST_REQUIRE_CLOSE(newConcentration[1], 0.0, 0.01); // I_2
	BOOST_REQUIRE_CLOSE(newConcentration[20], 2.11e-11, 0.01); // He
	BOOST_REQUIRE_CLOSE(newConcentration[10], 9.91e-06, 0.01); // V
	BOOST_REQUIRE_CLOSE(newConcentration[11], 1.51e-06, 0.01); // V_2
	BOOST_REQUIRE_CLOSE(newConcentration[12], 2.60e-07, 0.01); // V_3
	BOOST_REQUIRE_CLOSE(newConcentration[13], 1.58e-07, 0.01); // V_4
	BOOST_REQUIRE_CLOSE(newConcentration[14], 6.29e-08, 0.01); // V_5
	BOOST_REQUIRE_CLOSE(newConcentration[18], 3.16e-08, 0.01); // V_9

	// Remove the created file
	std::string tempFile = "param.txt";
	std::remove(tempFile.c_str());

	// Finalize MPI
	MPI_Finalize();

	return;
}

BOOST_AUTO_TEST_SUITE_END()