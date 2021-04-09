// Includes
#include "PetscSolver.h"
#include <xolotlPerf.h>
#include <VizHandlerRegistryFactory.h>
#include <PlotType.h>
#include <CvsXDataProvider.h>
#include <CvsXYDataProvider.h>
#include <LabelProvider.h>
#include <Constants.h>
#include <petscts.h>
#include <petscsys.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <PSISuperCluster.h>
#include <NESuperCluster.h>
#include <MathUtils.h>
#include <MPIUtils.h>
#include "RandomNumberGenerator.h"
#include "xolotlCore/io/XFile.h"
#include "xolotlSolver/monitor/Monitor.h"

namespace xolotlSolver {

// Declaration of the functions defined in Monitor.cpp
extern PetscErrorCode checkTimeStep(TS ts);
extern PetscErrorCode monitorTime(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void *ictx);
extern PetscErrorCode computeFluence(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void *ictx);
extern PetscErrorCode monitorPerf(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void *ictx);

// Declaration of the variables defined in Monitor.cpp
extern std::shared_ptr<xolotlViz::IPlot> perfPlot;
extern double timeStepThreshold;

//! How often HDF5 file is written
PetscReal hdf5Stride3D = 0.0;
//! Previous time for HDF5
PetscInt hdf5Previous3D = 0;
//! HDF5 output file name
std::string hdf5OutputName3D = "xolotlStop.h5";
//! The pointer to the 2D plot used in MonitorSurfaceXY3D.
std::shared_ptr<xolotlViz::IPlot> surfacePlotXY3D;
//! The pointer to the 2D plot used in MonitorSurfaceXZ3D.
std::shared_ptr<xolotlViz::IPlot> surfacePlotXZ3D;
//! The variable to store the interstitial flux at the previous time step.
std::vector<std::vector<double> > previousIFlux3D;
//! The variable to store the total number of interstitials going through the surface.
std::vector<std::vector<double> > nInterstitial3D;
//! Variables to keep track of Xe reaching the surface
double previousXeFlux3D = 0.0;
double nXeSurf3D = 0.0;
//! The variable to store the sputtering yield at the surface.
double sputteringYield3D = 0.0;
// The vector of depths at which bursting happens
std::vector<std::tuple<int, int, int> > depthPositions3D;
// Declare the vector that will store the Id of the clusters
std::vector<int> indices3D;
// Declare the vector that will store the weight of the clusters
// (their He composition)
std::vector<int> weights3D;
// Declare the vector that will store the radii of bubbles
std::vector<double> radii3D;
// The id of the largest cluster
int largestClusterId3D = -1;
// The concentration threshold for the largest cluster
double largestThreshold3D = 1.0e-12;
// Tracks the previous TS number
int previousTSNumber3D = -1;

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorLargest3D")
/**
 * This is a monitoring method that looks at the largest cluster concentration
 */
PetscErrorCode monitorLargest3D(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void*) {
	// Initial declaration
	PetscErrorCode ierr;
	double ****solutionArray, *gridPointSolution;
	PetscInt xs, xm, ys, ym, zs, zm;

	PetscFunctionBeginUser;

	// Get the MPI communicator
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	// Get the number of processes
	int worldSize;
	MPI_Comm_size(xolotlComm, &worldSize);
	// Gets the process ID (important when it is running in parallel)
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);

	// Loop on the local grid
	for (PetscInt k = zs; k < zs + zm; k++)
		for (PetscInt j = ys; j < ys + ym; j++)
			for (PetscInt i = xs; i < xs + xm; i++) {
				// Get the pointer to the beginning of the solution data for this grid point
				gridPointSolution = solutionArray[k][j][i];
				// Check the concentration
				if (gridPointSolution[largestClusterId3D]
						> largestThreshold3D) {
					ierr = TSSetConvergedReason(ts, TS_CONVERGED_USER);
					CHKERRQ(ierr);
					// Send an error
					throw std::string(
							"\nxolotlSolver::Monitor3D: The largest cluster concentration is too high!!");
				}
			}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "startStop3D")
/**
 * This is a monitoring method that will update an hdf5 file at each time step.
 */
PetscErrorCode startStop3D(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void*) {
	// Initial declarations
	PetscErrorCode ierr;
	const double ****solutionArray, *gridPointSolution;
	PetscInt xs, xm, Mx, ys, ym, My, zs, zm, Mz;

	PetscFunctionBeginUser;

	// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

	// Compute the dt
	double previousTime = solverHandler.getPreviousTime();
	double dt = time - previousTime;

	// Don't do anything if it is not on the stride
	if (((int) ((time + dt / 10.0) / hdf5Stride3D) <= hdf5Previous3D)
			&& timestep > 0)
		PetscFunctionReturn(0);

	// Update the previous time
	if ((int) ((time + dt / 10.0) / hdf5Stride3D) > hdf5Previous3D)
		hdf5Previous3D++;

	// Gets the process ID (important when it is running in parallel)
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);
	// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the network
	auto &network = solverHandler.getNetwork();

	// Network size
	const int dof = network.getDOF();

	// Create an array for the concentration
	double concArray[dof][2];

	// Get the vector of positions of the surface
	std::vector<std::vector<int> > surfaceIndices;
	for (PetscInt i = 0; i < My; i++) {
		// Create a temporary vector
		std::vector<int> temp;
		for (PetscInt j = 0; j < Mz; j++) {
			temp.push_back(solverHandler.getSurfacePosition(i, j));
		}
		// Add the temporary vector to the vector of surface indices
		surfaceIndices.push_back(temp);
	}

	// Open the existing HDF5 file.
	xolotlCore::XFile checkpointFile(hdf5OutputName3D, xolotlComm,
			xolotlCore::XFile::AccessMode::OpenReadWrite);

	// Get the current time step
	double currentTimeStep;
	ierr = TSGetTimeStep(ts, &currentTimeStep);
	CHKERRQ(ierr);

	// Add a concentration sub group
	auto concGroup = checkpointFile.getGroup<
			xolotlCore::XFile::ConcentrationGroup>();
	assert(concGroup);
	auto tsGroup = concGroup->addTimestepGroup(timestep, time, previousTime,
			currentTimeStep);

	if (solverHandler.moveSurface()) {
		// Write the surface positions in the concentration sub group
		tsGroup->writeSurface3D(surfaceIndices, nInterstitial3D,
				previousIFlux3D);
	}

	// Loop on the full grid
	for (PetscInt k = 0; k < Mz; k++) {
		for (PetscInt j = 0; j < My; j++) {
			for (PetscInt i = 0; i < Mx; i++) {
				// Wait for all the processes
				MPI_Barrier(xolotlComm);

				// Size of the concentration that will be stored
				int concSize = -1;
				// To save which proc has the information
				int concId = 0;
				// To know which process should write
				bool write = false;

				// If it is the locally owned part of the grid
				if (i >= xs && i < xs + xm && j >= ys && j < ys + ym && k >= zs
						&& k < zs + zm) {
					write = true;
					// Get the pointer to the beginning of the solution data for this grid point
					gridPointSolution = solutionArray[k][j][i];

					// Loop on the concentrations
					for (int l = 0; l < dof; l++) {
						if (std::fabs(gridPointSolution[l]) > 1.0e-16) {
							// Increase concSize
							concSize++;
							// Fill the concArray
							concArray[concSize][0] = (double) l;
							concArray[concSize][1] = gridPointSolution[l];
						}
					}

					// Increase concSize one last time
					concSize++;

					// Save the procId
					concId = procId;
				}

				// Get which processor will send the information
				int concProc = 0;
				MPI_Allreduce(&concId, &concProc, 1, MPI_INT, MPI_SUM,
						xolotlComm);

				// Broadcast the size
				MPI_Bcast(&concSize, 1, MPI_INT, concProc, xolotlComm);

				// Skip the grid point if the size is 0
				if (concSize == 0)
					continue;

				// All processes create the dataset and fill it
				tsGroup->writeConcentrationDataset(concSize, concArray, write,
						i, j, k);
			}
		}
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeHeliumRetention3D")
/**
 * This is a monitoring method that will compute the helium retention.
 */
PetscErrorCode computeHeliumRetention3D(TS ts, PetscInt, PetscReal time,
		Vec solution, void*) {
	// Initial declarations
	PetscErrorCode ierr;
	PetscInt xs, xm, ys, ym, zs, zm, Mx, My, Mz;

	PetscFunctionBeginUser;

	// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

	// Get the flux handler that will be used to compute fluxes.
	auto fluxHandler = solverHandler.getFluxHandler();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);
	// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the physical grid in the x direction
	auto grid = solverHandler.getXGrid();

	// Get the network
	auto &network = solverHandler.getNetwork();

	// Setup step size variables
	double hy = solverHandler.getStepSizeY();
	double hz = solverHandler.getStepSizeZ();

	// Get the array of concentration
	double ****solutionArray, *gridPointSolution;
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	// Store the concentration over the grid
	double heConcentration = 0.0, dConcentration = 0.0, tConcentration = 0.0;

	// Loop on the grid
	for (PetscInt zk = zs; zk < zs + zm; zk++) {
		for (PetscInt yj = ys; yj < ys + ym; yj++) {

			// Get the surface position
			int surfacePos = solverHandler.getSurfacePosition(yj, zk);

			for (PetscInt xi = xs; xi < xs + xm; xi++) {

				// Boundary conditions
				if (xi < surfacePos + solverHandler.getLeftOffset()
						|| xi >= Mx - solverHandler.getRightOffset())
					continue;

				// Get the pointer to the beginning of the solution data for
				// this grid point
				gridPointSolution = solutionArray[zk][yj][xi];

				double hx = grid[xi + 1] - grid[xi];

				// Update the concentration in the network
				network.updateConcentrationsFromArray(gridPointSolution);

				// Get the total helium concentration at this grid point
				heConcentration += network.getTotalAtomConcentration(0) * hx
						* hy * hz;
				dConcentration += network.getTotalAtomConcentration(1) * hx * hy
						* hz;
				tConcentration += network.getTotalAtomConcentration(2) * hx * hy
						* hz;
			}
		}
	}

	// Get the current process ID
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Determine total concentrations for He, D, T.
	std::array<double, 3> myConcData { heConcentration, dConcentration,
			tConcentration };
	std::array<double, 3> totalConcData;

	MPI_Reduce(myConcData.data(), totalConcData.data(), myConcData.size(),
	MPI_DOUBLE,
	MPI_SUM, 0, xolotlComm);

	// Extract total He, D, T concentrations.  Values are valid only on rank 0.
	double totalHeConcentration = totalConcData[0];
	double totalDConcentration = totalConcData[1];
	double totalTConcentration = totalConcData[2];

	// Master process
	if (procId == 0) {

		// Get the total size of the grid rescale the concentrations
		PetscInt Mx, My, Mz;
		ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
		PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
		CHKERRQ(ierr);

		// Compute the total surface irradiated by the helium flux
		double surface = (double) (My * Mz) * hy * hz;

		// Rescale the concentration
		totalHeConcentration = totalHeConcentration / surface;
		totalDConcentration = totalDConcentration / surface;
		totalTConcentration = totalTConcentration / surface;

		// Get the fluence
		double fluence = fluxHandler->getFluence();

		// Print the result
		std::cout << "\nTime: " << time << std::endl;
		std::cout << "Helium content = " << totalHeConcentration << std::endl;
		std::cout << "Deuterium content = " << totalDConcentration << std::endl;
		std::cout << "Tritium content = " << totalTConcentration << std::endl;
		std::cout << "Fluence = " << fluence << "\n" << std::endl;

		// Uncomment to write the retention and the fluence in a file
		std::ofstream outputFile;
		outputFile.open("retentionOut.txt", ios::app);
		outputFile << fluence << " " << totalHeConcentration << " "
				<< totalDConcentration << " " << totalTConcentration
				<< std::endl;
		outputFile.close();
	}

	// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeXenonRetention3D")
/**
 * This is a monitoring method that will compute the xenon retention
 */
PetscErrorCode computeXenonRetention3D(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void*) {

	// Initial declarations
	PetscErrorCode ierr;
	PetscInt xs, xm, ys, ym, zs, zm;

	PetscFunctionBeginUser;

	// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

	// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

	// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);

	// Get the total size of the grid
	PetscInt Mx, My, Mz;
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

	// Get the physical grid
	auto grid = solverHandler.getXGrid();

	// Setup step size variables
	double hy = solverHandler.getStepSizeY();
	double hz = solverHandler.getStepSizeZ();

	// Get the network
	auto &network = solverHandler.getNetwork();

	// Get the complete data array, including ghost cells
	Vec localSolution;
	ierr = DMGetLocalVector(da, &localSolution);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, solution, INSERT_VALUES, localSolution);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, solution, INSERT_VALUES, localSolution);
	CHKERRQ(ierr);
	// Get the array of concentration
	PetscReal ****solutionArray, *gridPointSolution;
	ierr = DMDAVecGetArrayDOFRead(da, localSolution, &solutionArray);
	CHKERRQ(ierr);

	// Store the concentration and other values over the grid
	double xeConcentration = 0.0, bubbleConcentration = 0.0, radii = 0.0,
			partialBubbleConcentration = 0.0, partialRadii = 0.0;

	// Get the minimum size for the radius
	auto minSizes = solverHandler.getMinSizes();
	double sphereFactor = 4.0 * xolotlCore::pi / 3.0;

	// Loop on the grid
	for (PetscInt zk = zs; zk < zs + zm; zk++) {
		for (PetscInt yj = ys; yj < ys + ym; yj++) {
			for (PetscInt xi = xs; xi < xs + xm; xi++) {

				// Get the pointer to the beginning of the solution data for this grid point
				gridPointSolution = solutionArray[zk][yj][xi];

				double hx = grid[xi + 1] - grid[xi];

				// Update the concentration in the network
				network.updateConcentrationsFromArray(gridPointSolution);

				// Initialize the volume fraction
				double volumeFrac = 0.0;

				// Loop on all the indices
				for (unsigned int i = 0; i < indices3D.size(); i++) {
					// Add the current concentration times the number of xenon in the cluster
					// (from the weight vector)
					double conc = gridPointSolution[indices3D[i]];
					xeConcentration += conc * weights3D[i] * hx * hy * hz;
					bubbleConcentration += conc * hx * hy * hz;
					radii += conc * radii3D[i] * hx * hy * hz;
					if (weights3D[i] >= minSizes[0] && conc > 1.0e-16) {
						partialBubbleConcentration += conc * hx * hy * hz;
						partialRadii += conc * radii3D[i] * hx * hy * hz;
						// Update the volume fraction
						volumeFrac += gridPointSolution[indices3D[i]]
								* sphereFactor * pow(radii3D[i], 3.0);
					}
					// Set the monomer concentration
					if (weights3D[i] == 1)
						solverHandler.setMonomerConc(
								gridPointSolution[indices3D[i]], xi - xs,
								yj - ys, zk - zs);
				}

				// Loop on all the super clusters
				for (auto const &superMapItem : network.getAll(
						ReactantType::NESuper)) {
					auto const &cluster =
							static_cast<NESuperCluster&>(*(superMapItem.second));
					double conc = cluster.getTotalConcentration();
					xeConcentration += cluster.getTotalXenonConcentration() * hx
							* hy * hz;
					bubbleConcentration += conc * hx * hy * hz;
					radii += conc * cluster.getReactionRadius() * hx * hy * hz;
					if (cluster.getSize() >= minSizes[0] && conc > 1.0e-16) {
						partialBubbleConcentration += conc * hx * hy * hz;
						partialRadii += conc * cluster.getReactionRadius() * hx
								* hy * hz;
						// Update the volume fraction
						volumeFrac += cluster.getTotalConcentration()
								* sphereFactor
								* pow(cluster.getReactionRadius(), 3.0);
					}
				}

				// Set the volume fraction
				solverHandler.setVolumeFraction(volumeFrac, xi - xs, yj - ys,
						zk - zs);
			}
		}
	}

	// Get the current process ID
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

	// Sum all the concentrations through MPI reduce
	std::array<double, 5> myConcData { xeConcentration, bubbleConcentration,
			radii, partialBubbleConcentration, partialRadii };
	std::array<double, 5> totalConcData { 0.0, 0.0, 0.0, 0.0, 0.0 };
	MPI_Reduce(myConcData.data(), totalConcData.data(), myConcData.size(),
	MPI_DOUBLE, MPI_SUM, 0, xolotlComm);

	// GB
	// Get the delta time from the previous timestep to this timestep
	double dt = time - solverHandler.getPreviousTime();
	// Sum and gather the previous flux
	double globalXeFlux = 0.0;
	// Get the vector from the solver handler
	auto gbVector = solverHandler.getGBVector();
	// Get the previous flux vector
	auto &localNE = solverHandler.getLocalNE();
	// Loop on the GB
	for (auto const &pair : gbVector) {
		// Middle
		int xi = std::get<0>(pair);
		int yj = std::get<1>(pair);
		int zk = std::get<2>(pair);
		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym && zk >= zs
				&& zk < zs + zm) {
			double previousXeFlux = std::get<1>(
					localNE[xi - xs][yj - ys][zk - zs]);
			globalXeFlux += previousXeFlux * (grid[xi + 1] - grid[xi]) * hy
					* hz;
			// Set the amount in the vector we keep
			solverHandler.setLocalXeRate(previousXeFlux * dt, xi - xs, yj - ys,
					zk - zs);
		}
	}
	double totalXeFlux = 0.0;
	MPI_Reduce(&globalXeFlux, &totalXeFlux, 1, MPI_DOUBLE, MPI_SUM, 0,
			xolotlComm);

// Master process
	if (procId == 0) {
		// Get the previous value of Xe that went to the GB
		double nXenon = solverHandler.getNXeGB();
		// Compute the total number of Xe that went to the GB
		nXenon += totalXeFlux * dt;
		solverHandler.setNXeGB(nXenon);
	}

// Loop on the GB
	for (auto const &pair : gbVector) {
		// Local rate
		double localRate = 0.0;
		// Define left and right with reference to the middle point
		// Middle
		int xi = std::get<0>(pair);
		int yj = std::get<1>(pair);
		int zk = std::get<2>(pair);

		// Factor for finite difference
		double hxLeft = 0.0, hxRight = 0.0;
		if (xi - 1 >= 0 && xi < Mx) {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else if (xi - 1 < 0) {
			hxLeft = grid[xi + 1] - grid[xi];
			hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
		} else {
			hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
			hxRight = grid[xi + 1] - grid[xi];
		}
		double factor = 2.0 / (hxLeft + hxRight);

		// Check we are on the right proc
		if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym && zk >= zs
				&& zk < zs + zm) {

			// X segment
			// Left
			xi = std::get<0>(pair) - 1;
			// Get the Xe_1 cluster
			auto const &cluster = *(network.get(Species::Xe, 1));
			// Get its id
			int id = cluster.getId() - 1;
			// Get its size and diffusion coefficient
			int size = cluster.getSize();
			// Compute the flux coming from the left
			localRate += (double) size * solutionArray[zk][yj][xi][id]
					* cluster.getDiffusionCoefficient(xi + 1 - xs) * factor
					/ hxLeft;

			// Right
			xi = std::get<0>(pair) + 1;
			// Compute the flux coming from the right
			localRate += (double) size * solutionArray[zk][yj][xi][id]
					* cluster.getDiffusionCoefficient(xi + 1 - xs) * factor
					/ hxRight;

			// Y segment
			// Bottom
			xi = std::get<0>(pair);
			yj = std::get<1>(pair) - 1;
			// Compute the flux coming from the bottom
			localRate += (double) size * solutionArray[zk][yj][xi][id]
					* cluster.getDiffusionCoefficient(xi + 1 - xs) / (hy * hy);

			// Top
			yj = std::get<1>(pair) + 1;
			// Compute the flux coming from the top
			localRate += (double) size * solutionArray[zk][yj][xi][id]
					* cluster.getDiffusionCoefficient(xi + 1 - xs) / (hy * hy);

			// Z segment
			// Back
			yj = std::get<1>(pair);
			zk = std::get<2>(pair) - 1;
			// Compute the flux coming from the back
			localRate += (double) size * solutionArray[zk][yj][xi][id]
					* cluster.getDiffusionCoefficient(xi + 1 - xs) / (hz * hz);

			// Front
			zk = std::get<2>(pair) + 1;
			// Compute the flux coming from the front
			localRate += (double) size * solutionArray[zk][yj][xi][id]
					* cluster.getDiffusionCoefficient(xi + 1 - xs) / (hz * hz);

			// Middle
			xi = std::get<0>(pair);
			yj = std::get<1>(pair);
			zk = std::get<2>(pair);
			solverHandler.setPreviousXeFlux(localRate, xi - xs, yj - ys,
					zk - zs);
		}
	}

	// Free surface
	if (solverHandler.getLeftOffset() == 1) {
		if (procId == 0) {
			// Get the delta time from the previous timestep to this timestep
			double dt = time - solverHandler.getPreviousTime();
			// Compute the total number of impurities that went to the surface
			nXeSurf3D += previousXeFlux3D * dt;
		}

		// Get the Xe_1 cluster
		auto &cluster = *(network.get(Species::Xe, 1));
		// Get its id
		int id = cluster.getId() - 1;
		// Get its size and diffusion coefficient
		int size = cluster.getSize();
		// Init previous flux
		double localXeFlux3D = 0.0;

		// Loop on every Y position
		for (PetscInt k = 0; k < Mz; k++) {
			for (PetscInt j = 0; j < My; j++) {
				// Get the local surface position
				int xi = solverHandler.getSurfacePosition(j, k);

				// Check we are on the right proc
				if (xi >= xs && xi < xs + xm && j >= ys && j < ys + ym
						&& k >= zs && k < zs + zm) {
					// Factor for finite difference
					double hxLeft = 0.0, hxRight = 0.0;
					if (xi - 1 >= 0 && xi < Mx) {
						hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
						hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
					} else if (xi - 1 < 0) {
						hxLeft = grid[xi + 1] - grid[xi];
						hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
					} else {
						hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
						hxRight = grid[xi + 1] - grid[xi];
					}
					double factor = 2.0 * hy * hz / (hxLeft + hxRight);

					// Initialize the value for the flux
					double newFlux = 0.0;

					// Right
					xi += 1;
					// Compute the flux coming from the right
					localXeFlux3D += (double) size * solutionArray[k][j][xi][id]
							* cluster.getDiffusionCoefficient(xi + 1 - xs)
							* factor;
				}
			}
		}

		// Add the data from each process
		MPI_Reduce(&localXeFlux3D, &previousXeFlux3D, 1,
		MPI_DOUBLE, MPI_SUM, 0, xolotlComm);
	}

// Master process
	if (procId == 0) {
		// Compute the total surface irradiated
		double surface = (double) My * hy * (double) Mz * hz;
		// Get the number of Xe that went to the GB
		double nXenon = solverHandler.getNXeGB();

		totalConcData[0] = totalConcData[0] / surface;

		// Print the result
		std::cout << "\nTime: " << time << std::endl;
		std::cout << "Xenon concentration = " << totalConcData[0] << std::endl;
		std::cout << "Xenon GB = " << nXenon / surface << std::endl;
		std::cout << "Xenon Surf = " << nXeSurf3D / surface << std::endl
				<< std::endl;

		// Make sure the average partial radius makes sense
		double averagePartialRadius = totalConcData[4] / totalConcData[3];
		double minRadius = pow(
				(3.0 * (double) minSizes[0])
						/ (4.0 * xolotlCore::pi * network.getDensity()),
				(1.0 / 3.0));
		if (partialBubbleConcentration < 1.e-16
				|| averagePartialRadius < minRadius)
			averagePartialRadius = minRadius;

		// Uncomment to write the retention and the fluence in a file
		std::ofstream outputFile;
		outputFile.open("retentionOut.txt", ios::app);
		outputFile << time << " " << totalConcData[0] << " "
				<< totalConcData[2] / totalConcData[1] << " "
				<< averagePartialRadius << " " << nXenon / surface << " "
				<< nXeSurf3D / surface << std::endl;
		outputFile.close();
	}

// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, localSolution, &solutionArray);
	CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &localSolution);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "computeTRIDYN3D")
/**
 * This is a monitoring method that will compute the data to send to TRIDYN
 */
PetscErrorCode computeTRIDYN3D(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void *ictx) {
// Initial declarations
	PetscErrorCode ierr;
	PetscInt xs, xm, ys, ym, zs, zm;

	PetscFunctionBeginUser;

// Get the number of processes
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int worldSize;
	MPI_Comm_size(xolotlComm, &worldSize);

// Gets the process ID
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

// Get the network
	auto &network = solverHandler.getNetwork();

// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);

// Get the total size of the grid rescale the concentrations
	PetscInt Mx, My, Mz;
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

// Get the physical grid
	auto grid = solverHandler.getXGrid();

// Get the array of concentration
	double ****solutionArray, *gridPointSolution;
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

// Create the output file
	std::ofstream outputFile;
	if (procId == 0) {
		std::stringstream name;
		name << "TRIDYN_" << timestep << ".dat";
		outputFile.open(name.str());
	}

// Loop on the entire grid
	for (int xi = 0; xi < Mx; xi++) {
		// Set x
		double x = (grid[xi] + grid[xi + 1]) / 2.0 - grid[1];

		// Initialize the concentrations at this grid point
		double heLocalConc = 0.0, dLocalConc = 0.0, tLocalConc = 0.0,
				vLocalConc = 0.0, iLocalConc = 0.0;

		// Loop on the y
		for (PetscInt yj = ys; yj < ys + ym; yj++) {
			// Loop on the z
			for (PetscInt zk = zs; zk < zs + zm; zk++) {
				// Get the surface position
				int surfacePos = solverHandler.getSurfacePosition(yj, zk);
				// Boundary conditions
				if (xi < surfacePos + solverHandler.getLeftOffset()
						|| xi >= Mx - solverHandler.getRightOffset())
					continue;

				// If it is the locally owned part of the grid
				if (xi >= xs && xi < xs + xm) {
					// Get the pointer to the beginning of the solution data for this grid point
					gridPointSolution = solutionArray[zk][yj][xi];

					// Update the concentration in the network
					network.updateConcentrationsFromArray(gridPointSolution);

					// Get the total helium concentration at this grid point
					heLocalConc += gridPointSolution[0];
					dLocalConc += gridPointSolution[1];
					tLocalConc += network.getTotalAtomConcentration(2);
					vLocalConc += network.getTotalVConcentration();
					iLocalConc += network.getTotalIConcentration();
				}
			}
		}

		std::array<double, 5> myConcData { heLocalConc, dLocalConc, tLocalConc,
				vLocalConc, iLocalConc };
		std::array<double, 5> totalConcData = { 0.0, 0.0, 0.0, 0.0, 0.0 };

		MPI_Reduce(myConcData.data(), totalConcData.data(), myConcData.size(),
		MPI_DOUBLE,
		MPI_SUM, 0, xolotlComm);

		// The master process writes the cumulative value and writes in the file
		if (procId == 0) {
			outputFile
					<< x
							- (grid[solverHandler.getSurfacePosition(0, 0) + 1]
									- grid[1]) << " "
					<< totalConcData[0] / (My * Mz) << " "
					<< totalConcData[1] / (My * Mz) << " "
					<< totalConcData[2] / (My * Mz) << " "
					<< totalConcData[3] / (My * Mz) << " "
					<< totalConcData[4] / (My * Mz) << std::endl;
		}
	}

// Close the file
	if (procId == 0) {
		outputFile.close();
	}

// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorSurfaceXY3D")
/**
 * This is a monitoring method that will save 2D plots of the concentration of
 * a specific cluster at each grid point on the XY surface, integrating over Z.
 */
PetscErrorCode monitorSurfaceXY3D(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void*) {
// Initial declarations
	PetscErrorCode ierr;
	const double ****solutionArray, *gridPointSolution;
	PetscInt xs, xm, Mx, ys, ym, My, zs, zm, Mz;
	double x, y;

	PetscFunctionBeginUser;

// Don't do anything if it is not on the stride
	if (timestep % 10 != 0)
		PetscFunctionReturn(0);

// Gets the process ID
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);
// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

// Get the network
	auto &network = solverHandler.getNetwork();

// Get the physical grid in the x direction
	auto grid = solverHandler.getXGrid();

// Setup step size variables
	double hy = solverHandler.getStepSizeY();

// Choice of the cluster to be plotted
	int iCluster = 0;

// Create a Point vector to store the data to give to the data provider
// for the visualization
	auto myPoints = std::make_shared<std::vector<xolotlViz::Point> >();
// Create a point here so that it is not created and deleted in the loop
	xolotlViz::Point thePoint;

// Loop on the full grid, Y and X first because they are the axis of the plot
	for (PetscInt j = 0; j < My; j++) {
		// Compute y
		y = (double) j * hy;

		for (PetscInt i = 0; i < Mx; i++) {
			// Compute x
			x = (grid[i] + grid[i + 1]) / 2.0 - grid[1];

			// Initialize the value of the concentration to integrate over Z
			double conc = 0.0;

			for (PetscInt k = 0; k < Mz; k++) {
				// If it is the locally owned part of the grid
				if (i >= xs && i < xs + xm && j >= ys && j < ys + ym && k >= zs
						&& k < zs + zm) {
					// Get the pointer to the beginning of the solution data for this grid point
					gridPointSolution = solutionArray[k][j][i];

					// Integrate over Z
					conc += gridPointSolution[iCluster];
				}
			} // End of the loop on Z

			// Sum all the concentration on Z
			double totalConc = 0.0;
			MPI_Reduce(&conc, &totalConc, 1, MPI_DOUBLE, MPI_SUM, 0,
					xolotlComm);

			// If it is procId == 0 just store the integrated value in the myPoints vector
			if (procId == 0) {
				thePoint.value = totalConc;
				thePoint.t = time;
				thePoint.x = x;
				thePoint.y = y;
				myPoints->push_back(thePoint);
			}
		}
	}

// Plot everything from procId == 0
	if (procId == 0) {
		// Get the data provider and give it the points
		surfacePlotXY3D->getDataProvider()->setPoints(myPoints);

		// Get the iCluster cluster to have access to its name
		auto const &reactants = network.getAll();
		IReactant const &cluster = reactants.at(iCluster);

		// Change the title of the plot and the name of the data
		std::stringstream title;
		title << cluster.getName();
		surfacePlotXY3D->getDataProvider()->setDataName(title.str());
		title << " concentration";
		surfacePlotXY3D->plotLabelProvider->titleLabel = title.str();
		// Give the time to the label provider
		std::stringstream timeLabel;
		timeLabel << "time: " << std::setprecision(4) << time << "s";
		surfacePlotXY3D->plotLabelProvider->timeLabel = timeLabel.str();
		// Get the current time step
		PetscReal currentTimeStep;
		ierr = TSGetTimeStep(ts, &currentTimeStep);
		CHKERRQ(ierr);
		// Give the timestep to the label provider
		std::stringstream timeStepLabel;
		timeStepLabel << "dt: " << std::setprecision(4) << currentTimeStep
				<< "s";
		surfacePlotXY3D->plotLabelProvider->timeStepLabel = timeStepLabel.str();

		// Render and save in file
		std::stringstream fileName;
		fileName << cluster.getName() << "_surfaceXY_TS" << timestep << ".png";
		surfacePlotXY3D->write(fileName.str());
	}

// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "monitorSurfaceXZ3D")
/**
 * This is a monitoring method that will save 2D plots of the concentration of
 * a specific cluster at each grid point on the XZ surface, integrating over Y.
 */
PetscErrorCode monitorSurfaceXZ3D(TS ts, PetscInt timestep, PetscReal time,
		Vec solution, void*) {
// Initial declarations
	PetscErrorCode ierr;
	const double ****solutionArray, *gridPointSolution;
	PetscInt xs, xm, Mx, ys, ym, My, zs, zm, Mz;
	double x, z;

	PetscFunctionBeginUser;

// Don't do anything if it is not on the stride
	if (timestep % 10 != 0)
		PetscFunctionReturn(0);

// Gets the process ID
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);
// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

// Get the network
	auto &network = solverHandler.getNetwork();

// Get the physical grid in the x direction
	auto grid = solverHandler.getXGrid();

// Setup step size variables
	double hz = solverHandler.getStepSizeZ();

// Choice of the cluster to be plotted
	int iCluster = 0;

// Create a Point vector to store the data to give to the data provider
// for the visualization
	auto myPoints = std::make_shared<std::vector<xolotlViz::Point> >();
// Create a point here so that it is not created and deleted in the loop
	xolotlViz::Point thePoint;

// Loop on the full grid, Z and X first because they are the axis of the plot
	for (PetscInt k = 0; k < Mz; k++) {
		// Compute z
		z = (double) k * hz;

		for (PetscInt i = 0; i < Mx; i++) {
			// Compute x
			x = (grid[i] + grid[i + 1]) / 2.0 - grid[1];

			// Initialize the value of the concentration to integrate over Y
			double conc = 0.0;

			for (PetscInt j = 0; j < My; j++) {
				// If it is the locally owned part of the grid
				if (i >= xs && i < xs + xm && j >= ys && j < ys + ym && k >= zs
						&& k < zs + zm) {
					// Get the pointer to the beginning of the solution data for this grid point
					gridPointSolution = solutionArray[k][j][i];

					// Integrate over Y
					conc += gridPointSolution[iCluster];
				}
			} // End of the loop on Y

			// Sum all the concentration on Y
			double totalConc = 0.0;
			MPI_Reduce(&conc, &totalConc, 1, MPI_DOUBLE, MPI_SUM, 0,
					xolotlComm);

			// If it is procId == 0 just store the integrated value in the myPoints vector
			if (procId == 0) {
				thePoint.value = totalConc;
				thePoint.t = time;
				thePoint.x = x;
				thePoint.y = z;
				myPoints->push_back(thePoint);
			}
		}
	}

// Plot everything from procId == 0
	if (procId == 0) {
		// Get the data provider and give it the points
		surfacePlotXZ3D->getDataProvider()->setPoints(myPoints);

		// Get the iCluster cluster to have access to its name
		auto const &reactants = network.getAll();
		IReactant const &cluster = reactants.at(iCluster);

		// Change the title of the plot and the name of the data
		std::stringstream title;
		title << cluster.getName();
		surfacePlotXZ3D->getDataProvider()->setDataName(title.str());
		title << " concentration";
		surfacePlotXZ3D->plotLabelProvider->titleLabel = title.str();
		// Give the time to the label provider
		std::stringstream timeLabel;
		timeLabel << "time: " << std::setprecision(4) << time << "s";
		surfacePlotXZ3D->plotLabelProvider->timeLabel = timeLabel.str();
		// Get the current time step
		PetscReal currentTimeStep;
		ierr = TSGetTimeStep(ts, &currentTimeStep);
		CHKERRQ(ierr);
		// Give the timestep to the label provider
		std::stringstream timeStepLabel;
		timeStepLabel << "dt: " << std::setprecision(4) << currentTimeStep
				<< "s";
		surfacePlotXZ3D->plotLabelProvider->timeStepLabel = timeStepLabel.str();

		// Render and save in file
		std::stringstream fileName;
		fileName << cluster.getName() << "_surfaceXZ_TS" << timestep << ".png";
		surfacePlotXZ3D->write(fileName.str());
	}

// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "eventFunction3D")
/**
 * This is a method that checks if the surface should move or bursting happen
 */
PetscErrorCode eventFunction3D(TS ts, PetscReal time, Vec solution,
		PetscScalar *fvalue, void*) {
// Initial declaration
	PetscErrorCode ierr;
	double ****solutionArray, *gridPointSolution;
	PetscInt xs, xm, xi, Mx, ys, ym, yj, My, zs, zm, zk, Mz;
	fvalue[0] = 1.0, fvalue[1] = 1.0;
	depthPositions3D.clear();

	PetscFunctionBeginUser;

	PetscInt TSNumber = -1;
	ierr = TSGetStepNumber(ts, &TSNumber);

	// Skip if it is the same TS as before
	if (TSNumber == previousTSNumber3D)
		PetscFunctionReturn(0);

	// Set the previous TS number
	previousTSNumber3D = TSNumber;

// Gets the process ID
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

// Get the solutionArray
	ierr = DMDAVecGetArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);

// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

// Get the network
	auto &network = solverHandler.getNetwork();
// Get the physical grid and step size
	auto grid = solverHandler.getXGrid();
	double hy = solverHandler.getStepSizeY();
	double hz = solverHandler.getStepSizeZ();

// Get the flux handler to know the flux amplitude.
	auto fluxHandler = solverHandler.getFluxHandler();
	double heliumFluxAmplitude = fluxHandler->getFluxAmplitude();

// Get the delta time from the previous timestep to this timestep
	double dt = time - solverHandler.getPreviousTime();

// Work of the moving surface first
	if (solverHandler.moveSurface()) {
		// Write the initial surface positions
		if (procId == 0 && xolotlCore::equal(time, 0.0)) {
			std::ofstream outputFile;
			outputFile.open("surface.txt", ios::app);
			outputFile << time << " ";

			// Loop on the possible yj
			for (yj = 0; yj < My; yj++) {
				for (zk = 0; zk < Mz; zk++) {
					// Get the position of the surface at yj, zk
					int surfacePos = solverHandler.getSurfacePosition(yj, zk);
					outputFile << (double) yj * hy << " " << (double) zk * hz
							<< " " << grid[surfacePos + 1] - grid[1] << " ";
				}
			}
			outputFile << std::endl;
			outputFile.close();
		}

		// Get the initial vacancy concentration
		double initialVConc = solverHandler.getInitialVConc();

		// Loop on the possible zk and yj
		for (zk = 0; zk < Mz; zk++) {
			for (yj = 0; yj < My; yj++) {

				// Compute the total density of intersitials that escaped from the
				// surface since last timestep using the stored flux
				nInterstitial3D[yj][zk] += previousIFlux3D[yj][zk] * dt;

				// Remove the sputtering yield since last timestep
				nInterstitial3D[yj][zk] -= sputteringYield3D
						* heliumFluxAmplitude * dt;

				// Get the position of the surface at yj
				int surfacePos = solverHandler.getSurfacePosition(yj, zk);
				xi = surfacePos + solverHandler.getLeftOffset();

				// Initialize the value for the flux
				double newFlux = 0.0;

				// if xi is on this process
				if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym
						&& zk >= zs && zk < zs + zm) {
					// Get the concentrations at xi = surfacePos + 1
					gridPointSolution = solutionArray[zk][yj][xi];

					// Factor for finite difference
					double hxLeft = 0.0, hxRight = 0.0;
					if (xi - 1 >= 0 && xi < Mx) {
						hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
						hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
					} else if (xi - 1 < 0) {
						hxLeft = grid[xi + 1] - grid[xi];
						hxRight = (grid[xi + 2] - grid[xi]) / 2.0;
					} else {
						hxLeft = (grid[xi + 1] - grid[xi - 1]) / 2.0;
						hxRight = grid[xi + 1] - grid[xi];
					}
					double factor = 2.0 / (hxLeft + hxRight);

					// Loop on all the interstitial clusters to add the contribution from deeper
					for (auto const &iMapItem : network.getAll(ReactantType::I)) {
						// Get the cluster
						auto const &cluster = *(iMapItem.second);
						// Get its id and concentration
						int id = cluster.getId() - 1;
						double conc = gridPointSolution[id];
						// Get its size and diffusion coefficient
						int size = cluster.getSize();
						double coef = cluster.getDiffusionCoefficient(xi - xs);
						// Compute the flux going to the left
						newFlux += (double) size * factor * coef * conc;
					}
				}

				// Gather newFlux values at this position
				double newTotalFlux = 0.0;
				MPI_Allreduce(&newFlux, &newTotalFlux, 1, MPI_DOUBLE, MPI_SUM,
						xolotlComm);

				// Update the previous flux
				previousIFlux3D[yj][zk] = newTotalFlux;

				// Compare nInterstitials to the threshold to know if we should move the surface

				// The density of tungsten is 62.8 atoms/nm3, thus the threshold is
				double threshold = (62.8 - initialVConc)
						* (grid[xi] - grid[xi - 1]);
				if (nInterstitial3D[yj][zk] > threshold) {
					// The surface is moving
					fvalue[0] = 0.0;
				}

				// Moving the surface back
				else if (nInterstitial3D[yj][zk] < -threshold / 10.0) {
					// The surface is moving
					fvalue[0] = 0.0;
				}
			}
		}
	}

// Now work on the bubble bursting
	if (solverHandler.burstBubbles()) {
		// Compute the prefactor for the probability (arbitrary)
		double prefactor = heliumFluxAmplitude * dt
				* solverHandler.getBurstingFactor();

		// The depth parameter to know where the bursting should happen
		double depthParam = solverHandler.getTauBursting();	// nm
		// The number of He per V in a bubble
		double heVRatio = solverHandler.getHeVRatio();

		// For now we are not bursting
		bool burst = false;

		// Loop on the full grid
		for (zk = 0; zk < Mz; zk++) {
			for (yj = 0; yj < My; yj++) {
				// Get the surface position
				int surfacePos = solverHandler.getSurfacePosition(yj, zk);
				for (xi = surfacePos + solverHandler.getLeftOffset();
						xi < Mx - solverHandler.getRightOffset(); xi++) {

					// If this is the locally owned part of the grid
					if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym
							&& zk >= zs && zk < zs + zm) {

						// Get the pointer to the beginning of the solution data for this grid point
						gridPointSolution = solutionArray[zk][yj][xi];
						// Update the concentration in the network
						network.updateConcentrationsFromArray(
								gridPointSolution);

						// Get the distance from the surface
						double distance = (grid[xi] + grid[xi + 1]) / 2.0
								- grid[surfacePos + 1];

						// Compute the helium density at this grid point
						double heDensity = network.getTotalAtomConcentration();

						// Compute the radius of the bubble from the number of helium
						double nV = heDensity * (grid[xi + 1] - grid[xi]) * hy
								* hz / heVRatio;
						double latticeParam = network.getLatticeParameter();
						double tlcCubed = latticeParam * latticeParam
								* latticeParam;
						double radius = (sqrt(3.0) / 4) * latticeParam
								+ cbrt(
										(3.0 * tlcCubed * nV)
												/ (8.0 * xolotlCore::pi))
								- cbrt(
										(3.0 * tlcCubed)
												/ (8.0 * xolotlCore::pi));

						// If the radius is larger than the distance to the surface, burst
						if (radius > distance) {
							burst = true;
							depthPositions3D.push_back(
									std::make_tuple(zk, yj, xi));
							// Exit the loop
							continue;
						}
						// Add randomness
						double prob = prefactor
								* (1.0 - (distance - radius) / distance)
								* min(1.0,
										exp(
												-(distance - depthParam)
														/ (depthParam * 2.0)));
						double test = solverHandler.getRNG().GetRandomDouble();

						if (prob > test) {
							burst = true;
							depthPositions3D.push_back(
									std::make_tuple(zk, yj, xi));
						}
					}
				}
			}
		}

		// If at least one grid point is bursting
		if (burst) {
			// The event is happening
			fvalue[1] = 0.0;
		}
	}

// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOFRead(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ Actual__FUNCT__("xolotlSolver", "postEventFunction3D")
/**
 * This is a method that moves the surface or burst bubbles
 */
PetscErrorCode postEventFunction3D(TS ts, PetscInt nevents,
		PetscInt eventList[], PetscReal time, Vec solution, PetscBool, void*) {

// Initial declaration
	PetscErrorCode ierr;
	double ****solutionArray, *gridPointSolution;
	PetscInt xs, xm, xi, Mx, ys, ym, yj, My, zs, zm, zk, Mz;

	PetscFunctionBeginUser;

// Check if the surface has moved
	if (nevents == 0)
		PetscFunctionReturn(0);

// Gets the process ID
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);

// Get the solutionArray
	ierr = DMDAVecGetArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

// Get the corners of the grid
	ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
	CHKERRQ(ierr);

// Get the size of the total grid
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);

// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

// Get the network
	auto &network = solverHandler.getNetwork();
	int dof = network.getDOF();

// Get the physical grid
	auto grid = solverHandler.getXGrid();
// Get the step sizes
	double hy = solverHandler.getStepSizeY();
	double hz = solverHandler.getStepSizeZ();

// Take care of bursting

// Loop on each bursting depth
	for (int i = 0; i < depthPositions3D.size(); i++) {
		// Get the coordinates of the point
		int xi = std::get<2>(depthPositions3D[i]), yj = std::get<1>(
				depthPositions3D[i]), zk = std::get<0>(depthPositions3D[i]);
		// Get the pointer to the beginning of the solution data for this grid point
		gridPointSolution = solutionArray[zk][yj][xi];
		// Update the concentration in the network
		network.updateConcentrationsFromArray(gridPointSolution);

		// Get the surface position
		int surfacePos = solverHandler.getSurfacePosition(yj, zk);
		// Get the distance from the surface
		double distance = (grid[xi] + grid[xi + 1]) / 2.0
				- grid[surfacePos + 1];

		std::cout << "bursting at: " << zk * hz << " " << yj * hy << " "
				<< distance << std::endl;

		// Pinhole case
		// Consider each He to reset their concentration at this grid point
		for (auto const &heMapItem : network.getAll(ReactantType::He)) {
			auto const &cluster = *(heMapItem.second);

			int id = cluster.getId() - 1;
			gridPointSolution[id] = 0.0;
		}
		// Consider each D to reset their concentration at this grid point
		for (auto const &dMapItem : network.getAll(ReactantType::D)) {
			auto const &cluster = *(dMapItem.second);

			int id = cluster.getId() - 1;
			gridPointSolution[id] = 0.0;
		}
		// Consider each T to reset their concentration at this grid point
		for (auto const &tMapItem : network.getAll(ReactantType::T)) {
			auto const &cluster = *(tMapItem.second);

			int id = cluster.getId() - 1;
			gridPointSolution[id] = 0.0;
		}

		// Consider each HeV cluster to transfer their concentration to the V cluster of the
		// same size at this grid point
		for (auto const &heVMapItem : network.getAll(ReactantType::PSIMixed)) {
			auto const &cluster = *(heVMapItem.second);

			// Get the V cluster of the same size
			auto const &comp = cluster.getComposition();
			auto vCluster = network.get(Species::V,
					comp[toCompIdx(Species::V)]);
			int vId = vCluster->getId() - 1;
			int id = cluster.getId() - 1;
			gridPointSolution[vId] += gridPointSolution[id];
			gridPointSolution[id] = 0.0;
		}

		// Loop on the super clusters to transfer their concentration to the V cluster of the
		// same size at this grid point
		for (auto const &superMapItem : network.getAll(ReactantType::PSISuper)) {
			auto const &cluster =
					static_cast<PSISuperCluster&>(*(superMapItem.second));

			// Loop on the V boundaries
			for (auto const &j : cluster.getBounds(3)) {
				// Get the total concentration at this v
				double conc = cluster.getIntegratedVConcentration(j);
				// Get the corresponding V cluster and its Id
				auto vCluster = network.get(Species::V, j);
				int vId = vCluster->getId() - 1;
				// Add the concentration
				gridPointSolution[vId] += conc;
			}

			// Reset the super cluster concentration
			int id = cluster.getId() - 1;
			gridPointSolution[id] = 0.0;
			id = cluster.getMomentId(0) - 1;
			gridPointSolution[id] = 0.0;
			id = cluster.getMomentId(1) - 1;
			gridPointSolution[id] = 0.0;
			id = cluster.getMomentId(2) - 1;
			gridPointSolution[id] = 0.0;
			id = cluster.getMomentId(3) - 1;
			gridPointSolution[id] = 0.0;
		}
	}

// Now takes care of moving surface
	bool moving = false;
	for (int i = 0; i < nevents; i++) {
		if (eventList[i] == 0)
			moving = true;
	}

// Skip if nothing is moving
	if (!moving) {
		// Restore the solutionArray
		ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
		CHKERRQ(ierr);

		PetscFunctionReturn(0);
	}

// Get the initial vacancy concentration
	double initialVConc = solverHandler.getInitialVConc();

// Loop on the possible zk and yj
	for (zk = 0; zk < Mz; zk++) {
		for (yj = 0; yj < My; yj++) {
			// Get the position of the surface at yj
			int surfacePos = solverHandler.getSurfacePosition(yj, zk);
			xi = surfacePos + solverHandler.getLeftOffset();

			// The density of tungsten is 62.8 atoms/nm3, thus the threshold is
			double threshold = (62.8 - initialVConc)
					* (grid[xi] - grid[xi - 1]);

			// Move the surface up
			if (nInterstitial3D[yj][zk] > threshold) {
				int nGridPoints = 0;
				// Move the surface up until it is smaller than the next threshold
				while (nInterstitial3D[yj][zk] > threshold) {
					// Move the surface higher
					surfacePos--;
					xi = surfacePos + solverHandler.getLeftOffset();
					nGridPoints++;
					// Update the number of interstitials
					nInterstitial3D[yj][zk] -= threshold;
					// Update the thresold
					double threshold = (62.8 - initialVConc)
							* (grid[xi] - grid[xi - 1]);
				}

				// Throw an exception if the position is negative
				if (surfacePos < 0) {
					PetscBool flagCheck;
					ierr = PetscOptionsHasName(NULL, NULL, "-check_collapse",
							&flagCheck);
					CHKERRQ(ierr);
					if (flagCheck) {
						// Write the convergence reason
						std::ofstream outputFile;
						outputFile.open("solverStatus.txt");
						outputFile << "overgrid" << std::endl;
						outputFile.close();
					}
					throw std::string(
							"\nxolotlSolver::Monitor3D: The surface is trying to go outside of the grid!!");
				}

				// Printing information about the extension of the material
				if (procId == 0) {
					std::cout << "Adding " << nGridPoints
							<< " points to the grid on " << yj * hy << " and "
							<< zk * hz << " at time: " << time << " s."
							<< std::endl;
				}

				// Set it in the solver
				solverHandler.setSurfacePosition(surfacePos, yj, zk);

				// Initialize the vacancy concentration and the temperature on the new grid points
				// Get the single vacancy ID
				auto singleVacancyCluster = network.get(Species::V, 1);
				int vacancyIndex = -1;
				if (singleVacancyCluster)
					vacancyIndex = singleVacancyCluster->getId() - 1;
				// Get the surface temperature
				double temp = 0.0;
				if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym
						&& zk >= zs && zk < zs + zm) {
					temp = solutionArray[zk][yj][xi][dof - 1];
				}
				double surfTemp = 0.0;
				MPI_Allreduce(&temp, &surfTemp, 1, MPI_DOUBLE, MPI_SUM,
						xolotlComm);
				// Loop on the new grid points
				while (nGridPoints >= 0) {
					// Position of the newly created grid point
					xi = surfacePos + nGridPoints;

					// If xi is on this process
					if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym
							&& zk >= zs && zk < zs + zm) {
						// Get the concentrations
						gridPointSolution = solutionArray[zk][yj][xi];

						// Set the new surface temperature
						gridPointSolution[dof - 1] = surfTemp;

						if (vacancyIndex > 0 && nGridPoints > 0) {
							// Initialize the vacancy concentration
							gridPointSolution[vacancyIndex] = initialVConc;
						}
					}

					// Decrease the number of grid points
					--nGridPoints;
				}
			}

			// Moving the surface back
			else if (nInterstitial3D[yj][zk] < -threshold / 10.0) {
				// Move it back as long as the number of interstitials in negative
				while (nInterstitial3D[yj][zk] < 0.0) {
					// Compute the threshold to a deeper grid point
					threshold = (62.8 - initialVConc)
							* (grid[xi + 1] - grid[xi]);
					// Set all the concentrations to 0.0 at xi = surfacePos + 1
					// if xi is on this process
					if (xi >= xs && xi < xs + xm && yj >= ys && yj < ys + ym
							&& zk >= zs && zk < zs + zm) {
						// Get the concentrations at xi = surfacePos + 1
						gridPointSolution = solutionArray[zk][yj][xi];
						// Loop on DOF
						for (int i = 0; i < dof - 1; i++) {
							gridPointSolution[i] = 0.0;
						}
					}

					// Move the surface deeper
					surfacePos++;
					xi = surfacePos + solverHandler.getLeftOffset();
					// Update the number of interstitials
					nInterstitial3D[yj][zk] += threshold;
				}

				// Printing information about the extension of the material
				if (procId == 0) {
					std::cout << "Removing grid points to the grid on "
							<< yj * hy << " and " << zk * hz << " at time: "
							<< time << " s." << std::endl;
				}

				// Set it in the solver
				solverHandler.setSurfacePosition(surfacePos, yj, zk);
			}
		}
	}
// Get the modified trap-mutation handler to reinitialize it
	auto mutationHandler = solverHandler.getMutationHandler();
	auto advecHandlers = solverHandler.getAdvectionHandlers();

// Get the vector of positions of the surface
	std::vector<std::vector<int> > surfaceIndices;
	for (PetscInt i = 0; i < My; i++) {
		// Create a temporary vector
		std::vector<int> temp;
		for (PetscInt j = 0; j < Mz; j++) {
			temp.push_back(solverHandler.getSurfacePosition(i, j));
		}
		// Add the temporary vector to the vector of surface indices
		surfaceIndices.push_back(temp);
	}

	mutationHandler->initializeIndex3D(surfaceIndices, network, advecHandlers,
			grid, xm, xs, ym, hy, ys, zm, hz, zs);

// Write the surface positions
	if (procId == 0) {
		std::ofstream outputFile;
		outputFile.open("surface.txt", ios::app);
		outputFile << time << " ";

		// Loop on the possible yj
		for (yj = 0; yj < My; yj++) {
			for (zk = 0; zk < Mz; zk++) {
				// Get the position of the surface at yj, zk
				int surfacePos = solverHandler.getSurfacePosition(yj, zk);
				outputFile << (double) yj * hy << " " << (double) zk * hz << " "
						<< grid[surfacePos + 1] - grid[1] << " ";
			}
		}
		outputFile << std::endl;
		outputFile.close();
	}

// Restore the solutionArray
	ierr = DMDAVecRestoreArrayDOF(da, solution, &solutionArray);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/**
 * This operation sets up different monitors
 *  depending on the options.
 * @param ts The time stepper
 * @return A standard PETSc error code
 */
PetscErrorCode setupPetsc3DMonitor(TS &ts) {
	PetscErrorCode ierr;

// Get the process ID
	auto xolotlComm = xolotlCore::MPIUtils::getMPIComm();
	int procId;
	MPI_Comm_rank(xolotlComm, &procId);

// Get the xolotlViz handler registry
	auto vizHandlerRegistry = xolotlFactory::getVizHandlerRegistry();

// Flags to launch the monitors or not
	PetscBool flagCheck, flagPerf, flagHeRetention, flagXeRetention, flagStatus,
			flag2DXYPlot, flag2DXZPlot, flagTRIDYN, flagLargest;

// Check the option -check_collapse
	ierr = PetscOptionsHasName(NULL, NULL, "-check_collapse", &flagCheck);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-check_collapse) failed.");

// Check the option -plot_perf
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_perf", &flagPerf);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-plot_perf) failed.");

// Check the option -plot_2d_xy
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_2d_xy", &flag2DXYPlot);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-plot_2d_xy) failed.");

// Check the option -plot_2d_xz
	ierr = PetscOptionsHasName(NULL, NULL, "-plot_2d_xz", &flag2DXZPlot);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-plot_2d_xz) failed.");

// Check the option -helium_retention
	ierr = PetscOptionsHasName(NULL, NULL, "-helium_retention",
			&flagHeRetention);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-helium_retention) failed.");

// Check the option -xenon_retention
	ierr = PetscOptionsHasName(NULL, NULL, "-xenon_retention",
			&flagXeRetention);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-xenon_retention) failed.");

// Check the option -start_stop
	ierr = PetscOptionsHasName(NULL, NULL, "-start_stop", &flagStatus);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-start_stop) failed.");

// Check the option -tridyn
	ierr = PetscOptionsHasName(NULL, NULL, "-tridyn", &flagTRIDYN);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-tridyn) failed.");

	// Check the option -largest_conc
	ierr = PetscOptionsHasName(NULL, NULL, "-largest_conc", &flagLargest);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: PetscOptionsHasName (-largest_conc) failed.");

// Get the solver handler
	auto &solverHandler = PetscSolver::getSolverHandler();

// Get the network and its size
	auto &network = solverHandler.getNetwork();

// Determine if we have an existing restart file,
// and if so, it it has had timesteps written to it.
	std::unique_ptr<xolotlCore::XFile> networkFile;
	std::unique_ptr<xolotlCore::XFile::TimestepGroup> lastTsGroup;
	std::string networkName = solverHandler.getNetworkName();
	bool hasConcentrations = false;
	if (not networkName.empty()) {
		networkFile.reset(new xolotlCore::XFile(networkName));
		auto concGroup = networkFile->getGroup<
				xolotlCore::XFile::ConcentrationGroup>();
		hasConcentrations = (concGroup and concGroup->hasTimesteps());
		if (hasConcentrations) {
			lastTsGroup = concGroup->getLastTimestepGroup();
		}
	}

// Get the da from ts
	DM da;
	ierr = TSGetDM(ts, &da);
	CHKERRQ(ierr);
	checkPetscError(ierr, "setupPetsc3DMonitor: TSGetDM failed.");

// Get the total size of the grid
	PetscInt Mx, My, Mz;
	ierr = DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
	PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
	CHKERRQ(ierr);
	checkPetscError(ierr, "setupPetsc3DMonitor: DMDAGetInfo failed.");

// Set the post step processing to stop the solver if the time step collapses
	if (flagCheck) {
		// Find the threshold
		PetscBool flag;
		ierr = PetscOptionsGetReal(NULL, NULL, "-check_collapse",
				&timeStepThreshold, &flag);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: PetscOptionsGetReal (-check_collapse) failed.");
		if (!flag)
			timeStepThreshold = 1.0e-16;

		// Set the post step process that tells the solver when to stop if the time step collapse
		ierr = TSSetPostStep(ts, checkTimeStep);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSSetPostStep (checkTimeStep) failed.");
	}

// Set the monitor to save the status of the simulation in hdf5 file
	if (flagStatus) {
		// Find the stride to know how often the HDF5 file has to be written
		PetscBool flag;
		ierr = PetscOptionsGetReal(NULL, NULL, "-start_stop", &hdf5Stride3D,
				&flag);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: PetscOptionsGetReal (-start_stop) failed.");
		if (!flag)
			hdf5Stride3D = 1.0;

		// Compute the correct hdf5Previous3D for a restart
		if (hasConcentrations) {
			assert(lastTsGroup);

			// Get the previous time from the HDF5 file
			double previousTime = lastTsGroup->readPreviousTime();
			solverHandler.setPreviousTime(previousTime);
			hdf5Previous3D = (int) (previousTime / hdf5Stride3D);
		}

		// Don't do anything if both files have the same name
		if (hdf5OutputName3D != solverHandler.getNetworkName()) {

			// Get the solver handler
			auto &solverHandler = PetscSolver::getSolverHandler();

			// Get the physical grid in the x direction
			auto grid = solverHandler.getXGrid();

			// Setup step size variables
			double hy = solverHandler.getStepSizeY();
			double hz = solverHandler.getStepSizeZ();

			// Get the compostion list and save it
			auto compList = network.getCompositionList();

			// Create a checkpoint file.
			// Create and initialize a checkpoint file.
			// We do this in its own scope so that the file
			// is closed when the file object goes out of scope.
			// We want it to close before we (potentially) copy
			// the network from another file using a single-process
			// MPI communicator.
			{
				xolotlCore::XFile checkpointFile(hdf5OutputName3D, grid,
						compList, xolotlComm, My, hy, Mz, hz);
			}

			// Copy the network group from the given file (if it has one).
			// We open the files using a single-process MPI communicator
			// because it is faster for a single process to do the
			// copy with HDF5's H5Ocopy implementation than it is
			// when all processes call the copy function.
			// The checkpoint file must be closed before doing this.
			writeNetwork(xolotlComm, solverHandler.getNetworkName(),
					hdf5OutputName3D, network);
		}

		// startStop3D will be called at each timestep
		ierr = TSMonitorSet(ts, startStop3D, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (startStop3D) failed.");
	}

// If the user wants the surface to be able to move
	if (solverHandler.moveSurface() || solverHandler.burstBubbles()) {
		// Surface
		if (solverHandler.moveSurface()) {
			// Initialize nInterstitial3D and previousIFlux3D before monitoring the
			// interstitial flux
			for (PetscInt j = 0; j < My; j++) {
				// Create a one dimensional vector of double
				std::vector<double> tempVector;
				for (PetscInt k = 0; k < Mz; k++) {
					tempVector.push_back(0.0);
				}
				// Add the tempVector to nInterstitial3D and previousIFlux3D
				// to create their initial structure
				nInterstitial3D.push_back(tempVector);
				previousIFlux3D.push_back(tempVector);
			}

			// Get the interstitial information at the surface if concentrations were stored
			if (hasConcentrations) {
				// Get the interstitial quantity from the HDF5 file
				nInterstitial3D = lastTsGroup->readData3D("nInterstitial");
				// Get the previous I flux from the HDF5 file
				previousIFlux3D = lastTsGroup->readData3D("previousIFlux");
				// Get the previous time from the HDF5 file
				double previousTime = lastTsGroup->readPreviousTime();
				solverHandler.setPreviousTime(previousTime);
			}

			// Get the sputtering yield
			sputteringYield3D = solverHandler.getSputteringYield();

			// Master process
			if (procId == 0) {
				// Clear the file where the surface will be written
				std::ofstream outputFile;
				outputFile.open("surface.txt");
				outputFile.close();
			}
		}

		// Bursting
		if (solverHandler.burstBubbles()) {
			// No need to seed the random number generator here.
			// The solver handler has already done it.
		}

		// Set directions and terminate flags for the surface event
		PetscInt direction[2];
		PetscBool terminate[2];
		direction[0] = 0, direction[1] = 0;
		terminate[0] = PETSC_FALSE, terminate[1] = PETSC_FALSE;
		// Set the TSEvent
		ierr = TSSetEventHandler(ts, 2, direction, terminate, eventFunction3D,
				postEventFunction3D, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSSetEventHandler (eventFunction3D) failed.");
	}

// Set the monitor to save performance plots (has to be in parallel)
	if (flagPerf) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a ScatterPlot
			perfPlot = vizHandlerRegistry->getPlot("perfPlot",
					xolotlViz::PlotType::SCATTER);

			// Create and set the label provider
			auto labelProvider = std::make_shared<xolotlViz::LabelProvider>(
					"labelProvider");
			labelProvider->axis1Label = "Process ID";
			labelProvider->axis2Label = "Solver Time";

			// Give it to the plot
			perfPlot->setLabelProvider(labelProvider);

			// Create the data provider
			auto dataProvider = std::make_shared<xolotlViz::CvsXDataProvider>(
					"dataProvider");

			// Give it to the plot
			perfPlot->setDataProvider(dataProvider);
		}

		// monitorPerf will be called at each timestep
		ierr = TSMonitorSet(ts, monitorPerf, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (monitorPerf) failed.");
	}

// Set the monitor to compute the helium fluence for the retention calculation
	if (flagHeRetention) {

		// Get the previous time if concentrations were stored and initialize the fluence
		if (hasConcentrations) {
			// Get the previous time from the HDF5 file
			double previousTime = lastTsGroup->readPreviousTime();
			solverHandler.setPreviousTime(previousTime);
			// Initialize the fluence
			auto fluxHandler = solverHandler.getFluxHandler();
			// Increment the fluence with the value at this current timestep
			fluxHandler->computeFluence(previousTime);
		}

		// computeFluence will be called at each timestep
		ierr = TSMonitorSet(ts, computeFluence, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (computeFluence) failed.");

		// computeHeliumRetention3D will be called at each timestep
		ierr = TSMonitorSet(ts, computeHeliumRetention3D, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (computeHeliumRetention3D) failed.");

		// Master process
		if (procId == 0) {
			// Uncomment to clear the file where the retention will be written
			std::ofstream outputFile;
			outputFile.open("retentionOut.txt");
			outputFile.close();
		}
	}

// Set the monitor to compute the xenon fluence and the retention
// for the retention calculation
	if (flagXeRetention) {
		// Loop on the xenon clusters
		for (auto const &xeMapItem : network.getAll(ReactantType::Xe)) {
			auto const &cluster = *(xeMapItem.second);

			int id = cluster.getId() - 1;
			// Add the Id to the vector
			indices3D.push_back(id);
			// Add the number of xenon of this cluster to the weight
			weights3D.push_back(cluster.getSize());
			radii3D.push_back(cluster.getReactionRadius());
		}

		// Get the da from ts
		DM da;
		ierr = TSGetDM(ts, &da);
		checkPetscError(ierr, "setupPetsc3DMonitor: TSGetDM failed.");
		// Get the local boundaries
		PetscInt xm, ym, zm;
		ierr = DMDAGetCorners(da, NULL, NULL, NULL, &xm, &ym, &zm);
		checkPetscError(ierr, "setupPetsc3DMonitor: DMDAGetCorners failed.");
		// Create the local vectors on each process
		solverHandler.createLocalNE(xm, ym, zm);

		// Get the previous time if concentrations were stored and initialize the fluence
		if (hasConcentrations) {

			assert(lastTsGroup);

			// Get the previous time from the HDF5 file
			double previousTime = lastTsGroup->readPreviousTime();
			solverHandler.setPreviousTime(previousTime);
			// Initialize the fluence
			auto fluxHandler = solverHandler.getFluxHandler();
			// Increment the fluence with the value at this current timestep
			fluxHandler->computeFluence(previousTime);
		}

		// computeFluence will be called at each timestep
		ierr = TSMonitorSet(ts, computeFluence, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (computeFluence) failed.");

		// computeXenonRetention3D will be called at each timestep
		ierr = TSMonitorSet(ts, computeXenonRetention3D, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (computeXenonRetention3D) failed.");

		// Master process
		if (procId == 0) {
			// Uncomment to clear the file where the retention will be written
			std::ofstream outputFile;
			outputFile.open("retentionOut.txt");
			outputFile.close();
		}
	}

// Set the monitor to save surface plots of clusters concentration
	if (flag2DXYPlot) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a SurfacePlot
			surfacePlotXY3D = vizHandlerRegistry->getPlot("surfacePlotXY3D",
					xolotlViz::PlotType::SURFACE);

			// Create and set the label provider
			auto labelProvider = std::make_shared<xolotlViz::LabelProvider>(
					"labelProvider");
			labelProvider->axis1Label = "Depth (nm)";
			labelProvider->axis2Label = "Y (nm)";
			labelProvider->axis3Label = "Concentration";

			// Give it to the plot
			surfacePlotXY3D->setLabelProvider(labelProvider);

			// Create the data provider
			auto dataProvider = std::make_shared<xolotlViz::CvsXYDataProvider>(
					"dataProvider");

			// Give it to the plot
			surfacePlotXY3D->setDataProvider(dataProvider);
		}

		// monitorSurfaceXY3D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorSurfaceXY3D, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (monitorSurfaceXY3D) failed.");
	}

// Set the monitor to save surface plots of clusters concentration
	if (flag2DXZPlot) {
		// Only the master process will create the plot
		if (procId == 0) {
			// Create a SurfacePlot
			surfacePlotXZ3D = vizHandlerRegistry->getPlot("surfacePlotXZ3D",
					xolotlViz::PlotType::SURFACE);

			// Create and set the label provider
			auto labelProvider = std::make_shared<xolotlViz::LabelProvider>(
					"labelProvider");
			labelProvider->axis1Label = "Depth (nm)";
			labelProvider->axis2Label = "Z (nm)";
			labelProvider->axis3Label = "Concentration";

			// Give it to the plot
			surfacePlotXZ3D->setLabelProvider(labelProvider);

			// Create the data provider
			auto dataProvider = std::make_shared<xolotlViz::CvsXYDataProvider>(
					"dataProvider");

			// Give it to the plot
			surfacePlotXZ3D->setDataProvider(dataProvider);
		}

		// monitorSurfaceXZ3D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorSurfaceXZ3D, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (monitorSurfaceXZ3D) failed.");
	}

// Set the monitor to output data for TRIDYN
	if (flagTRIDYN) {
		// computeTRIDYN3D will be called at each timestep
		ierr = TSMonitorSet(ts, computeTRIDYN3D, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (computeTRIDYN3D) failed.");
	}

	// Set the monitor to monitor the concentration of the largest cluster
	if (flagLargest) {
		// Look for the largest cluster
		int largestSize = 0;
		auto const &reactants = network.getAll();
		for (int i = 0; i < reactants.size(); i++) {
			IReactant const &cluster = reactants.at(i);
			auto size = cluster.getSize();
			if (size > largestSize) {
				largestClusterId3D = cluster.getId() - 1;
				largestSize = size;
			}
		}

		// Find the threshold
		PetscBool flag;
		ierr = PetscOptionsGetReal(NULL, NULL, "-largest_conc",
				&largestThreshold3D, &flag);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: PetscOptionsGetReal (-largest_conc) failed.");

		// monitorLargest3D will be called at each timestep
		ierr = TSMonitorSet(ts, monitorLargest3D, NULL, NULL);
		checkPetscError(ierr,
				"setupPetsc3DMonitor: TSMonitorSet (monitorLargest3D) failed.");
	}

// Set the monitor to simply change the previous time to the new time
// monitorTime will be called at each timestep
	ierr = TSMonitorSet(ts, monitorTime, NULL, NULL);
	checkPetscError(ierr,
			"setupPetsc3DMonitor: TSMonitorSet (monitorTime) failed.");

	PetscFunctionReturn(0);
}

}

/* end namespace xolotlSolver */
