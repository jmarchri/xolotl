#pragma once

#include <cmath>
#include <fstream>
#include <iostream>

#include <xolotl/core/flux/FluxHandler.h>
#include <xolotl/util/Filesystem.h>
#include <xolotl/util/MPIUtils.h>
#include <xolotl/util/TokenizedLineReader.h>

namespace xolotl
{
namespace core
{
namespace flux
{
/**
 * This class realizes the FluxHandler interface for custom input data.
 */
class CustomFitFluxHandler : public FluxHandler
{
private:
	/**
	 * Parameters for the polynomial fits that will be read from a file
	 */
	std::vector<std::vector<double>> fitParams;

	/**
	 * The total depths on which the flux is defined.
	 */
	std::vector<double> totalDepths;

	/**
	 * The reduction factors for each deposition.
	 */
	std::vector<double> reductionFactors;

	/**
	 * Value of the fit function integrated on the grid.
	 */
	std::vector<double> normFactors;

	/**
	 * File path for custom profiles
	 */
	fs::path profileFilePath;

	using FluxHandler::FitFunction;

	/**
	 * \see FluxHandler.h
	 */
	double
	FitFunction(double x, int i)
	{
		if (x > totalDepths[i])
			return 0.0;

		// Compute the polynomial fit
		auto params = fitParams[i];
		double value = params[0] + params[1] * x + params[2] * pow(x, 2.0) +
			params[3] * pow(x, 3.0) + params[4] * pow(x, 4.0) +
			params[5] * pow(x, 5.0) + params[6] * pow(x, 6.0) +
			params[7] * pow(x, 7.0) + params[8] * pow(x, 8.0) +
			params[9] * pow(x, 9.0) + params[10] * pow(x, 10.0) +
			params[11] * pow(x, 11.0) + params[12] * pow(x, 12.0) +
			params[13] * pow(x, 13.0) + params[14] * pow(x, 14.0) +
			params[15] * pow(x, 15.0);

		return std::max(value, 0.0);
	}

public:
	/**
	 * The constructor
	 */
	CustomFitFluxHandler(const options::IOptions& options) :
		FluxHandler(options),
		profileFilePath(options.getFluxDepthProfileFilePath())
	{
	}

	/**
	 * The Destructor
	 */
	~CustomFitFluxHandler()
	{
	}

	/**
	 * \see IFluxHandler.h
	 */
	void
	initializeFluxHandler(network::IReactionNetwork& network, int surfacePos,
		std::vector<double> grid)
	{
		// Clear everything
		incidentFluxVec.clear();
		fluxIndices.clear();
		amplitudes.clear();
		fitParams.clear();
		totalDepths.clear();
		reductionFactors.clear();
		normFactors.clear();

		// Set the grid
		xGrid = grid;

		// Read the parameter file
		fs::ifstream paramFile(profileFilePath);

		// Gets the process ID
		int procId;
		auto xolotlComm = util::getMPIComm();
		MPI_Comm_rank(xolotlComm, &procId);

		if (!paramFile.good()) {
			// Print a message
			if (procId == 0)
				std::cout
					<< "No parameter files for custom flux, the flux will be 0"
					<< std::endl;
		}
		else {
			// Build an input stream from the string
			util::TokenizedLineReader<std::string> reader;
			// Get the line
			std::string line;
			getline(paramFile, line);
			auto lineSS = std::make_shared<std::istringstream>(line);
			reader.setInputStream(lineSS);

			using AmountType = network::IReactionNetwork::AmountType;

			// Read the first line
			auto tokens = reader.loadLine();
			// And start looping on the lines
			int index = 0;
			while (tokens.size() > 0) {
				auto comp =
					std::vector<AmountType>(network.getSpeciesListSize(), 0);

				// Read the cluster type
				auto clusterSpecies = network.parseSpeciesId(tokens[0]);
				// Get the cluster
				comp[clusterSpecies()] = std::stoi(tokens[1]);
				auto clusterId = network.findClusterId(comp);
				// Check that it is present in the network
				if (clusterId == network::IReactionNetwork::invalidIndex()) {
					throw std::runtime_error(
						"\nThe requested cluster is not present "
						"in the network: " +
						tokens[0] + "_" + tokens[1] +
						", cannot use the flux option!");
				}
				else
					fluxIndices.push_back(clusterId);

				// Get the reduction factor
				reductionFactors.push_back(std::stod(tokens[2]));

				// Check if the reduction factor is positive
				if (reductionFactors[index] < 0.0) {
					// Print a message
					if (procId == 0)
						std::cout
							<< "One of the reduction factors for the custom "
							   "flux is negative, "
							   "check if this is really what you want to do."
							<< std::endl;
				}

				// Set the parameters for the fit
				getline(paramFile, line);
				lineSS = std::make_shared<std::istringstream>(line);
				reader.setInputStream(lineSS);
				tokens = reader.loadLine();
				std::vector<double> params;
				params.push_back(std::stod(tokens[0]));
				params.push_back(std::stod(tokens[1]));
				params.push_back(std::stod(tokens[2]));
				params.push_back(std::stod(tokens[3]));
				params.push_back(std::stod(tokens[4]));
				params.push_back(std::stod(tokens[5]));
				params.push_back(std::stod(tokens[6]));
				params.push_back(std::stod(tokens[7]));
				params.push_back(std::stod(tokens[8]));
				params.push_back(std::stod(tokens[9]));
				params.push_back(std::stod(tokens[10]));
				params.push_back(std::stod(tokens[11]));
				params.push_back(std::stod(tokens[12]));
				params.push_back(std::stod(tokens[13]));
				params.push_back(std::stod(tokens[14]));
				params.push_back(std::stod(tokens[15]));
				fitParams.push_back(params);
				// Set the total depth where the fit is defined
				totalDepths.push_back(std::stod(tokens[16]) + 0.1);

				if (xGrid.size() > 0) {
					// Compute the norm factor because the fit function has an
					// arbitrary amplitude
					normFactors.push_back(0.0);
					incidentFluxVec.push_back(std::vector<double>());
					// Loop on the x grid points skipping the first after the
					// surface position and last because of the boundary
					// conditions
					for (int i = surfacePos + 1; i < xGrid.size() - 3; i++) {
						// Get the x position
						double x = (xGrid[i] + xGrid[i + 1]) / 2.0 -
							xGrid[surfacePos + 1];

						// Add the the value of the function times the step size
						normFactors[index] +=
							FitFunction(x, index) * (xGrid[i + 1] - xGrid[i]);
					}

					// Factor the incident flux will be multiplied by to get
					// the wanted intensity
					double fluxNormalized = 0.0;
					if (normFactors[index] > 0.0) {
						fluxNormalized = fluxAmplitude *
							reductionFactors[index] / normFactors[index];
					}

					// Clear the flux vector
					incidentFluxVec[index].clear();
					// The first value corresponding to the surface position
					// should always be 0.0
					incidentFluxVec[index].push_back(0.0);

					// Starts at i = surfacePos + 1 because the first value was
					// already put in the vector
					for (int i = surfacePos + 1; i < xGrid.size() - 3; i++) {
						// Get the x position
						auto x = (xGrid[i] + xGrid[i + 1]) / 2.0 -
							xGrid[surfacePos + 1];

						// Compute the flux value
						double incidentFlux =
							fluxNormalized * FitFunction(x, index);
						// Add it to the vector
						incidentFluxVec[index].push_back(incidentFlux);
					}

					// The last value should always be 0.0 because of boundary
					// conditions
					incidentFluxVec[index].push_back(0.0);
				}

				// Read the next line
				getline(paramFile, line);
				lineSS = std::make_shared<std::istringstream>(line);
				reader.setInputStream(lineSS);
				tokens = reader.loadLine();
				// Increase the index
				index++;
			}

			// Prints both incident vectors in a file
			if (procId == 0 && incidentFluxVec.size() > 0) {
				std::ofstream outputFile;
				outputFile.open("incidentVectors.txt");
				for (int i = 0; i < incidentFluxVec[0].size(); i++) {
					outputFile
						<< (grid[surfacePos + i] + grid[surfacePos + i + 1]) /
								2.0 -
							grid[surfacePos + 1]
						<< " ";
					for (int j = 0; j < index; j++)
						outputFile << incidentFluxVec[j][i] << " ";
					outputFile << std::endl;
				}
				outputFile.close();
			}
		}

		// Close the file
		paramFile.close();

		return;
	}

	/**
	 * \see IFluxHandler.h
	 */
	void
	computeIncidentFlux(
		double currentTime, double* updatedConcOffset, int xi, int surfacePos)
	{
		// Recompute the flux vector if a time profile is used
		if (useTimeProfile) {
			fluxAmplitude = getProfileAmplitude(currentTime);
			recomputeFluxHandler(surfacePos);
		}

		if (xGrid.size() == 0) {
			// Update the concentration array
			for (int i = 0; i < fluxIndices.size(); i++) {
				updatedConcOffset[fluxIndices[i]] +=
					fluxAmplitude * reductionFactors[i];
			}

			return;
		}

		// Update the concentration array
		for (int i = 0; i < fluxIndices.size(); i++) {
			updatedConcOffset[fluxIndices[i]] +=
				incidentFluxVec[i][xi - surfacePos];
		}

		return;
	}

	/**
	 * \see IFluxHandler.h
	 */
	void
	recomputeFluxHandler(int surfacePos)
	{
		// Loop on the different types of clusters
		for (int index = 0; index < fluxIndices.size(); index++) {
			// Factor the incident flux will be multiplied by
			double fluxNormalized = 0.0;
			if (normFactors[index] > 0.0)
				fluxNormalized = fluxAmplitude * reductionFactors[index] /
					normFactors[index];

			// Starts a i = surfacePos + 1 because the first values were already
			// put in the vector
			for (int i = surfacePos + 1; i < xGrid.size() - 3; i++) {
				// Get the x position
				auto x =
					(xGrid[i] + xGrid[i + 1]) / 2.0 - xGrid[surfacePos + 1];

				// Compute the flux value
				double incidentFlux = fluxNormalized * FitFunction(x, index);
				// Add it to the vector
				incidentFluxVec[index][i - surfacePos] = incidentFlux;
			}
		}

		return;
	}
};
// end class CustomFitFluxHandler

} // namespace flux
} // namespace core
} // namespace xolotl
