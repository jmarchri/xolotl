#ifndef ISOLVERHANDLER_H
#define ISOLVERHANDLER_H

// Includes
#include <petscsys.h>
#include <petscts.h>
#include <petscdmda.h>
#include <memory>
#include <xolotl/core/temperature/ITemperatureHandler.h>
#include <xolotl/core/diffusion/IDiffusionHandler.h>
#include <xolotl/core/advection/IAdvectionHandler.h>
#include <xolotl/core/modified/ITrapMutationHandler.h>
#include <xolotl/core/modified/IHeterogeneousNucleationHandler.h>
#include <xolotl/factory/material/IMaterialFactory.h>
#include <xolotl/core/network/IReactionNetwork.h>
#include <xolotl/util/NDArray.h>
#include <xolotl/util/RandomNumberGenerator.h>

namespace xolotl {
namespace solver {
namespace handler {

template<typename ValueType, typename SeedType>
class RandomNumberGenerator;

/**
 * Realizations of this interface are responsible for the actual implementation of
 * each piece of the solver. It is created to handle the multiple dimensions more easily.
 *
 * TODO - It will be better to have a PETSc-free solver handler interface (like ISolver)
 */
class ISolverHandler {

public:

	/**
	 * The destructor.
	 */
	virtual ~ISolverHandler() {
	}

	/**
	 * Initialize all the physics handlers that are needed to solve the ADR equations.
	 *
	 * @param material The material factory
	 * @param tempHandler The temperature handler
	 * @param options The Xolotl options
	 */
	virtual void initializeHandlers(
			std::shared_ptr<factory::material::IMaterialFactory> material,
			std::shared_ptr<core::temperature::ITemperatureHandler> tempHandler,
			const options::Options &opts) = 0;

	/**
	 * Create everything needed before starting to solve.
	 *
	 * @param da The PETSc distributed array
	 */
	virtual void createSolverContext(DM &da) = 0;

	/**
	 * Initialize the concentration solution vector.
	 *
	 * @param da The PETSc distributed array
	 * @param C The PETSc solution vector
	 */
	virtual void initializeConcentration(DM &da, Vec &C) = 0;

	/**
	 * Compute the new concentrations for the RHS function given an initial
	 * vector of concentrations.
	 *
	 * @param ts The PETSc time stepper
	 * @param localC The PETSc local solution vector
	 * @param F The updated PETSc solution vector
	 * @param ftime The real time
	 */
	virtual void updateConcentration(TS &ts, Vec &localC, Vec &F,
			PetscReal ftime) = 0;

	/**
	 * Compute the full Jacobian.
	 *
	 * @param ts The PETSc time stepper
	 * @param localC The PETSc local solution vector
	 * @param J The Jacobian
	 * @param ftime The real time
	 */
	virtual void computeJacobian(TS &ts, Vec &localC, Mat &J,
			PetscReal ftime) = 0;

	/**
	 * Get the grid in the x direction.
	 *
	 * @return The grid in the x direction
	 */
	virtual std::vector<double> getXGrid() const = 0;

	/**
	 * Get the step size in the y direction.
	 *
	 * @return The step size in the y direction
	 */
	virtual double getStepSizeY() const = 0;

	/**
	 * Get the step size in the z direction.
	 *
	 * @return The step size in the z direction
	 */
	virtual double getStepSizeZ() const = 0;

	/**
	 * Get the number of dimensions of the problem.
	 *
	 * @return The number of dimensions
	 */
	virtual int getDimension() const = 0;

	/**
	 * Get the position of the surface.
	 *
	 * @param j The index on the grid in the y direction
	 * @param k The index on the grid in the z direction
	 * @return The position of the surface at this y,z coordinates
	 */
	virtual int getSurfacePosition(int j = -1, int k = -1) const = 0;

	/**
	 * Set the position of the surface.
	 *
	 * @param pos The index of the position
	 * @param j The index on the grid in the y direction
	 * @param k The index on the grid in the z direction
	 */
	virtual void setSurfacePosition(int pos, int j = -1, int k = -1) = 0;

	/**
	 * Get the initial vacancy concentration.
	 *
	 * @return The initial vacancy concentration
	 */
	virtual double getInitialVConc() const = 0;

	/**
	 * Get the sputtering yield.
	 *
	 * @return The sputtering yield
	 */
	virtual double getSputteringYield() const = 0;

	/**
	 * Get the bursting depth parameter.
	 *
	 * @return The depth parameter
	 */
	virtual double getTauBursting() const = 0;

	/**
	 * Get the grid left offset.
	 *
	 * @return The offset
	 */
	virtual int getLeftOffset() const = 0;

	/**
	 * Get the grid right offset.
	 *
	 * @return The offset
	 */
	virtual int getRightOffset() const = 0;

	/**
	 * To know if the surface should be able to move.
	 *
	 * @return True if the surface should be able to move.
	 */
	virtual bool moveSurface() const = 0;

	/**
	 * To know if the bubble bursting should be used.
	 *
	 * @return True if we want the bubble bursting.
	 */
	virtual bool burstBubbles() const = 0;

	/**
	 * Get the minimum size for computing average radius.
	 *
	 * @return The minimum size
	 */
	virtual util::Array<int, 4> getMinSizes() const = 0;

	/**
	 * Get the flux handler.
	 *
	 * @return The flux handler
	 */
	virtual core::flux::IFluxHandler* getFluxHandler() const = 0;

	/**
	 * Get the temperature handler.
	 *
	 * @return The temperature handler
	 */
	virtual core::temperature::ITemperatureHandler* getTemperatureHandler() const = 0;

	/**
	 * Get the diffusion handler.
	 *
	 * @return The diffusion handler
	 */
	virtual core::diffusion::IDiffusionHandler* getDiffusionHandler() const = 0;

	/**
	 * Get the advection handler.
	 *
	 * @return The first advection handler
	 */
	virtual core::advection::IAdvectionHandler* getAdvectionHandler() const = 0;

	/**
	 * Get the advection handlers.
	 *
	 * @return The first advection handlers
	 */
	virtual std::vector<core::advection::IAdvectionHandler*> getAdvectionHandlers() const = 0;

	/**
	 * Get the modified trap-mutation handler.
	 *
	 * @return The modified trap-mutation handler
	 */
	virtual core::modified::ITrapMutationHandler* getMutationHandler() const = 0;

	/**
	 * Get the heterogeneous nucleation handler.
	 *
	 * @return The heterogeneous nucleation handler
	 */
	virtual core::modified::IHeterogeneousNucleationHandler* getHeterogeneousNucleationHandler() const = 0;

	/**
	 * Get the network.
	 *
	 * @return The network
	 */
	virtual core::network::IReactionNetwork& getNetwork() const = 0;

	/**
	 * Get the network name.
	 *
	 * @return The network name
	 */
	virtual std::string getNetworkName() const = 0;

	/**
	 * Access the random number generator.
	 * The generator will have already been seeded.
	 *
	 * @return The RandomNumberGenerator object to use.
	 */
	virtual util::RandomNumberGenerator<int, unsigned int>& getRNG(void) const = 0;

	/**
	 * Get the vector containing the location of GB.
	 *
	 * @return The GB vector
	 */
	virtual std::vector<std::tuple<int, int, int> > getGBVector() const = 0;

};
//end class ISolverHandler

} /* namespace handler */
} /* namespace solver */
} /* namespace xolotl */
#endif
