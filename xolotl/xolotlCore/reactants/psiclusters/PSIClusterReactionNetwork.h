#ifndef PSI_CLUSTER_REACTION_NETWORK_H
#define PSI_CLUSTER_REACTION_NETWORK_H

// Includes
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <algorithm>
#include "ReactionNetwork.h"
#include "PSISuperCluster.h"
#include "ReactantType.h"

namespace xolotlCore {

/**
 *  This class manages the set of reactants and compound reactants (
 *  combinations of normal reactants) for PSI clusters. It also manages a
 *  set of properties that describes the total collection.
 *
 *  This class is a very heavyweight class that should not be abused.
 *
 *  Reactants that are added to this network must be added as with shared_ptrs.
 *  Furthermore, reactants that are added to this network have their ids set to
 *  a network specific id. Reactants should not be shared between separate
 *  instances of a PSIClusterReactionNetwork.
 */
class PSIClusterReactionNetwork: public ReactionNetwork {

private:
	/**
	 * Concise name for map supporting quick lookup of supercluster containing
	 * specifc number of He and V.
	 *
	 * We could use a map, but because we expect it to be dense (i.e.,
	 * most pairs of He and V counts will have a valid super cluster),
	 * a 2D matrix indexed by nHe and nV gives better performance for
	 * lookups without costing too much more (if any) in terms of memory.
	 * We use a vector of vectors instead of array of arrays because
	 * we don't know the dimensions at compile time.  We use a vector of
	 * vectors instead of a single vector with more complex indexing
	 * to simplify indexing.  It may be worthwhile to compare performance
	 * with a single vector implementation to see if the single
	 * memory allocation ends up giving better performance.
	 */
	using HeVToSuperClusterMap = std::vector<std::vector<ReactantMap::const_iterator> >;

	/**
	 * Map supporting quick identification of super cluster containing
	 * given number of He and V.
	 */
	HeVToSuperClusterMap superClusterLookupMap;

	//! The dimension of the phase space
	int psDim = 0;

	//! The indexList.
	Array<int, 5> indexList;

	/**
	 * Calculate the dissociation constant of the first cluster with respect to
	 * the single-species cluster of the same type based on the current clusters
	 * atomic volume, reaction rate constant, and binding energies.
	 *
	 * @param reaction The reaction
	 * @param i The location on the grid in the depth direction
	 * @return The dissociation constant
	 */
	double calculateDissociationConstant(
			const DissociationReaction& reaction, int i) override;

	/**
	 * Calculate the binding energy for the dissociation cluster to emit the single
	 * and second cluster.
	 *
	 * @param reaction The reaction
	 * @param i The location on the grid in the depth direction
	 * @return The binding energy corresponding to this dissociation
	 */
	double computeBindingEnergy(const DissociationReaction& reaction) const
			override;

	/**
	 * Find the super cluster that contains the original cluster with nHe
	 * helium atoms and nV vacancies.
	 *
	 * @param nHe The number of helium atoms
	 * @param nD The number of deuterium atoms
	 * @param nT The number of tritium atoms
	 * @param nV The number of vacancies
	 * @return The super cluster representing the cluster with nHe helium
	 * and nV vacancies, or nullptr if no such cluster exists.
	 */
	IReactant * getSuperFromComp(IReactant::SizeType nHe,
			IReactant::SizeType nD, IReactant::SizeType nT,
			IReactant::SizeType nV) const override;

	ProductionReaction& defineReactionBase(IReactant& r1, IReactant& r2,
			int a[4] = defaultInit, bool secondProduct = false)
					__attribute__((always_inline)) {

		// Add a production reaction to our network.
		std::unique_ptr<ProductionReaction> reaction(
				new ProductionReaction(r1, r2));
		auto& prref = add(std::move(reaction));

		// Tell the reactants that they are involved in this reaction
		// if this is not the second product
		if (secondProduct)
			return prref;

		r1.participateIn(prref, a);
		r2.participateIn(prref, a);

		return prref;
	}

	void defineAnnihilationReaction(IReactant& r1, IReactant& r2,
			IReactant& product) {

		// Define the basic reaction.
		auto& reaction = defineReactionBase(r1, r2);

		// Tell the product it results from the reaction.
		product.resultFrom(reaction);
	}

	void defineCompleteAnnihilationReaction(IReactant& r1, IReactant& r2) {

		// Define the basic reaction
		defineReactionBase(r1, r2);

		// Nothing else to do since there is no product.
	}

	void defineProductionReaction(IReactant& r1, IReactant& super,
			IReactant& product, int a[4] = defaultInit, int b[4] = defaultInit,
			bool secondProduct = false) {
		// Define the basic production reaction.
		auto& reaction = defineReactionBase(r1, super, b, secondProduct);

		// Tell product it is a product of this reaction.
		product.resultFrom(reaction, a, b);

		if (secondProduct)
			return;

		// Check if reverse reaction is allowed.
		checkForDissociation(product, reaction, a, b);
	}

	/**
	 * Define a batch of production reactions for the given
	 * pair of reactants.
	 *
	 * @param r1 A reactant involved in a production reaction.
	 * @param r2 The super reactant involved in a production reaction.
	 * @param pris Information about reactants are involved with each reaction.
	 * @param secondProduct If we are setting the reaction for the second product.
	 */
	void defineProductionReactions(IReactant& r1, IReactant& super,
			const std::vector<PendingProductionReactionInfo>& pris,
			bool secondProduct = false);

	/**
	 * Define a batch of production reactions for the given
	 * pair of reactants.
	 *
	 * @param r1 A reactant involved in a production reaction.
	 * @param r2 The super reactant involved in a production reaction.
	 * @param product The cluster created by the reaction.
	 * @param secondProduct If we are setting the reaction for the second product.
	 */
	void defineAnaProductionReactions(IReactant& r1, IReactant& super,
			IReactant& product, bool secondProduct = false);

	// TODO should we default a, b, c, d to 0?
	void defineDissociationReaction(ProductionReaction& forwardReaction,
			IReactant& emitting, int a[4] = defaultInit,
			int b[4] = defaultInit) {

		std::unique_ptr<DissociationReaction> dissociationReaction(
				new DissociationReaction(emitting, forwardReaction.first,
						forwardReaction.second, &forwardReaction));
		auto& drref = add(std::move(dissociationReaction));

		// Tell the reactants that they are in this reaction
		forwardReaction.first.participateIn(drref, a, b);
		forwardReaction.second.participateIn(drref, a, b);
		emitting.emitFrom(drref, a);
	}

	/**
	 * Define a batch of dissociation reactions for the given
	 * forward reaction.
	 *
	 * @param forwardReaction The forward reaction in question.
	 * @param prodMap Map of reaction parameters, keyed by the product
	 * of the reaction.
	 */
	// TODO possible to use a ref for the key?
	using ProductToProductionMap =
	std::unordered_map<IReactant*, std::vector<PendingProductionReactionInfo> >;

	void defineDissociationReactions(ProductionReaction& forwardReaction,
			const ProductToProductionMap& prodMap);

	/**
	 * Define a batch of production dissociation reactions for the given
	 * forward reaction.
	 *
	 * @param forwardReaction The forward reaction in question.
	 * @param emitting The cluster emitting.
	 */
	void defineAnaDissociationReactions(ProductionReaction& forwardReaction,
			IReactant& emitting);

	/**
	 * Check whether dissociation reaction is allowed for
	 * given production reaction.
	 *
	 * @param emittingReactant The reactant that would emit the pair
	 * @param reaction The reaction to test.
	 * @return true iff dissociation for the given reaction is allowed.
	 */
	bool canDissociate(IReactant& emittingReactant,
			ProductionReaction& reaction) const;

	/**
	 * Add the dissociation connectivity for the reverse reaction if it is allowed.
	 *
	 * @param emittingReactant The reactant that would emit the pair
	 * @param reaction The reaction we want to reverse
	 * @param a The helium number for the emitting superCluster
	 * @param b The vacancy number for the emitting superCluster
	 *
	 */
	void checkForDissociation(IReactant& emittingReactant,
			ProductionReaction& reaction, int a[4] = defaultInit, int b[4] =
					defaultInit);

	/**
	 * Determine the column indices for partials, He momentum partials,
	 * or V momentum partials.
	 *
	 * @param reactantIndex  Index within PETSc data of slice for desired
	 *      type of partials.
	 * @param size Array containing number of valid partials for given reactant.
	 * @param indices Array containing the indices of the valid partials
	 *      for the given reactant.
	 */
	void FindPartialsColumnIndices(size_t reactantIndex, std::vector<int>& size,
			int* indices) const;

	/**
	 * Determine if the reaction is possible given then reactants and product
	 *
	 * @param r1 First reactant.
	 * @param r2 Second reactant.
	 * @param prod Potential product.
	 */
	bool checkOverlap(PSICluster& r1, PSICluster& r2, PSICluster& prod) {
		// Check if an interstitial cluster is involved
		int iSize = 0;
		if (r1.getType() == ReactantType::I) {
			iSize = r1.getSize();
		} else if (r2.getType() == ReactantType::I) {
			iSize = r2.getSize();
		}

		// Loop on the different type of clusters in grouping
		for (int i = 1; i < 5; i++) {
			// Check the boundaries in all the directions
			auto const& bounds = prod.getBounds(i - 1);
			int productLo = *(bounds.begin()), productHi = *(bounds.end()) - 1;
			auto const& r1Bounds = r1.getBounds(i - 1);
			int r1Lo = *(r1Bounds.begin()), r1Hi = *(r1Bounds.end()) - 1;
			auto const& r2Bounds = r2.getBounds(i - 1);
			int r2Lo = *(r2Bounds.begin()), r2Hi = *(r2Bounds.end()) - 1;

			// Compute the corresponding overlap width
			int width = std::min(productHi, r1Hi + r2Hi)
					- std::max(productLo, r1Lo + r2Lo) + 1;

			// Special case for V and I
			if (i == 4)
				width = std::min(productHi, r1Hi + r2Hi - iSize)
						- std::max(productLo, r1Lo + r2Lo - iSize) + 1;

			if (width <= 0)
				return false;
		}

		return true;
	}

	/**
	 * Determine if the reaction is possible given then reactants, product,
	 * and maximum size of I for trap mutation.
	 *
	 * @param r1 First reactant.
	 * @param r2 Second reactant.
	 * @param prod Potential product.
	 * @param prod Potential product.
	 */
	int checkOverlap(PSICluster& r1, PSICluster& r2, PSICluster& prod,
			int maxI) {
		// Initial declaration
		int iSizeToReturn = 0;

		// Loop on the possible I starting by the smallest
		for (int iSize = 1; iSize <= maxI; iSize++) {

			// Loop on the different type of clusters in grouping
			for (int i = 1; i < psDim; i++) {
				// Check the boundaries in all the directions
				auto const& bounds = prod.getBounds(indexList[i] - 1);
				int productLo = *(bounds.begin()), productHi = *(bounds.end())
						- 1;
				auto const& r1Bounds = r1.getBounds(indexList[i] - 1);
				int r1Lo = *(r1Bounds.begin()), r1Hi = *(r1Bounds.end()) - 1;
				auto const& r2Bounds = r2.getBounds(indexList[i] - 1);
				int r2Lo = *(r2Bounds.begin()), r2Hi = *(r2Bounds.end()) - 1;

				// Compute the corresponding overlap width
				int width = std::min(productHi, r1Hi + r2Hi)
						- std::max(productLo, r1Lo + r2Lo) + 1;

				// Special case for V and I
				if (indexList[i] == 4)
					width = std::min(productHi - iSize, r1Hi + r2Hi)
							- std::max(productLo - iSize, r1Lo + r2Lo) + 1;

				if (width <= 0)
					iSizeToReturn = -1;
			}

			// Check if it overlapped at this I size
			if (iSizeToReturn == 0)
				return iSize;
		}

		return iSizeToReturn;
	}

public:

	/**
	 * Default constructor, deleted to force construction using parameters.
	 */
	PSIClusterReactionNetwork() = delete;

	/**
	 * The Constructor
	 *
	 * @param registry The performance handler registry
	 */
	PSIClusterReactionNetwork(
			std::shared_ptr<xolotlPerf::IHandlerRegistry> registry);

	/**
	 * Copy constructor, deleted to prevent use.
	 */
	PSIClusterReactionNetwork(const PSIClusterReactionNetwork& other) = delete;

	/**
	 * Computes the full reaction connectivity matrix for this network.
	 */
	void createReactionConnectivity();

	/**
	 * This operation sets the temperature at which the reactants currently
	 * exists. It calls setTemperature() on each reactant.
	 *
	 * This is the simplest way to set the temperature for all reactants is to
	 * call the ReactionNetwork::setTemperature() operation.
	 * @param i The location on the grid in the depth direction
	 *
	 * @param temp The new temperature
	 */
	void setTemperature(double temp, int i) override;

	/**
	 * This operation reinitializes the network.
	 *
	 * It computes the cluster Ids and network size from the allReactants vector.
	 */
	void reinitializeNetwork() override;

	/**
	 * This method redefines the connectivities for each cluster in the
	 * allReactans vector.
	 */
	void reinitializeConnectivities() override;

	/**
	 * This operation updates the concentrations for all reactants in the
	 * network from an array.
	 *
	 * @param concentrations The array of doubles that will be for the
	 * concentrations. This operation does NOT create, destroy or resize the
	 * array. Properly aligning the array in memory so that this operation
	 * does not overrun is up to the caller.
	 */
	void updateConcentrationsFromArray(double * concentrations) override;

	/**
	 * This operation returns the number of super reactants in the network.
	 *
	 * @return The number of super reactants in the network
	 */
	int getSuperSize() const override {
		return getAll(ReactantType::PSISuper).size();
	}

	/**
	 * This operation returns the size or number of reactants and momentums in the network.
	 *
	 * @return The number of degrees of freedom
	 */
	virtual int getDOF() const override {
		return size() + (psDim - 1) * getAll(ReactantType::PSISuper).size() + 1;
	}

	/**
	 * This operation returns the list (vector) of each reactant in the network.
	 *
	 * @return The list of compositions
	 */
	virtual std::vector<std::vector<int> > getCompositionList() const override;

	/**
	 * Get the diagonal fill for the Jacobian, corresponding to the reactions.
	 *
	 * @param sfm Connectivity map.
	 */
	void getDiagonalFill(SparseFillMap& sfm) override;

	/**
	 * Get the total concentration of atoms contained in the network.
	 *
	 * Here the atoms that are considered are:
	 * 0 helium
	 * 1 deuterium
	 * 2 tritium
	 *
	 * @param i Index to switch between the different types of atoms
	 * @param minSize The minimum size to take into account
	 * @return The total concentration
	 */
	double getTotalAtomConcentration(int i = 0, int minSize = 0) override;

	/**
	 * Get the total concentration of atoms contained in bubbles in the network.
	 *
	 * Here the atoms that are considered are:
	 * 0 helium
	 * 1 deuterium
	 * 2 tritium
	 *
	 * @param i Index to switch between the different types of atoms
	 * @param minSize The minimum size to take into account
	 * @return The total concentration
	 */
	double getTotalTrappedAtomConcentration(int i = 0, int minSize = 0)
			override;

	/**
	 * Get the total concentration of vacancies contained in the network.
	 *
	 * @return The total concentration
	 */
	double getTotalVConcentration() override;

	/**
	 * Get the total concentration of tungsten interstitials in the network.
	 *
	 * @return The total concentration
	 */
	double getTotalIConcentration() override;

	/**
	 * Get the total concentration of bubbles in the network.
	 *
	 * @param minSize The minimum size to take into account
	 * @return The total concentration
	 */
	double getTotalBubbleConcentration(int minSize = 0)
			override;

	/**
	 * Compute the fluxes generated by all the reactions
	 * for all the clusters and their momentums.
	 *
	 * @param updatedConcOffset The pointer to the array of the concentration at the grid
	 * point where the fluxes are computed used to find the next solution
	 * @param i The location on the grid in the depth direction
	 */
	void computeAllFluxes(double *updatedConcOffset, int i) override;

	/**
	 * Compute the partial derivatives generated by all the reactions
	 * for all the clusters and their momentum.
	 *
	 * @param startingIdx Starting index of items owned by each reactant
	 *      within the partials values array and the indices array.
	 * @param indices The indices of the clusters for the partial derivatives.
	 * @param vals The values of partials for the reactions
	 * @param i The location on the grid in the depth direction
	 */
	void computeAllPartials(const std::vector<size_t>& startingIdx,
			const std::vector<long int>& indices, std::vector<double>& vals,
			int i) const override;

	/**
	 * Set the phase space to save time and memory
	 *
	 * @param dim The total dimension of the phase space
	 * @param list The list of indices that constitute the phase space
	 */
	void setPhaseSpace(int dim, Array<int, 5> list) {
		// Set the dimension
		psDim = dim;

		// Loop on the dimension to set the list
		for (int i = 0; i < 5; i++) {
			indexList[i] = list[i];
		}

		// Set the phase space in each reactant
		std::for_each(allReactants.begin(), allReactants.end(),
				[&dim,&list](IReactant& currReactant) {
					auto& currCluster = static_cast<PSICluster&>(currReactant);
					currCluster.setPhaseSpace(dim, list);
				});
	}

	/**
	 * This operation returns the phase space list needed to set up the grouping
	 * correctly in PSI.
	 *
	 * @return The phase space list
	 */
	virtual Array<int, 5> getPhaseSpaceList() const override {
		return indexList;
	}
};

} // namespace xolotlCore

#endif
