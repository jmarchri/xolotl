#pragma once

#include <xolotl/core/network/detail/PSITrapMutation.h>
#include <xolotl/core/network/detail/impl/SinkReactionGenerator.tpp>
#include <xolotl/core/network/detail/impl/TrapMutationClusterData.tpp>
#include <xolotl/core/network/detail/impl/TrapMutationReactionGenerator.tpp>
#include <xolotl/core/network/impl/PSIClusterGenerator.tpp>
#include <xolotl/core/network/impl/PSIReaction.tpp>
#include <xolotl/core/network/impl/ReactionNetwork.tpp>

namespace xolotl
{
namespace core
{
namespace network
{
template <typename TSpeciesEnum>
PSIReactionNetwork<TSpeciesEnum>::PSIReactionNetwork(const Subpaving& subpaving,
	IndexType gridSize, const options::IOptions& options) :
	Superclass(subpaving, gridSize, options),
	_tmHandler(psi::getTrapMutationHandler(
		this->_enableTrapMutation, options.getMaterial()))
{
}

template <typename TSpeciesEnum>
PSIReactionNetwork<TSpeciesEnum>::PSIReactionNetwork(
	const std::vector<AmountType>& maxSpeciesAmounts,
	const std::vector<SubdivisionRatio>& subdivisionRatios, IndexType gridSize,
	const options::IOptions& options) :
	Superclass(maxSpeciesAmounts, subdivisionRatios, gridSize, options),
	_tmHandler(psi::getTrapMutationHandler(
		this->_enableTrapMutation, options.getMaterial()))
{
}

template <typename TSpeciesEnum>
PSIReactionNetwork<TSpeciesEnum>::PSIReactionNetwork(
	const std::vector<AmountType>& maxSpeciesAmounts, IndexType gridSize,
	const options::IOptions& options) :
	Superclass(maxSpeciesAmounts, gridSize, options),
	_tmHandler(psi::getTrapMutationHandler(
		this->_enableTrapMutation, options.getMaterial()))
{
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::initializeExtraClusterData(
	const options::IOptions& options)
{
	if (!this->_enableTrapMutation) {
		return;
	}

	this->_clusterData.extraData.trapMutationData.initialize();
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::updateExtraClusterData(
	const std::vector<double>& gridTemps)
{
	if (!this->_enableTrapMutation) {
		return;
	}

	_tmHandler->updateData(gridTemps[0]);

	auto& tmData = this->_clusterData.extraData.trapMutationData;

	using Kokkos::HostSpace;
	using Kokkos::MemoryUnmanaged;

	auto desorpInit = _tmHandler->getDesorptionInitializer();
	Composition comp{};
	comp[Species::He] = desorpInit.size;
	auto desorp = create_mirror_view(tmData.desorption);
	desorp() = detail::Desorption{
		desorpInit, this->findCluster(comp, plsm::onHost).getId()};
	deep_copy(tmData.desorption, desorp);

	auto depths = Kokkos::View<const double[7], HostSpace, MemoryUnmanaged>(
		_tmHandler->getDepths().data());
	deep_copy(tmData.tmDepths, depths);

	auto vSizes = Kokkos::View<const AmountType[7], HostSpace, MemoryUnmanaged>(
		_tmHandler->getVacancySizes().data());
	deep_copy(tmData.tmVSizes, vSizes);
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::selectTrapMutationReactions(
	double depth, double spacing)
{
	auto& tmData = this->_clusterData.extraData.trapMutationData;
	auto depths = create_mirror_view(tmData.tmDepths);
	deep_copy(depths, tmData.tmDepths);
	auto enable = create_mirror_view(tmData.tmEnabled);
	for (std::size_t l = 0; l < depths.size(); ++l) {
		enable[l] = false;
		if (depths[l] == 0.0) {
			continue;
		}
		if (depths[l] < depth + 0.01 && depths[l] > depth - spacing - 0.01) {
			enable[l] = true;
		}
	}
	deep_copy(tmData.tmEnabled, enable);
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::computeFluxesPreProcess(
	ConcentrationsView concentrations, FluxesView fluxes, IndexType gridIndex,
	double surfaceDepth, double spacing)
{
	if (this->_enableTrapMutation) {
		updateDesorptionLeftSideRate(concentrations, gridIndex);
		selectTrapMutationReactions(surfaceDepth, spacing);
	}
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::computePartialsPreProcess(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	IndexType gridIndex, double surfaceDepth, double spacing)
{
	if (this->_enableTrapMutation) {
		updateDesorptionLeftSideRate(concentrations, gridIndex);
		selectTrapMutationReactions(surfaceDepth, spacing);
	}
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::updateBurstingConcs(
	double* gridPointSolution, double factor, std::vector<double>& nBurst)
{
	using detail::toIndex;

	// Loop on every cluster
	for (unsigned int i = 0; i < this->getNumClusters(); i++) {
		const auto& clReg = this->getCluster(i, plsm::onHost).getRegion();
		// Non-grouped clusters
		if (clReg.isSimplex()) {
			// Get the composition
			Composition comp = clReg.getOrigin();
			// Pure He, D, or T case
			if (comp.isOnAxis(Species::He)) {
				// Compute the number of atoms released
				nBurst[toIndex(Species::He)] +=
					gridPointSolution[i] * (double)comp[Species::He] * factor;
				// Reset concentration
				gridPointSolution[i] = 0.0;
				continue;
			}
			if constexpr (psi::hasDeuterium<Species>) {
				if (comp.isOnAxis(Species::D)) {
					// Compute the number of atoms released
					nBurst[toIndex(Species::D)] += gridPointSolution[i] *
						(double)comp[Species::D] * factor;
					// Reset concentration
					gridPointSolution[i] = 0.0;
					continue;
				}
			}
			if constexpr (psi::hasTritium<Species>) {
				if (comp.isOnAxis(Species::T)) {
					// Compute the number of atoms released
					nBurst[toIndex(Species::T)] += gridPointSolution[i] *
						(double)comp[Species::T] * factor;
					// Reset concentration
					gridPointSolution[i] = 0.0;
					continue;
				}
			}
			// Mixed cluster case
			if (!comp.isOnAxis(Species::V) && !comp.isOnAxis(Species::I)) {
				// Compute the number of atoms released
				nBurst[toIndex(Species::He)] +=
					gridPointSolution[i] * (double)comp[Species::He] * factor;
				if constexpr (psi::hasDeuterium<Species>) {
					nBurst[toIndex(Species::D)] += gridPointSolution[i] *
						(double)comp[Species::D] * factor;
				}
				if constexpr (psi::hasTritium<Species>) {
					nBurst[toIndex(Species::T)] += gridPointSolution[i] *
						(double)comp[Species::T] * factor;
				}
				// Transfer concentration to V of the same size
				Composition vComp = Composition::zero();
				vComp[Species::V] = comp[Species::V];
				auto vCluster = this->findCluster(vComp, plsm::onHost);
				gridPointSolution[vCluster.getId()] += gridPointSolution[i];
				gridPointSolution[i] = 0.0;

				continue;
			}
		}
		// Grouped clusters
		else {
			// Compute the number of atoms released
			double concFactor = clReg.volume() / clReg[Species::He].length();
			for (auto j : makeIntervalRange(clReg[Species::He])) {
				nBurst[toIndex(Species::He)] +=
					gridPointSolution[i] * (double)j * concFactor * factor;
			}
			if constexpr (psi::hasDeuterium<Species>) {
				concFactor = clReg.volume() / clReg[Species::D].length();
				for (auto j : makeIntervalRange(clReg[Species::D])) {
					nBurst[toIndex(Species::D)] +=
						gridPointSolution[i] * (double)j * concFactor * factor;
				}
			}
			if constexpr (psi::hasTritium<Species>) {
				concFactor = clReg.volume() / clReg[Species::T].length();
				for (auto j : makeIntervalRange(clReg[Species::T])) {
					nBurst[toIndex(Species::T)] +=
						gridPointSolution[i] * (double)j * concFactor * factor;
				}
			}

			// Get the factor
			concFactor = clReg.volume() / clReg[Species::V].length();
			// Loop on the Vs
			for (auto j : makeIntervalRange(clReg[Species::V])) {
				// Transfer concentration to V of the same size
				Composition vComp = Composition::zero();
				vComp[Species::V] = j;
				auto vCluster = this->findCluster(vComp, plsm::onHost);
				// TODO: refine formula with V moment
				gridPointSolution[vCluster.getId()] +=
					gridPointSolution[i] * concFactor;
			}

			// Reset the concentration and moments
			gridPointSolution[i] = 0.0;
			auto momentIds = this->getCluster(i, plsm::onHost).getMomentIds();
			for (std::size_t j = 0; j < momentIds.extent(0); j++) {
				if (momentIds(j) != this->invalidIndex())
					gridPointSolution[momentIds(j)] = 0.0;
			}
		}
	}
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::updateReactionRates()
{
	Superclass::updateReactionRates();

	using TrapMutationReactionType =
		typename Superclass::Traits::TrapMutationReactionType;
	// TODO: is this just the local largest rate? Is it correct?
	auto largestRate = this->getLargestRate();
	auto tmReactions =
		this->_reactions.template getView<TrapMutationReactionType>();
	Kokkos::parallel_for(
		tmReactions.size(), KOKKOS_LAMBDA(IndexType i) {
			tmReactions[i].updateRates(largestRate);
		});
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::updateTrapMutationDisappearingRate(
	double totalTrappedHeliumConc)
{
	// Set the rate to have an exponential decrease
	if (this->_enableAttenuation) {
		auto& tmData = this->_clusterData.extraData.trapMutationData;
		auto mirror = create_mirror_view(tmData.currentDisappearingRate);
		mirror() = exp(-4.0 * totalTrappedHeliumConc);
		deep_copy(tmData.currentDisappearingRate, mirror);
	}
}

template <typename TSpeciesEnum>
void
PSIReactionNetwork<TSpeciesEnum>::updateDesorptionLeftSideRate(
	ConcentrationsView concentrations, IndexType gridIndex)
{
	// TODO: Desorption is constant. So make it available on both host and
	// device. Either DualView or just direct value type that gets copied
	auto& tmData = this->_clusterData.extraData.trapMutationData;
	auto desorp = create_mirror_view(tmData.desorption);
	deep_copy(desorp, tmData.desorption);
	auto lsRate = create_mirror_view(tmData.currentDesorpLeftSideRate);
	lsRate() = this->getLeftSideRate(concentrations, desorp().id, gridIndex);
	deep_copy(tmData.currentDesorpLeftSideRate, lsRate);
}

template <typename TSpeciesEnum>
double
PSIReactionNetwork<TSpeciesEnum>::checkLatticeParameter(double latticeParameter)
{
	if (latticeParameter <= 0.0) {
		return tungstenLatticeConstant;
	}
	return latticeParameter;
}

template <typename TSpeciesEnum>
double
PSIReactionNetwork<TSpeciesEnum>::checkImpurityRadius(double impurityRadius)
{
	if (impurityRadius <= 0.0) {
		return heliumRadius;
	}
	return impurityRadius;
}

template <typename TSpeciesEnum>
typename PSIReactionNetwork<TSpeciesEnum>::IndexType
PSIReactionNetwork<TSpeciesEnum>::checkLargestClusterId()
{
	// Copy the cluster data for the parallel loop
	auto clData = typename PSIReactionNetwork<TSpeciesEnum>::ClusterDataRef(
		this->_clusterData);
	using Reducer = Kokkos::MaxLoc<PSIReactionNetwork<TSpeciesEnum>::AmountType,
		PSIReactionNetwork<TSpeciesEnum>::IndexType>;
	typename Reducer::value_type maxLoc;
	Kokkos::parallel_reduce(
		this->_numClusters,
		KOKKOS_LAMBDA(IndexType i, typename Reducer::value_type & update) {
			const auto& clReg = clData.getCluster(i).getRegion();
			Composition hi = clReg.getUpperLimitPoint();
			auto size = hi[Species::He] + hi[Species::V];
			if constexpr (psi::hasDeuterium<Species>) {
				size += hi[Species::D];
			}
			if constexpr (psi::hasTritium<Species>) {
				size += hi[Species::T];
			}
			if (size > update.val) {
				update.val = size;
				update.loc = i;
			}
		},
		Reducer(maxLoc));

	return maxLoc.loc;
}

namespace detail
{
template <typename TSpeciesEnum>
PSIReactionGenerator<TSpeciesEnum>::PSIReactionGenerator(
	const PSIReactionNetwork<TSpeciesEnum>& network) :
	Superclass(network)
{
	bool enableTrapMutation = network.getEnableTrapMutation();
	if (!enableTrapMutation) {
		return;
	}

	auto tmHandler =
		psi::getTrapMutationHandler(enableTrapMutation, network.getMaterial());
	auto vSizes = tmHandler->getAllVSizes();
	for (std::size_t n = 0; n < vSizes.size(); ++n) {
		if (vSizes[n].empty()) {
			continue;
		}
		auto nv = vSizes[n].size();
		_tmVSizes[n] = Kokkos::View<AmountType*>(
			"TM Vacancy Sizes - He" + std::to_string(n + 1), nv);
		auto vSizesMirror = Kokkos::View<AmountType*, Kokkos::HostSpace,
			Kokkos::MemoryUnmanaged>(vSizes[n].data(), nv);
		deep_copy(_tmVSizes[n], vSizesMirror);
	}
}

template <typename TSpeciesEnum>
template <typename TTag>
KOKKOS_INLINE_FUNCTION
void
PSIReactionGenerator<TSpeciesEnum>::operator()(
	IndexType i, IndexType j, TTag tag) const
{
	using Species = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;
	using AmountType = typename NetworkType::AmountType;

	constexpr auto species = NetworkType::getSpeciesRange();
	constexpr auto speciesNoI = NetworkType::getSpeciesRangeNoI();

	if (i == j) {
		addSinks(i, tag);
	}

	auto numClusters = this->getNumberOfClusters();

	// Get the composition of each cluster
	const auto& cl1Reg = this->getCluster(i).getRegion();
	const auto& cl2Reg = this->getCluster(j).getRegion();
	Composition lo1 = cl1Reg.getOrigin();
	Composition hi1 = cl1Reg.getUpperLimitPoint();
	Composition lo2 = cl2Reg.getOrigin();
	Composition hi2 = cl2Reg.getUpperLimitPoint();

	auto& subpaving = this->getSubpaving();
	auto previousIndex = subpaving.invalidIndex();

	// Special case for I + I
	if (lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::I)) {
		// Compute the composition of the new cluster
		auto minSize = lo1[Species::I] + lo2[Species::I];
		auto maxSize = hi1[Species::I] + hi2[Species::I] - 2;
		// Find the corresponding clusters
		for (auto k = minSize; k <= maxSize; k++) {
			Composition comp = Composition::zero();
			comp[Species::I] = k;
			auto iProdId = subpaving.findTileId(comp, plsm::onDevice);
			if (iProdId != subpaving.invalidIndex() &&
				iProdId != previousIndex) {
				this->addProductionReaction(tag, {i, j, iProdId});
				if (lo1[Species::I] == 1 || lo2[Species::I] == 1) {
					this->addDissociationReaction(tag, {iProdId, i, j});
				}
				previousIndex = iProdId;
			}
		}
		return;
	}

	// Special case for I + V
	if ((lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::V)) ||
		(lo1.isOnAxis(Species::V) && lo2.isOnAxis(Species::I))) {
		// Find out which one is which
		auto vSize =
			lo1.isOnAxis(Species::V) ? lo1[Species::V] : lo2[Species::V];
		// Int can be grouped
		auto iReg = lo1.isOnAxis(Species::I) ? cl1Reg : cl2Reg;
		for (auto k : makeIntervalRange(iReg[Species::I])) {
			// Compute the product size
			int prodSize = vSize - k;
			// 3 cases
			if (prodSize > 0) {
				// Looking for V cluster
				Composition comp = Composition::zero();
				comp[Species::V] = prodSize;
				auto vProdId = subpaving.findTileId(comp, plsm::onDevice);
				if (vProdId != subpaving.invalidIndex() &&
					vProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, vProdId});
					previousIndex = vProdId;
					// No dissociation
				}
			}
			else if (prodSize < 0) {
				// Looking for I cluster
				Composition comp = Composition::zero();
				comp[Species::I] = -prodSize;
				auto iProdId = subpaving.findTileId(comp, plsm::onDevice);
				if (iProdId != subpaving.invalidIndex() &&
					iProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, iProdId});
					previousIndex = iProdId;
					// No dissociation
				}
			}
			else {
				// No product
				this->addProductionReaction(tag, {i, j});
			}
		}
		return;
	}

	// General case
	constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
	using BoundsArray =
		Kokkos::Array<Kokkos::pair<AmountType, AmountType>, numSpeciesNoI>;
	plsm::EnumIndexed<BoundsArray, Species> bounds;
	// Loop on the species
	for (auto l : species) {
		auto low = lo1[l] + lo2[l];
		auto high = hi1[l] + hi2[l] - 2;
		// Special case for I
		if (l == Species::I) {
			bounds[Species::V].first -= high;
			bounds[Species::V].second -= low;
		}
		else {
			bounds[l] = {low, high};
		}
	}

	// Look for potential product
	IndexType nProd = 0;
	for (IndexType k = 0; k < numClusters; ++k) {
		// Get the composition
		const auto& prodReg = this->getCluster(k).getRegion();
		bool isGood = true;
		// Loop on the species
		// TODO: check l correspond to the same species in bounds and prod
		for (auto l : speciesNoI) {
			if (prodReg[l()].begin() > bounds[l()].second) {
				isGood = false;
				break;
			}
			if (prodReg[l()].end() - 1 < bounds[l()].first) {
				isGood = false;
				break;
			}
		}

		if (isGood) {
			// Increase nProd
			nProd++;
			this->addProductionReaction(tag, {i, j, k});
			// TODO: will have to add some rules, i or j should be a simplex
			// cluster of max size 1
			if (!cl1Reg.isSimplex() || !cl2Reg.isSimplex() ||
				!prodReg.isSimplex()) {
				continue;
			}
			// Loop on the species
			bool isOnAxis1 = false, isOnAxis2 = false;
			for (auto l : species) {
				if (lo1.isOnAxis(l()) && lo1[l()] == 1)
					isOnAxis1 = true;
				if (lo2.isOnAxis(l()) && lo2[l()] == 1)
					isOnAxis2 = true;
			}
			if (isOnAxis1 || isOnAxis2) {
				if (lo1.isOnAxis(Species::I) || lo2.isOnAxis(Species::I)) {
					continue;
				}

				this->addDissociationReaction(tag, {k, i, j});
			}
		}
	}

	// Modified Trap-Mutation
	if (this->_clusterData.enableTrapMutation()) {
		auto heAmt = lo1[Species::He];
		if (cl1Reg.isSimplex() && cl2Reg.isSimplex() && 1 <= heAmt &&
			heAmt <= 7) {
			Composition comp1 = Composition::zero();
			comp1[Species::He] = heAmt;
			auto comp2 = comp1;
			auto& vSizes = _tmVSizes[heAmt - 1];
			for (std::size_t n = 0; n < vSizes.extent(0); ++n) {
				auto& vSize = vSizes[n];
				comp2[Species::V] = vSize;
				if (lo1 == comp1 && lo2 == comp2) {
					Composition compI = Composition::zero();
					compI[Species::I] = vSize;
					auto iClusterId =
						subpaving.findTileId(compI, plsm::onDevice);
					this->addTrapMutationReaction(tag, {i, j, iClusterId});
				}
			}
		}
	}

	// Special case for trap-mutation
	if (nProd == 0) {
		// Look for larger clusters only if one of the reactant is pure He
		if (!(cl1Reg.isSimplex() && lo1.isOnAxis(Species::He)) &&
			!(cl2Reg.isSimplex() && lo2.isOnAxis(Species::He))) {
			return;
		}

		// Check that both reactants contain He
		if (cl1Reg[Species::He].begin() < 1 ||
			cl2Reg[Species::He].begin() < 1) {
			return;
		}

		// Loop on possible I sizes
		// TODO: get the correct value for maxISize
		AmountType maxISize = 6;
		for (AmountType n = 1; n <= maxISize; ++n) {
			// Find the corresponding cluster
			Composition comp = Composition::zero();
			comp[Species::I] = n;
			auto iClusterId = subpaving.findTileId(comp, plsm::onDevice);

			bounds[Species::V].first += 1;
			bounds[Species::V].second += 1;

			// Look for potential product
			IndexType nProd = 0;
			for (IndexType k = 0; k < numClusters; ++k) {
				// Get the composition
				const auto& prodReg = this->getCluster(k).getRegion();
				bool isGood = true;
				// Loop on the species
				// TODO: check l correspond to the same species in bounds
				// and prod
				for (auto l : speciesNoI) {
					if (prodReg[l()].begin() > bounds[l()].second) {
						isGood = false;
						break;
					}
					if (prodReg[l()].end() - 1 < bounds[l()].first) {
						isGood = false;
						break;
					}
				}

				if (isGood) {
					// Increase nProd
					nProd++;
					this->addProductionReaction(tag, {i, j, k, iClusterId});
					// No dissociation
				}
			}
			// Stop if we found a product
			if (nProd > 0) {
				break;
			}
		}
	}
}

template <typename TSpeciesEnum>
template <typename TTag>
KOKKOS_INLINE_FUNCTION
void
PSIReactionGenerator<TSpeciesEnum>::addSinks(IndexType i, TTag tag) const
{
	using Species = typename NetworkType::Species;
	using Composition = typename NetworkType::Composition;

	const auto& clReg = this->getCluster(i).getRegion();
	Composition lo = clReg.getOrigin();

	// I
	if (lo.isOnAxis(Species::I)) {
		this->addSinkReaction(tag, {i, NetworkType::invalidIndex()});
	}

	// V
	if (clReg.isSimplex() && lo.isOnAxis(Species::V)) {
		if (lo[Species::V] == 1)
			this->addSinkReaction(tag, {i, NetworkType::invalidIndex()});
	}
}

template <typename TSpeciesEnum>
inline ReactionCollection<
	typename PSIReactionGenerator<TSpeciesEnum>::NetworkType>
PSIReactionGenerator<TSpeciesEnum>::getReactionCollection() const
{
	ReactionCollection<NetworkType> ret(this->_clusterData.gridSize,
		this->getProductionReactions(), this->getDissociationReactions(),
		this->getSinkReactions(), this->getTrapMutationReactions());
	return ret;
}
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl
