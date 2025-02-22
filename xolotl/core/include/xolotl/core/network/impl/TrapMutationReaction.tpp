#pragma once

namespace xolotl
{
namespace core
{
namespace network
{
template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
TrapMutationReaction<TNetwork, TDerived>::TrapMutationReaction(
	ReactionDataRef reactionData, ClusterDataRef clusterData,
	IndexType reactionId, IndexType cluster0, IndexType cluster1,
	IndexType cluster2) :
	Superclass(reactionData, clusterData, reactionId),
	_heClId(cluster0),
	_heVClId(cluster1),
	_iClId(cluster2)
{
	this->initialize();

	auto heVComp = clusterData.getCluster(_heVClId).getOriginComposition();
	_heAmount = heVComp[Species::He];
	_vSize = heVComp[Species::V];
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
TrapMutationReaction<TNetwork, TDerived>::TrapMutationReaction(
	ReactionDataRef reactionData, ClusterDataRef clusterData,
	IndexType reactionId, const detail::ClusterSet& clusterSet) :
	TrapMutationReaction(reactionData, clusterData, reactionId,
		clusterSet.cluster0, clusterSet.cluster1, clusterSet.cluster2)
{
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
TrapMutationReaction<TNetwork, TDerived>::computeRate(IndexType gridIndex)
{
	return 0.0;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
TrapMutationReaction<TNetwork, TDerived>::computeRate(double largestRate)
{
	const auto& desorp =
		this->_clusterData.extraData.trapMutationData.desorption();
	if (_heClId == desorp.id) {
		return (1.0 - desorp.portion) / desorp.portion;
	}

	// Multiply the biggest rate in the network by 1000.0
	// so that trap-mutation overcomes any other reaction
	return (1000.0 * largestRate);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
TrapMutationReaction<TNetwork, TDerived>::updateRates(double largestRate)
{
	auto rate = this->asDerived()->computeRate(largestRate);
	for (IndexType i = 0; i < this->_rate.extent(0); ++i) {
		this->_rate(i) = rate;
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
TrapMutationReaction<TNetwork, TDerived>::computeConnectivity(
	const Connectivity& connectivity)
{
	this->addConnectivity(_heClId, _heClId, connectivity);
	this->addConnectivity(_heVClId, _heClId, connectivity);
	this->addConnectivity(_iClId, _heClId, connectivity);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
TrapMutationReaction<TNetwork, TDerived>::computeReducedConnectivity(
	const Connectivity& connectivity)
{
	this->addConnectivity(_heClId, _heClId, connectivity);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
TrapMutationReaction<TNetwork, TDerived>::getAppliedRate(
	IndexType gridIndex) const
{
	double rate = this->_rate[gridIndex];
	auto& tmData = this->_clusterData.extraData.trapMutationData;
	if (_heClId == tmData.desorption().id) {
		rate *= tmData.currentDesorpLeftSideRate();
	}
	rate *= tmData.currentDisappearingRate();
	return rate;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
bool
TrapMutationReaction<TNetwork, TDerived>::getEnabled() const
{
	const auto& tmData = this->_clusterData.extraData.trapMutationData;
	return tmData.tmEnabled[_heAmount - 1] &&
		(tmData.tmVSizes[_heAmount - 1] == _vSize);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
TrapMutationReaction<TNetwork, TDerived>::computeFlux(
	ConcentrationsView concentrations, FluxesView fluxes, IndexType gridIndex)
{
	if (!getEnabled()) {
		return;
	}

	auto rate = getAppliedRate(gridIndex);
	auto f = rate * concentrations[_heClId];

	Kokkos::atomic_sub(&fluxes[_heClId], f);
	Kokkos::atomic_add(&fluxes[_heVClId], f);
	Kokkos::atomic_add(&fluxes[_iClId], f);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
TrapMutationReaction<TNetwork, TDerived>::computePartialDerivatives(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	Connectivity connectivity, IndexType gridIndex)
{
	if (!getEnabled()) {
		return;
	}

	auto rate = getAppliedRate(gridIndex);

	Kokkos::atomic_sub(&values(connectivity(_heClId, _heClId)), rate);
	Kokkos::atomic_add(&values(connectivity(_heVClId, _heClId)), rate);
	Kokkos::atomic_add(&values(connectivity(_iClId, _heClId)), rate);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
TrapMutationReaction<TNetwork, TDerived>::computeReducedPartialDerivatives(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	Connectivity connectivity, IndexType gridIndex)
{
	if (!getEnabled()) {
		return;
	}

	auto rate = getAppliedRate(gridIndex);
	Kokkos::atomic_sub(&values(connectivity(_heClId, _heClId)), rate);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
TrapMutationReaction<TNetwork, TDerived>::computeLeftSideRate(
	ConcentrationsView concentrations, IndexType clusterId, IndexType gridIndex)
{
	return 0.0;
}
} // namespace network
} // namespace core
} // namespace xolotl
